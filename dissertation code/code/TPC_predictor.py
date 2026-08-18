#!/usr/bin/env python3
"""
TPC_predictor.py  --  Main prediction pipeline for microbial Temperature-Performance Curves.

Input:
    --fasta        Genome FASTA (nucleotide) or proteome FASTA (amino acid)
    --medium       JSON file mapping exchange-reaction IDs to uptake rates (for FBA)
    --temp_min/max/step  Temperature range to predict (default 5-80 step 1 C)
    --ogt          Override OGT (C). If omitted, predicted from ESM-2 embedding.
    --output       Output CSV path (default: tpc_prediction.csv)

Output:
    CSV with columns: temperature_C, norm_shape, abs_growth_rate (if FBA used)
    Plot:  tpc_prediction.png

Pipeline
--------
    FASTA
     |--[Prodigal: gene calling]---> protein seqs --> ESM-2 embedding (1280-dim)
                                              |
                              +--------------+---------------+
                              |                              |
                         OGT MLP --> OGT (C)      core_model (UDE/UTPC)
                              |                              |
                              +--------> normalised TPC shape (peak = 1)
                                              |
                          [optional] CarveMe GEM + COBRApy FBA --> peak growth rate
                                              |
                                    absolute TPC = shape x peak_rate

Usage examples
--------------
    # Minimal: normalised TPC only
    python code/TPC_predictor.py --fasta genome.fna

    # With FBA absolute scaling
    python code/TPC_predictor.py --fasta genome.fna --medium examples/example_medium_ecoli.json

    # Override OGT manually
    python code/TPC_predictor.py --fasta genome.fna --ogt 37.0
"""

import math, pickle, warnings, argparse, subprocess, json, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR      = Path(__file__).parent
REPO_DIR        = SCRIPT_DIR.parent
RESULTS_DIR     = REPO_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "core_model_checkpoint.pt"
SCALER_PATH     = RESULTS_DIR / "core_model_scaler.pkl"

# ============================================================
# Architecture constants (must match core_model.py)
# ============================================================
ATTN_DIM = 128; N_HEADS = 4; N_LAYERS = 1; DROPOUT = 0.1; Z_DIM = 64
P_MAX = 10.0; E_MIN, E_MAX = 3.0, 60.0; X_MIN = -60.0

# ============================================================
# Model architecture
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class ESMTempEncoder_MLP(nn.Module):
    def __init__(self, emb_len, attn_dim=ATTN_DIM, n_heads=N_HEADS,
                 n_layers=N_LAYERS, out_dim=Z_DIM, p_drop=DROPOUT,
                 n_patches=64, mlp_hidden=128, tfeat_dim=2):
        super().__init__()
        self.emb_len   = emb_len
        self.n_patches = n_patches
        self.patch_dim = emb_len // n_patches
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_dim, mlp_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, attn_dim)
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(tfeat_dim, attn_dim), nn.ReLU(), nn.Dropout(p_drop)
        )
        self.pos = PositionalEncoding(attn_dim, max_len=n_patches + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=n_heads, dim_feedforward=attn_dim * 2,
            dropout=p_drop, batch_first=False, norm_first=True, activation='gelu'
        )
        self.tx  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(attn_dim, out_dim)

    def forward(self, emb_vec, tfeat):
        B, L = emb_vec.shape
        x = self.patch_mlp(emb_vec.view(B, self.n_patches, self.patch_dim))
        temp_tok = self.temp_proj(tfeat).unsqueeze(1)
        x = torch.cat([temp_tok, x], dim=1).transpose(0, 1)
        return self.out(self.tx(self.pos(x))[0])


class ParamHead(nn.Module):
    def __init__(self, in_dim=Z_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, 2)
        )

    def forward(self, z):
        return self.net(z)


class ResidualMLP(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden=128, out_scale=1e-3, y_clip=50.0):
        super().__init__()
        self.y_clip = y_clip
        self.net = nn.Sequential(
            nn.Linear(2 + z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.out_scale = out_scale

    def forward(self, t_norm, y, z):
        y_clip = torch.clamp(y, -self.y_clip, self.y_clip)
        x = torch.cat([t_norm.view(-1, 1), y_clip.view(-1, 1), z], dim=1)
        return self.net(x).squeeze(1) * self.out_scale


class UTPC_ODEFunc_Constrained(nn.Module):
    def __init__(self, residual_net, hard_gate=True, detach_z=True):
        super().__init__()
        self.residual  = residual_net
        self.hard_gate = hard_gate
        self.detach_z  = detach_z
        self.params = self.z = self.t_mean = self.t_std = None

    def set_context(self, Pmax, ToptC, E, z, t_mean, t_std):
        self.params = (Pmax, ToptC, E)
        self.z, self.t_mean, self.t_std = z, t_mean, t_std

    def forward(self, tK, y):
        t_in = tK.view(1) if tK.dim() == 0 else tK
        y_in = y.view(1)  if y.dim()  == 0 else y
        Tc   = t_in - 273.15
        Pmax, ToptC, E = self.params
        x_val = (Tc - ToptC) / (E + 1e-8)
        x_eff = torch.clamp(x_val, min=X_MIN, max=1.0)
        dbase = -(Pmax / (E + 1e-8)) * torch.exp(x_eff) * x_eff
        t_norm   = (t_in - self.t_mean) / (self.t_std + 1e-8)
        z_use    = self.z.detach() if self.detach_z else self.z
        dres_raw = self.residual(t_norm, y_in, z_use.expand(t_in.size(0), -1))
        dres     = torch.where(Tc > ToptC, -F.softplus(dres_raw), dres_raw) \
                   if self.hard_gate else dres_raw
        dy = dbase + dres
        dy = torch.where((x_val > 1.0) & (y_in <= 0.0), torch.zeros_like(dy), dy)
        return dy


class UDEModel_Constrained(nn.Module):
    def __init__(self, encoder, head, residual, hard_gate=True, detach_z=True):
        super().__init__()
        self.encoder = encoder
        self.head    = head
        self.odefunc = UTPC_ODEFunc_Constrained(residual, hard_gate, detach_z)

    def forward_curve(self, emb_std, Tk_vec, ogtK, t_mean_k, t_std_k, device):
        if not torch.is_tensor(emb_std): emb_std = torch.tensor(emb_std, dtype=torch.float32, device=device)
        if not torch.is_tensor(Tk_vec):  Tk_vec  = torch.tensor(Tk_vec,  dtype=torch.float32, device=device)
        if not torch.is_tensor(ogtK):    ogtK    = torch.tensor(float(ogtK), dtype=torch.float32, device=device)

        tfeat = torch.zeros((1, 2), dtype=torch.float32, device=device)
        z     = self.encoder(emb_std.view(1, -1), tfeat)
        raw   = self.head(z).view(-1)
        Pmax  = P_MAX * torch.sigmoid(raw[0]) + 1e-6
        E     = E_MIN + (E_MAX - E_MIN) * torch.sigmoid(raw[1])
        ToptC = ogtK - 273.15
        self.odefunc.set_context(Pmax, ToptC, E, z, t_mean_k, t_std_k)

        Tc0 = Tk_vec[:1] - 273.15
        x0  = torch.clamp((Tc0 - ToptC) / (E + 1e-8), min=X_MIN, max=1.0)
        y0  = torch.clamp(Pmax * torch.exp(x0) * (1.0 - x0), min=0.0).view(-1)

        try:
            from torchdiffeq import odeint as td_odeint
            traj = td_odeint(self.odefunc, y0, Tk_vec, rtol=1e-5, atol=1e-6, method="dopri5")
        except Exception:
            y = y0; ys = [y.clone()]
            for i in range(1, len(Tk_vec)):
                h  = (Tk_vec[i] - Tk_vec[i-1]).to(y.dtype)
                ti = Tk_vec[i-1]
                k1 = self.odefunc(ti,         y)
                k2 = self.odefunc(ti + 0.5*h, y + 0.5*h*k1)
                k3 = self.odefunc(ti + 0.5*h, y + 0.5*h*k2)
                k4 = self.odefunc(ti + h,     y + h*k3)
                y  = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                ys.append(y.clone())
            traj = torch.stack(ys, dim=0)

        y_pred = torch.clamp(traj.squeeze(-1), min=0.0)
        return y_pred, (Pmax, ToptC, E)

# ============================================================
# Load model
# ============================================================

def load_model(checkpoint_path=CHECKPOINT_PATH, scaler_path=SCALER_PATH):
    """Load the trained core model from results/.

    Returns
    -------
    model, scaler, meta, device
        meta keys: t_mean_k, t_std_k, esm_cols, emb_len
    """
    checkpoint_path = Path(checkpoint_path)
    scaler_path     = Path(scaler_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python code/core_model.py  to train the model first.")
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "Run  python code/core_model.py  to train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    hp     = ckpt.get("hyperparams", {})

    emb_len   = ckpt["emb_len"]
    n_patches = ckpt["n_patches"]
    z_dim     = hp.get("Z_DIM", Z_DIM)
    y_clip    = hp.get("Y_CLIP", 50.0)
    hard_gate = hp.get("HARD_NO_INCREASE_AFTER_OGT", True)
    attn_dim  = hp.get("ATTN_DIM", ATTN_DIM)
    n_heads   = hp.get("N_HEADS",  N_HEADS)
    n_layers  = hp.get("N_LAYERS", N_LAYERS)
    dropout   = hp.get("DROPOUT",  DROPOUT)

    encoder  = ESMTempEncoder_MLP(emb_len=emb_len, attn_dim=attn_dim, n_heads=n_heads,
                                  n_layers=n_layers, out_dim=z_dim, p_drop=dropout,
                                  n_patches=n_patches)
    head     = ParamHead(in_dim=z_dim)
    residual = ResidualMLP(z_dim=z_dim, y_clip=y_clip)

    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    residual.load_state_dict(ckpt["residual"])

    model = UDEModel_Constrained(encoder, head, residual, hard_gate=hard_gate)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    meta = {
        "t_mean_k": torch.tensor(ckpt["t_mean_k"], dtype=torch.float32, device=device),
        "t_std_k":  torch.tensor(ckpt["t_std_k"],  dtype=torch.float32, device=device),
        "esm_cols": ckpt["esm_cols"],
        "emb_len":  emb_len,
    }

    print(f"Core model loaded (device={device}, ESM dim={emb_len})")
    return model, scaler, meta, device

# ============================================================
# ESM embedding extraction
# ============================================================

def read_fasta(fasta_path):
    """Read FASTA file, return list of (header, sequence) tuples."""
    seqs = []
    header = seq = ""
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header and seq:
                    seqs.append((header, seq))
                header = line[1:].split()[0]
                seq = ""
            else:
                seq += line
    if header and seq:
        seqs.append((header, seq))
    return seqs


def is_nucleotide_fasta(fasta_path):
    """Heuristic: if >80% of chars are ACGTN it is a nucleotide FASTA."""
    text = Path(fasta_path).read_text(errors="ignore").upper()
    non_header = "".join(l for l in text.splitlines() if not l.startswith(">"))
    if not non_header:
        return True
    dna_chars = sum(non_header.count(c) for c in "ACGTN")
    return (dna_chars / len(non_header)) > 0.80


def call_prodigal(genome_fasta, out_dir):
    """Run Prodigal to predict protein sequences from a genome FASTA.

    Requires prodigal to be installed and on PATH.
    Returns path to the output protein FASTA.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protein_fasta = out_dir / "proteins.faa"
    cmd = ["prodigal", "-i", str(genome_fasta), "-a", str(protein_fasta),
           "-p", "meta", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Prodigal failed:\n{result.stderr}")
    return protein_fasta


def extract_esm_embedding(fasta_path, tmp_dir=None, max_proteins=2000):
    """Compute mean-pooled ESM-2 genome embedding from a FASTA file.

    Parameters
    ----------
    fasta_path  : str or Path -- genome (nucleotide) or proteome (amino acid) FASTA
    tmp_dir     : directory for Prodigal intermediate files
    max_proteins: maximum number of proteins to embed (random subset if more)

    Returns
    -------
    numpy array of shape (1280,)

    Requirements
    ------------
    pip install fair-esm  (or transformers with facebook/esm2_t33_650M_UR50D)
    prodigal must be on PATH for nucleotide FASTA inputs
    """
    fasta_path = Path(fasta_path)
    if tmp_dir is None:
        tmp_dir = fasta_path.parent / "_tmp_esm"

    if is_nucleotide_fasta(fasta_path):
        print("[ESM] Nucleotide FASTA detected, running Prodigal for gene calling ...")
        protein_fasta = call_prodigal(fasta_path, tmp_dir)
    else:
        protein_fasta = fasta_path

    seqs = read_fasta(protein_fasta)
    if not seqs:
        raise ValueError(f"No sequences found in {protein_fasta}")

    if len(seqs) > max_proteins:
        import random as _rnd
        seqs = _rnd.sample(seqs, max_proteins)
        print(f"[ESM] Subsampled to {max_proteins} proteins")

    print(f"[ESM] Embedding {len(seqs)} proteins with ESM-2 ...")
    try:
        import esm as esm_lib
        model_esm, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model_esm.eval()
        embeddings = []
        batch_size = 8
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            _, _, tokens = batch_converter([(h, s[:1022]) for h, s in batch])
            with torch.no_grad():
                results = model_esm(tokens, repr_layers=[33])
            reps = results["representations"][33]
            for j in range(len(batch)):
                seq_len = len(batch[j][1][:1022])
                embeddings.append(reps[j, 1:seq_len+1, :].mean(0).numpy())
    except ImportError:
        try:
            from transformers import AutoTokenizer, EsmModel
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model_esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
            model_esm.eval()
            embeddings = []
            batch_size = 4
            for i in range(0, len(seqs), batch_size):
                batch_seqs = [s[:1022] for _, s in seqs[i:i+batch_size]]
                inputs = tokenizer(batch_seqs, return_tensors="pt",
                                   padding=True, truncation=True, max_length=1024)
                with torch.no_grad():
                    outputs = model_esm(**inputs)
                hidden = outputs.last_hidden_state
                mask   = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                embeddings.extend(pooled.numpy())
        except ImportError:
            raise ImportError(
                "ESM-2 library not found. Install with:\n"
                "  pip install fair-esm\n"
                "or\n"
                "  pip install transformers")

    genome_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    return genome_embedding

# ============================================================
# Core prediction
# ============================================================

def predict_shape(model, scaler, meta, device, esm_embedding, ogt_c, temperatures):
    """Predict normalised TPC shape for one organism.

    Parameters
    ----------
    esm_embedding : numpy array (emb_len,)
    ogt_c         : float, predicted or known OGT in degrees C
    temperatures  : array-like of temperatures in degrees C (ascending, >= 2 points)

    Returns
    -------
    dict with keys: pred_shape, Pmax, ToptC, E, temperatures
    """
    esm_embedding = np.asarray(esm_embedding, dtype=np.float32).ravel()
    temperatures  = np.asarray(temperatures,  dtype=np.float32)

    if esm_embedding.shape[0] != meta["emb_len"]:
        raise ValueError(
            f"ESM embedding dimension mismatch: got {esm_embedding.shape[0]}, "
            f"expected {meta['emb_len']}")
    if len(temperatures) < 2:
        raise ValueError("temperatures must have at least 2 points")

    emb_std  = scaler.transform(esm_embedding.reshape(1, -1))[0].astype(np.float32)
    Tk_sorted = np.sort(temperatures + 273.15).astype(np.float32)
    ogtK      = float(ogt_c) + 273.15

    with torch.no_grad():
        y_raw, (Pmax, ToptC, E) = model.forward_curve(
            emb_std, Tk_sorted, ogtK, meta["t_mean_k"], meta["t_std_k"], device)

    y_np = y_raw.cpu().numpy().astype(np.float32)
    peak = float(np.max(y_np))
    y_np = y_np / (peak if abs(peak) > 1e-8 else 1.0)

    return {
        "pred_shape":   y_np,
        "Pmax":         float(Pmax.item()),
        "ToptC":        float(ToptC.item()),
        "E":            float(E.item()),
        "temperatures": np.sort(temperatures),
    }

# ============================================================
# Full pipeline
# ============================================================

def run_pipeline(fasta_path, temperatures, ogt_c=None, medium=None,
                 checkpoint_path=CHECKPOINT_PATH, scaler_path=SCALER_PATH,
                 tmp_dir=None):
    """End-to-end TPC prediction.

    Parameters
    ----------
    fasta_path    : genome or proteome FASTA
    temperatures  : numpy array of temperatures in degrees C
    ogt_c         : optional OGT override in degrees C
    medium        : optional dict {exchange_rxn_id: uptake_rate} for FBA
    checkpoint_path / scaler_path : model file paths

    Returns
    -------
    dict with:
        temperatures   -- temperature array (C)
        norm_shape     -- normalised TPC (peak = 1)
        abs_growth_rate-- absolute growth rate (h-1) if FBA was run, else None
        ogt_c          -- OGT used (C)
        Pmax, E        -- UTPC parameters
    """
    model, scaler, meta, device = load_model(checkpoint_path, scaler_path)

    esm_emb = extract_esm_embedding(fasta_path, tmp_dir=tmp_dir)

    if ogt_c is None:
        print("[OGT] Predicting OGT from ESM-2 embedding ...")
        try:
            from OGT_predictor import predict_ogt_from_embedding
            ogt_c = predict_ogt_from_embedding(
                esm_emb,
                model_dir=RESULTS_DIR / "ogt_mlp",
            )
            print(f"[OGT] Predicted OGT = {ogt_c:.1f} C")
        except Exception as exc:
            print(f"[OGT] Warning: OGT prediction failed ({exc}). Defaulting to 37 C.")
            ogt_c = 37.0

    result = predict_shape(model, scaler, meta, device, esm_emb, ogt_c, temperatures)

    abs_rate = None
    if medium is not None:
        try:
            from FBA_anchor_point import get_peak_growth_rate
            print("[FBA] Running FBA to get absolute peak growth rate ...")
            abs_rate = get_peak_growth_rate(
                fasta_path=fasta_path,
                medium=medium,
                temperature_c=ogt_c,
                tmp_dir=tmp_dir
            )
            print(f"[FBA] Peak growth rate = {abs_rate:.4f} h-1")
        except Exception as exc:
            print(f"[FBA] Warning: FBA failed ({exc}). Output remains normalised.")

    return {
        "temperatures":    result["temperatures"],
        "norm_shape":      result["pred_shape"],
        "abs_growth_rate": result["norm_shape"] * abs_rate if abs_rate is not None else None,
        "ogt_c":           ogt_c,
        "Pmax":            result["Pmax"],
        "E":               result["E"],
    }

# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict microbial Temperature-Performance Curve from FASTA")
    parser.add_argument("--fasta",     required=True,  type=Path,
                        help="Genome FASTA (nucleotide) or proteome FASTA (amino acid)")
    parser.add_argument("--medium",    default=None,   type=Path,
                        help="JSON file with exchange reaction IDs and uptake rates for FBA")
    parser.add_argument("--ogt",       default=None,   type=float,
                        help="Override OGT (C). If omitted, predicted from genomic features.")
    parser.add_argument("--temp_min",  default=5.0,    type=float, help="Min temperature (C)")
    parser.add_argument("--temp_max",  default=80.0,   type=float, help="Max temperature (C)")
    parser.add_argument("--temp_step", default=1.0,    type=float, help="Temperature step (C)")
    parser.add_argument("--output",    default="tpc_prediction.csv", type=Path,
                        help="Output CSV file path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, type=Path)
    parser.add_argument("--scaler",     default=SCALER_PATH,     type=Path)
    parser.add_argument("--tmp_dir",    default=None,  type=Path,
                        help="Directory for intermediate files (Prodigal output etc.)")
    args = parser.parse_args()

    if not args.fasta.exists():
        sys.exit(f"FASTA file not found: {args.fasta}")

    temperatures = np.arange(args.temp_min, args.temp_max + 1e-9, args.temp_step,
                             dtype=np.float32)

    medium = None
    if args.medium is not None:
        if not args.medium.exists():
            sys.exit(f"Medium JSON not found: {args.medium}")
        with open(args.medium) as fh:
            medium_raw = json.load(fh)
        medium = {k: v for k, v in medium_raw.items() if not k.startswith("_")}

    res = run_pipeline(
        fasta_path      = args.fasta,
        temperatures    = temperatures,
        ogt_c           = args.ogt,
        medium          = medium,
        checkpoint_path = args.checkpoint,
        scaler_path     = args.scaler,
        tmp_dir         = args.tmp_dir,
    )

    out_df = pd.DataFrame({
        "temperature_C": res["temperatures"],
        "norm_shape":    res["norm_shape"],
    })
    if res["abs_growth_rate"] is not None:
        out_df["abs_growth_rate_per_h"] = res["abs_growth_rate"]

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(f"OGT used:   {res['ogt_c']:.1f} C")
    print(f"UTPC Pmax:  {res['Pmax']:.4f}")
    print(f"UTPC E:     {res['E']:.4f}")

    plot_path = out_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(8, 4))
    if res["abs_growth_rate"] is not None:
        ax.plot(res["temperatures"], res["abs_growth_rate"], color="steelblue",
                lw=2, label="Absolute growth rate (h-1)")
        ax.set_ylabel("Growth rate (h-1)")
    else:
        ax.plot(res["temperatures"], res["norm_shape"], color="steelblue",
                lw=2, label="Normalised growth rate")
        ax.set_ylabel("Normalised growth rate")
    ax.axvline(res["ogt_c"], color="tomato", ls="--", lw=1.5, label=f"OGT = {res['ogt_c']:.1f} C")
    ax.set_xlabel("Temperature (C)")
    ax.set_title(f"TPC prediction: {args.fasta.stem}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {plot_path}")
