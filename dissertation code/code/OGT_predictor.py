#!/usr/bin/env python3
"""
OGT_predictor.py  --  Train and apply the proteome-feature OGT MLP.

Predicts Optimal Growth Temperature (OGT) from a proteome FASTA file by
extracting 425 amino-acid-sequence-derived features and passing them through
a multilayer perceptron (MLP) regressor.

Feature groups (425 total)
--------------------------
    Raw amino acid fractions (20 AA)                         20
    Proteome properties (mean length, charge, IVYWREL ratios) 5
    Dipeptide frequencies (20x20)                           400
    ---------------------------------------------------------
    Total                                                   425

All features are derived purely from the amino acid sequences in the
proteome FASTA — no genome sequence, rRNA, tRNA, or codon data required.

Modes
-----
    Train from pre-computed feature CSVs (protein-compatible columns only):
        python code/OGT_predictor.py train \\
            --bacteria_csv data/calculated_features_bacteria.csv \\
            --archaea_csv  data/calculated_features_archaea.csv

    Predict OGT from a proteome FASTA:
        python code/OGT_predictor.py predict --fasta proteome.faa

    Predict from a pre-computed feature CSV:
        python code/OGT_predictor.py predict --feature_csv features.csv
"""

import sys, json, argparse, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR  = Path(__file__).parent
REPO_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = REPO_DIR / "results" / "ogt_mlp"
TRAIN_DIR   = REPO_DIR / "Train"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

MLP_PATH          = RESULTS_DIR / "mlp.pkl"
SCALER_PATH       = RESULTS_DIR / "scaler.pkl"
FEATURE_COLS_PATH = RESULTS_DIR / "feature_cols.pkl"
CV_RESULTS_PATH   = RESULTS_DIR / "cv_results.json"

# Columns in the training CSV that are NOT features
META_COLS = {
    'Unnamed: 0', 'domain', 'phylum', 'class', 'order',
    'family', 'genus', 'species', 'OGT'
}

# Columns derived from genome sequence / nucleotide data — not usable
# when only a protein FASTA is provided.  These are excluded at training time
# so the saved feature_cols.pkl lists only the 425 protein-level features.
_GENOME_FEATURE_PATTERNS = (
    'genome_size',       # nucleotide genome length
    '5S_',               # 5S rRNA composition
    '16S_',              # 16S rRNA composition
    '23S_',              # 23S rRNA composition
    'tRNA_',             # tRNA composition
    'trna_',             # tRNA MFE
    'G_fraction',        # genome GC content
)
_EXCLUDE_SUBSTRINGS = (
    '_frac_normed',      # AA fraction / GC — requires genome GC
    '_ORF_',             # codon usage in CDS — requires nucleotide CDS
)

# ============================================================
# MLP hyperparameters  (mirrors Chapter2/OGT/Code/train_ogt_mlp.py)
# ============================================================
MLP_PARAMS = dict(
    hidden_layer_sizes  = (256, 128, 64),
    activation          = 'relu',
    solver              = 'adam',
    alpha               = 1e-4,
    learning_rate       = 'adaptive',
    learning_rate_init  = 1e-3,
    max_iter            = 500,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,
    batch_size          = 64,
    random_state        = 42,
)

# ============================================================
# Feature constants
# ============================================================
AA_ALPHABET = list('ACDEFGHIKLMNPQRSTVWY')   # 20 standard AAs, sorted

DIPEPTIDE_COLS = [f'Pro_{a}{b}' for a in AA_ALPHABET for b in AA_ALPHABET]  # 400


def _is_protein_feature(col):
    """Return True if col is a protein-derivable feature (not genome-level)."""
    if col in META_COLS:
        return False
    if any(col.startswith(p) for p in _GENOME_FEATURE_PATTERNS):
        return False
    if any(s in col for s in _EXCLUDE_SUBSTRINGS):
        return False
    return True


# ============================================================
# Training
# ============================================================

def _load_training_data(bacteria_csv, archaea_csv):
    """Merge the two feature CSVs; keep only 425 protein-level features."""
    frames = []
    for path, label in [(bacteria_csv, 'bacteria'), (archaea_csv, 'archaea')]:
        df = pd.read_csv(path)
        df['_domain'] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if 'OGT' not in df.columns:
        raise ValueError("'OGT' column not found in training CSV")

    feature_cols = [c for c in df.columns if _is_protein_feature(c) and c != '_domain']
    df['OGT'] = pd.to_numeric(df['OGT'], errors='coerce')
    df = df.dropna(subset=['OGT']).reset_index(drop=True)

    X = df[feature_cols].fillna(0.0).values.astype(np.float64)
    y = df['OGT'].values.astype(np.float64)

    print(f"Training data: {len(df)} samples "
          f"(Bacteria: {(df['_domain']=='bacteria').sum()}, "
          f"Archaea: {(df['_domain']=='archaea').sum()})")
    print(f"Features: {len(feature_cols)}  |  OGT range: [{y.min():.1f}, {y.max():.1f}] C")
    return X, y, feature_cols


def train_ogt_mlp(bacteria_csv, archaea_csv):
    """10-fold CV then final model on all data.

    Saves results/ogt_mlp/mlp.pkl, scaler.pkl, feature_cols.pkl, cv_results.json
    and Train/ogt_mlp_cv_log.csv.
    """
    X, y, feature_cols = _load_training_data(bacteria_csv, archaea_csv)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    all_preds = np.zeros(len(y))

    print("\n10-fold cross-validation:")
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        scaler = StandardScaler().fit(X[tr])
        mlp    = MLPRegressor(**MLP_PARAMS)
        mlp.fit(scaler.transform(X[tr]), y[tr])
        y_pred = mlp.predict(scaler.transform(X[va]))
        all_preds[va] = y_pred

        rmse = float(np.sqrt(mean_squared_error(y[va], y_pred)))
        mae  = float(mean_absolute_error(y[va], y_pred))
        r2   = float(r2_score(y[va], y_pred))
        fold_results.append({'fold': fold, 'rmse': round(rmse, 4),
                             'mae': round(mae, 4), 'r2': round(r2, 4),
                             'n_train': len(tr), 'n_test': len(va)})
        print(f"  Fold {fold:2d}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    overall_rmse = float(np.sqrt(mean_squared_error(y, all_preds)))
    overall_mae  = float(mean_absolute_error(y, all_preds))
    overall_r2   = float(r2_score(y, all_preds))
    print(f"\nOverall CV: RMSE={overall_rmse:.4f}  MAE={overall_mae:.4f}  R2={overall_r2:.4f}")

    cv_summary = {
        'n_samples':   int(len(y)),
        'n_features':  int(len(feature_cols)),
        'overall':     {'rmse': round(overall_rmse, 4),
                        'mae':  round(overall_mae, 4),
                        'r2':   round(overall_r2, 4)},
        'folds':       fold_results,
        'mlp_params':  MLP_PARAMS,
    }
    with open(CV_RESULTS_PATH, 'w') as fh:
        json.dump(cv_summary, fh, indent=2)
    pd.DataFrame(fold_results).to_csv(TRAIN_DIR / 'ogt_mlp_cv_log.csv', index=False)
    print(f"CV results saved: {CV_RESULTS_PATH}")

    print("\nFitting final model on all data ...")
    final_scaler = StandardScaler().fit(X)
    final_mlp    = MLPRegressor(**MLP_PARAMS)
    final_mlp.fit(final_scaler.transform(X), y)

    with open(MLP_PATH,          'wb') as fh: pickle.dump(final_mlp,    fh)
    with open(SCALER_PATH,       'wb') as fh: pickle.dump(final_scaler, fh)
    with open(FEATURE_COLS_PATH, 'wb') as fh: pickle.dump(feature_cols, fh)
    print(f"Saved: {MLP_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {FEATURE_COLS_PATH}")


# ============================================================
# Load saved model
# ============================================================

def load_ogt_model(model_dir=None):
    """Load trained OGT MLP, scaler, and feature column list."""
    d = Path(model_dir) if model_dir else RESULTS_DIR
    with open(d / 'mlp.pkl',          'rb') as fh: mlp  = pickle.load(fh)
    with open(d / 'scaler.pkl',        'rb') as fh: scl  = pickle.load(fh)
    with open(d / 'feature_cols.pkl',  'rb') as fh: cols = pickle.load(fh)
    return mlp, scl, cols


# ============================================================
# Feature extraction from protein FASTA
# ============================================================

def _read_fasta(path):
    seqs = []
    h = s = ''
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if h and s: seqs.append((h, s))
                h = line[1:].split()[0]; s = ''
            else:
                s += line
    if h and s: seqs.append((h, s))
    return seqs


def extract_protein_features(protein_fasta):
    """Extract 425 OGT-predictive features from a proteome FASTA.

    Parameters
    ----------
    protein_fasta : str or Path  -- amino acid proteome FASTA

    Returns
    -------
    dict mapping feature_name -> float
    """
    seqs_raw  = _read_fasta(protein_fasta)
    prot_seqs = [s.rstrip('*').upper() for _, s in seqs_raw]

    if not prot_seqs:
        raise ValueError(f"No sequences found in {protein_fasta}")

    concat_aa = ''.join(prot_seqs)
    n_aa = max(1, len(concat_aa))

    feats = {}

    # Raw AA fractions (20)
    aa_raw = {aa: concat_aa.count(aa) / n_aa for aa in AA_ALPHABET}
    for aa in AA_ALPHABET:
        feats[f'Pro_{aa}'] = aa_raw[aa]

    # Proteome aggregate properties (5)
    feats['Pro_mean_length']       = float(np.mean([len(s) for s in prot_seqs]))
    feats['Pro_polar_charged']     = sum(aa_raw[a] for a in 'DEKRH')
    feats['Pro_polar_hydrophobic'] = sum(aa_raw[a] for a in 'AVFILMWP')
    q  = aa_raw['Q'] or 1e-9
    qh = (aa_raw['Q'] + aa_raw['H']) or 1e-9
    feats['Pro_LK/Q']  = (aa_raw['L'] + aa_raw['K']) / q
    feats['Pro_EK/QH'] = (aa_raw['E'] + aa_raw['K']) / qh

    # Dipeptide frequencies (400)
    dp_counts = {f'Pro_{a}{b}': 0 for a in AA_ALPHABET for b in AA_ALPHABET}
    dp_total  = 0
    for seq in prot_seqs:
        for i in range(len(seq) - 1):
            dp = f'Pro_{seq[i]}{seq[i+1]}'
            if dp in dp_counts:
                dp_counts[dp] += 1
                dp_total += 1
    n_dp = max(1, dp_total)
    for k in dp_counts:
        feats[k] = dp_counts[k] / n_dp

    return feats


# ============================================================
# Prediction helpers
# ============================================================

def _features_to_vector(feat_dict, feature_cols):
    """Arrange feature dict into a numpy array matching feature_cols order."""
    return np.array([feat_dict.get(c, 0.0) for c in feature_cols],
                    dtype=np.float64).reshape(1, -1)


def predict_ogt_from_features(feat_dict, model_dir=None):
    """Predict OGT from a pre-computed feature dict."""
    mlp, scl, cols = load_ogt_model(model_dir)
    X = _features_to_vector(feat_dict, cols)
    return float(mlp.predict(scl.transform(X))[0])


def predict_ogt_from_fasta(fasta_path, model_dir=None):
    """Extract protein features from a proteome FASTA and predict OGT.

    Parameters
    ----------
    fasta_path : str or Path -- proteome FASTA (amino acid sequences)
    model_dir  : path to results/ogt_mlp/ (uses default if None)

    Returns
    -------
    float -- predicted OGT in degrees C
    """
    feats = extract_protein_features(fasta_path)
    return predict_ogt_from_features(feats, model_dir=model_dir)


def predict_ogt_from_csv(feature_csv, model_dir=None):
    """Batch-predict OGT from a pre-computed feature CSV."""
    mlp, scl, cols = load_ogt_model(model_dir)
    df = pd.read_csv(feature_csv)
    X = df[[c for c in cols if c in df.columns]].fillna(0.0).values.astype(np.float64)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} feature columns missing — filled with 0")
    df['predicted_OGT_C'] = mlp.predict(scl.transform(X))
    return df


# ============================================================
# CLI entry point
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OGT predictor: train or predict from proteome sequence features')
    sub = parser.add_subparsers(dest='mode', required=True)

    train_p = sub.add_parser('train', help='Train OGT MLP from feature CSVs')
    train_p.add_argument('--bacteria_csv', required=True, type=Path,
                         help='CSV with genomic features for bacteria (protein-compatible columns will be selected)')
    train_p.add_argument('--archaea_csv',  required=True, type=Path,
                         help='CSV with genomic features for archaea (protein-compatible columns will be selected)')

    pred_p = sub.add_parser('predict', help='Predict OGT from proteome FASTA or feature CSV')
    grp = pred_p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--fasta',       type=Path, help='Proteome FASTA file (amino acid sequences)')
    grp.add_argument('--feature_csv', type=Path, help='Pre-computed 425-feature CSV')
    pred_p.add_argument('--model_dir', type=Path, default=RESULTS_DIR)
    pred_p.add_argument('--output',    type=Path, default=None,
                         help='Output CSV (batch mode only)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_ogt_mlp(args.bacteria_csv, args.archaea_csv)

    elif args.mode == 'predict':
        if args.fasta:
            ogt = predict_ogt_from_fasta(args.fasta, model_dir=args.model_dir)
            print(f"Predicted OGT: {ogt:.1f} C")
        else:
            df_out = predict_ogt_from_csv(args.feature_csv, model_dir=args.model_dir)
            out = args.output or Path(args.feature_csv).with_suffix('_ogt_pred.csv')
            df_out.to_csv(out, index=False)
            print(f"Predictions saved to: {out}")
            print(df_out[['predicted_OGT_C']].describe())
