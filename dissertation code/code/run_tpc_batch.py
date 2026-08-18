#!/usr/bin/env python3
"""Batch OGT + normalized TPC prediction using the updated code bundle."""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def add_code_path(fp_code: Path) -> None:
    sys.path.insert(0, str(fp_code / "code"))


def gunzip_fasta(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as inp:
        with dst.open("w", encoding="utf-8", newline="") as out:
            for line in inp:
                out.write(line)


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_outputs(out_dir: Path, summary_rows: list[dict], curve_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "species_ogt_tpc_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / "species_tpc_curves_long.csv", index=False)


def parse_optional_float(value: object) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def make_plots(out_dir: Path, summary_rows: list[dict], curve_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(summary_rows)
    curves = pd.DataFrame(curve_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for proteome_id, grp in curves.groupby("proteome_id"):
        ax.plot(grp["temperature_C"], grp["norm_shape"], lw=0.8, alpha=0.35)
    ax.set_xlim(float(curves["temperature_C"].min()), float(curves["temperature_C"].max()))
    ax.set_ylim(0.0, 1.05)
    ax.margins(x=0, y=0)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Normalized predicted growth")
    ax.set_title("Predicted normalized TPCs across selected species")
    fig.tight_layout()
    fig.savefig(out_dir / "all_species_normalized_tpc_overlay.png", dpi=220)
    plt.close(fig)

    summary["abundance_weighted_absolute_latitude"] = pd.to_numeric(
        summary.get("abundance_weighted_absolute_latitude"), errors="coerce"
    )
    lat_summary = summary.dropna(subset=["abundance_weighted_absolute_latitude"])
    if not lat_summary.empty:
        for y_col, y_label, file_name in [
            ("predicted_ogt_C", "Predicted OGT (C)", "predicted_ogt_vs_absolute_latitude.png"),
            ("tpc_peak_temp_C", "Predicted TPC peak temperature (C)", "predicted_tpc_peak_vs_absolute_latitude.png"),
            ("E", "Predicted activation energy proxy E", "predicted_E_vs_absolute_latitude.png"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                lat_summary["abundance_weighted_absolute_latitude"],
                lat_summary[y_col],
                s=18,
                alpha=0.75,
            )
            ax.set_xlim(
                float(lat_summary["abundance_weighted_absolute_latitude"].min()),
                float(lat_summary["abundance_weighted_absolute_latitude"].max()),
            )
            ax.margins(x=0, y=0.05)
            ax.set_xlabel("Abundance-weighted absolute latitude (degrees)")
            ax.set_ylabel(y_label)
            ax.set_title(y_label + " vs. absolute latitude")
            fig.tight_layout()
            fig.savefig(out_dir / file_name, dpi=220)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/top150_temp_gradient_species/09_strict_fasta_ge20_species_manifest.csv",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "outputs/strict148_esm2_t33_full_unique_gpu/species_embeddings.tsv",
    )
    parser.add_argument("--fp-code", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/updated_tpc_batch_148")
    parser.add_argument("--temp-min", type=float, default=5.0)
    parser.add_argument("--temp-max", type=float, default=80.0)
    parser.add_argument("--temp-step", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    add_code_path(args.fp_code)
    from OGT_predictor import extract_protein_features, load_ogt_model  # noqa: E402
    from TPC_predictor import load_model, predict_shape  # noqa: E402

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    emb_df = pd.read_csv(args.embeddings, sep="\t")
    z_cols = sorted([c for c in emb_df.columns if c.startswith("z_")])
    emb_by_proteome = {row["proteome_id"]: row[z_cols].to_numpy(dtype=np.float32) for _, row in emb_df.iterrows()}

    ogt_mlp, ogt_scaler, ogt_cols = load_ogt_model(args.fp_code / "results/ogt_mlp")
    tpc_model, tpc_scaler, tpc_meta, device = load_model(
        checkpoint_path=args.fp_code / "results/core_model_checkpoint.pt",
        scaler_path=args.fp_code / "results/core_model_scaler.pkl",
    )
    temperatures = np.arange(args.temp_min, args.temp_max, args.temp_step, dtype=np.float32)

    plain_dir = args.out_dir / "plain_fastas"
    summary_rows: list[dict] = []
    curve_rows: list[dict] = []

    for index, row in enumerate(rows, start=1):
        proteome_id = row["uniprot_proteome_id"]
        species = row["candidate_species"]
        if proteome_id not in emb_by_proteome:
            print(f"[skip] missing embedding: {proteome_id} {species}")
            continue

        gz_fasta = ROOT / row["local_fasta_path"]
        plain_fasta = plain_dir / f"{proteome_id}.faa"
        gunzip_fasta(gz_fasta, plain_fasta)

        features = extract_protein_features(plain_fasta)
        X = np.array([features.get(col, 0.0) for col in ogt_cols], dtype=np.float64).reshape(1, -1)
        ogt_c = float(ogt_mlp.predict(ogt_scaler.transform(X))[0])

        result = predict_shape(
            tpc_model,
            tpc_scaler,
            tpc_meta,
            device,
            emb_by_proteome[proteome_id],
            ogt_c,
            temperatures,
        )
        shape = result["pred_shape"]
        tpc_peak_temp = float(result["temperatures"][int(np.argmax(shape))])

        summary = {
            "rank": index,
            "proteome_id": proteome_id,
            "species": species,
            "ncbi_taxid": row.get("ncbi_taxid", ""),
            "uniprot_taxid": row.get("uniprot_taxid", ""),
            "primary_latitude_group": row.get("primary_latitude_group", ""),
            "latitude_groups_observed": row.get("latitude_groups_observed", ""),
            "mean_absolute_latitude": row.get("mean_absolute_latitude", ""),
            "min_absolute_latitude": row.get("min_absolute_latitude", ""),
            "max_absolute_latitude": row.get("max_absolute_latitude", ""),
            "abundance_weighted_absolute_latitude": row.get("abundance_weighted_absolute_latitude", ""),
            "fasta_record_count": row.get("fasta_record_count", ""),
            "predicted_ogt_C": ogt_c,
            "tpc_peak_temp_C": tpc_peak_temp,
            "Pmax": result["Pmax"],
            "E": result["E"],
            "embedding_dim": len(emb_by_proteome[proteome_id]),
        }
        summary_rows.append(summary)

        for temp_c, norm_shape in zip(result["temperatures"], shape):
            curve_rows.append(
                {
                    "proteome_id": proteome_id,
                    "species": species,
                    "temperature_C": float(temp_c),
                    "norm_shape": float(norm_shape),
                    "predicted_ogt_C": ogt_c,
                    "primary_latitude_group": row.get("primary_latitude_group", ""),
                    "abundance_weighted_absolute_latitude": row.get("abundance_weighted_absolute_latitude", ""),
                }
            )

        if index % 10 == 0 or index == len(rows):
            write_outputs(args.out_dir, summary_rows, curve_rows)
            print(f"[progress] {index}/{len(rows)} species")

    write_outputs(args.out_dir, summary_rows, curve_rows)
    make_plots(args.out_dir, summary_rows, curve_rows)
    print(f"DONE species={len(summary_rows)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
