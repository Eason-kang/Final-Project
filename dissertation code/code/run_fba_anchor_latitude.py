#!/usr/bin/env python3
"""Run FBA peak-growth anchors for latitude-assigned TPC species."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FP_CODE = ROOT
GROUP_ORDER = ["low_abs_latitude", "mid_abs_latitude", "high_abs_latitude"]


def normalize_path(value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return ROOT / path


def gunzip_fasta(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as inp:
        with dst.open("w", encoding="utf-8", newline="") as out:
            for line in inp:
                out.write(line)


def check_dependencies() -> list[str]:
    missing = []
    for package in ["cobra", "carveme"]:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def choose_records(records: pd.DataFrame, per_group: int, run_all: bool) -> pd.DataFrame:
    records = records.copy()
    for col in ["group_n_occurrences", "group_sum_relative_abundance", "fasta_record_count"]:
        records[col] = pd.to_numeric(records[col], errors="coerce").fillna(0)

    if run_all:
        chosen = records.sort_values(
            ["group_n_occurrences", "group_sum_relative_abundance", "fasta_record_count"],
            ascending=[False, False, False],
        )
    else:
        frames = []
        for group in GROUP_ORDER:
            sub = records[records["analysis_latitude_group"] == group].copy()
            sub = sub.sort_values(
                ["group_n_occurrences", "group_sum_relative_abundance", "fasta_record_count"],
                ascending=[False, False, False],
            ).head(per_group)
            frames.append(sub)
        chosen = pd.concat(frames, ignore_index=True)

    # FBA is species/proteome-level, so reconstruct each proteome once even if it
    # appears in multiple latitude groups.
    chosen_unique = (
        chosen.sort_values(
            ["group_n_occurrences", "group_sum_relative_abundance", "fasta_record_count"],
            ascending=[False, False, False],
        )
        .drop_duplicates("uniprot_proteome_id")
        .reset_index(drop=True)
    )
    return chosen, chosen_unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-records",
        type=Path,
        default=ROOT / "outputs/soil_latitude_main_80_20_pipeline/latitude_trait_analysis_80_20/analysis_records_with_tpc_traits.csv",
    )
    parser.add_argument(
        "--species-curves",
        type=Path,
        default=ROOT / "outputs/soil_latitude_main_80_20_pipeline/ogt_tpc_371_local/species_tpc_curves_long.csv",
    )
    parser.add_argument(
        "--medium",
        type=Path,
        default=FP_CODE / "examples/example_medium_ecoli.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/soil_latitude_main_80_20_pipeline/fba_anchor_latitude_pilot")
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run all unique proteomes in the analysis records.")
    parser.add_argument("--universe", default="bacteria")
    parser.add_argument(
        "--gapfill-medium",
        default="M9",
        help="Optional CarveMe medium used for gap filling. Use an empty string to disable.",
    )
    parser.add_argument(
        "--init-medium",
        default="M9",
        help="Optional CarveMe medium used to initialize model medium. Use an empty string to disable.",
    )
    parser.add_argument("--skip-if-missing-deps", action="store_true")
    args = parser.parse_args()

    missing = check_dependencies()
    if missing:
        message = (
            "Missing FBA dependencies: "
            + ", ".join(missing)
            + "\nInstall with the conda environment in hpc/env_fba.yml. "
            + "CarveMe also requires DIAMOND and a solver such as GLPK."
        )
        if args.skip_if_missing_deps:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "FBA_NOT_RUN.txt").write_text(message, encoding="utf-8")
            print(message)
            return
        raise SystemExit(message)

    sys.path.insert(0, str(FP_CODE / "code"))
    from FBA_anchor_point import get_peak_growth_rate  # noqa: E402

    records = pd.read_csv(args.analysis_records)
    curves = pd.read_csv(args.species_curves)
    chosen_records, chosen_unique = choose_records(records, args.per_group, args.all)

    with args.medium.open("r", encoding="utf-8") as handle:
        medium_raw = json.load(handle)
    medium = {k: v for k, v in medium_raw.items() if not k.startswith("_")}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "fba_peak_growth_rates.csv",
        "fba_failed_species.csv",
        "selected_analysis_records_with_fba.csv",
        "fba_scaled_absolute_tpcs_by_species.csv",
    ]:
        stale_path = args.out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    plain_dir = args.out_dir / "plain_fastas"
    gem_dir = args.out_dir / "gems"
    tmp_dir = args.out_dir / "tmp"
    selected_records_path = args.out_dir / "selected_fba_analysis_records.csv"
    chosen_records.to_csv(selected_records_path, index=False)

    rows = []
    error_rows = []
    for index, row in chosen_unique.iterrows():
        proteome_id = row["uniprot_proteome_id"]
        species = row["candidate_species"]
        gz_fasta = normalize_path(row["local_fasta_path"])
        plain_fasta = plain_dir / f"{proteome_id}.faa"
        gem_path = gem_dir / f"{proteome_id}_gem.xml"

        print(f"[{index + 1}/{len(chosen_unique)}] {proteome_id} {species}")
        try:
            if not gz_fasta.exists():
                raise FileNotFoundError(f"FASTA not found: {gz_fasta}")
            gunzip_fasta(gz_fasta, plain_fasta)

            rate = get_peak_growth_rate(
                fasta_path=plain_fasta,
                medium=medium,
                temperature_c=float(row["predicted_ogt_C"]),
                gem_path=gem_path if gem_path.exists() else None,
                universe=args.universe,
                gapfill_medium=args.gapfill_medium or None,
                init_medium=args.init_medium or None,
                tmp_dir=tmp_dir / proteome_id,
            )
            rows.append(
                {
                    "proteome_id": proteome_id,
                    "species": species,
                    "ncbi_taxid": row.get("ncbi_taxid", ""),
                    "uniprot_taxid": row.get("uniprot_taxid", ""),
                    "predicted_ogt_C": row["predicted_ogt_C"],
                    "tpc_peak_temp_C": row["tpc_peak_temp_C"],
                    "fba_peak_growth_rate_h_inv": rate,
                    "fba_status": "success",
                    "medium": str(args.medium),
                    "universe": args.universe,
                    "gapfill_medium": args.gapfill_medium,
                    "init_medium": args.init_medium,
                    "gem_path": str(gem_path),
                    "local_fasta_path": str(gz_fasta),
                }
            )
            pd.DataFrame(rows).to_csv(args.out_dir / "fba_peak_growth_rates.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[FBA skip] {proteome_id} {species}: {type(exc).__name__}: {exc}")
            error_rows.append(
                {
                    "proteome_id": proteome_id,
                    "species": species,
                    "ncbi_taxid": row.get("ncbi_taxid", ""),
                    "uniprot_taxid": row.get("uniprot_taxid", ""),
                    "predicted_ogt_C": row.get("predicted_ogt_C", ""),
                    "tpc_peak_temp_C": row.get("tpc_peak_temp_C", ""),
                    "fba_status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "medium": str(args.medium),
                    "universe": args.universe,
                    "gapfill_medium": args.gapfill_medium,
                    "init_medium": args.init_medium,
                    "gem_path": str(gem_path),
                    "local_fasta_path": str(gz_fasta),
                }
            )
            pd.DataFrame(error_rows).to_csv(args.out_dir / "fba_failed_species.csv", index=False)
            continue

    rates = pd.DataFrame(rows)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(args.out_dir / "fba_failed_species.csv", index=False)
    if rates.empty:
        print(f"FBA unique proteomes=0 selected_records={len(chosen_records)} out_dir={args.out_dir}")
        return
    chosen_with_rates = chosen_records.merge(
        rates[["proteome_id", "fba_peak_growth_rate_h_inv"]],
        left_on="uniprot_proteome_id",
        right_on="proteome_id",
        how="left",
    )
    chosen_with_rates.to_csv(args.out_dir / "selected_analysis_records_with_fba.csv", index=False)

    scaled_curves = curves.merge(rates[["proteome_id", "fba_peak_growth_rate_h_inv"]], on="proteome_id", how="inner")
    scaled_curves["absolute_growth_rate_h_inv"] = scaled_curves["norm_shape"] * scaled_curves["fba_peak_growth_rate_h_inv"]
    scaled_curves.to_csv(args.out_dir / "fba_scaled_absolute_tpcs_by_species.csv", index=False)

    print(f"FBA unique proteomes={len(rates)} selected_records={len(chosen_records)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
