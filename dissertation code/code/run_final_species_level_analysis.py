#!/usr/bin/env python3
"""Final species-level analyses for the soil latitude TPC thesis.

The thermal prediction is one per unique proteome/species, so the primary
inference also uses one row per species rather than repeated
species-latitude occurrence records.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TPC_SUMMARY = ROOT / "outputs/soil_latitude_main_80_20_pipeline/ogt_tpc_371_local/species_ogt_tpc_summary.csv"
DEFAULT_SCHOOLFIELD = ROOT / (
    "outputs/soil_latitude_main_80_20_pipeline/"
    "schoolfield_fba_trait_analysis/schoolfield_fitted_traits_by_species.csv"
)
DEFAULT_OUT = ROOT / "outputs/soil_latitude_main_80_20_pipeline/final_species_level_analysis"


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def genus_from_species(name: object) -> str:
    text = str(name or "").strip()
    parts = text.split()
    if not parts:
        return "unknown"
    if parts[0].lower() == "candidatus" and len(parts) > 1:
        return "Candidatus " + parts[1]
    return parts[0]


def ols_hc3(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    n = len(y)
    if n < 3 or len(np.unique(x)) < 2:
        return {
            "n": n,
            "slope": np.nan,
            "intercept": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "hc3_se": np.nan,
            "classic_se": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "contrast_90_minus_10": np.nan,
            "contrast_ci_low": np.nan,
            "contrast_ci_high": np.nan,
        }

    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Classical standard error is kept for audit; HC3 is reported in the thesis.
    sigma2 = ss_res / max(n - 2, 1)
    classic_cov = sigma2 * xtx_inv
    classic_se = math.sqrt(max(classic_cov[1, 1], 0.0))

    hat = np.sum(X * (X @ xtx_inv), axis=1)
    denom = np.clip(1.0 - hat, 1e-12, None)
    omega = (resid / denom) ** 2
    hc3_cov = xtx_inv @ (X.T @ (omega[:, None] * X)) @ xtx_inv
    hc3_se = math.sqrt(max(hc3_cov[1, 1], 0.0))

    df = max(n - 2, 1)
    tcrit = float(t.ppf(0.975, df))
    tstat = beta[1] / hc3_se if hc3_se > 0 else np.nan
    p_value = float(2.0 * t.sf(abs(tstat), df)) if np.isfinite(tstat) else np.nan
    ci_low = float(beta[1] - tcrit * hc3_se)
    ci_high = float(beta[1] + tcrit * hc3_se)
    p10, p90 = np.nanpercentile(x, [10, 90])
    delta = float(p90 - p10)

    return {
        "n": int(n),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": p_value,
        "r2": float(r2),
        "hc3_se": float(hc3_se),
        "classic_se": float(classic_se),
        "p10": float(p10),
        "p90": float(p90),
        "contrast_90_minus_10": float(beta[1] * delta),
        "contrast_ci_low": float(ci_low * delta),
        "contrast_ci_high": float(ci_high * delta),
    }


def cluster_bootstrap_slope(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "genus_cluster",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    data = df[[x_col, y_col, cluster_col]].dropna().copy()
    if len(data) < 3 or data[x_col].nunique() < 2 or data[cluster_col].nunique() < 2:
        return {"cluster_type": cluster_col, "n_clusters": data[cluster_col].nunique(), "boot_ci_low": np.nan, "boot_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted(data[cluster_col].unique()))
    groups = {cluster: group for cluster, group in data.groupby(cluster_col, sort=False)}
    slopes: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([groups[c] for c in sampled], ignore_index=True)
        if boot[x_col].nunique() < 2:
            continue
        res = ols_hc3(boot[x_col].to_numpy(), boot[y_col].to_numpy())
        if np.isfinite(res["slope"]):
            slopes.append(float(res["slope"]))
    if not slopes:
        return {"cluster_type": cluster_col, "n_clusters": len(clusters), "boot_ci_low": np.nan, "boot_ci_high": np.nan}
    low, high = np.percentile(slopes, [2.5, 97.5])
    return {
        "cluster_type": cluster_col,
        "n_clusters": int(len(clusters)),
        "boot_ci_low": float(low),
        "boot_ci_high": float(high),
        "boot_n": int(len(slopes)),
    }


def standardize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - values.mean(skipna=True)) / sd


def multi_ols_hc3(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    data = df[[y_col] + x_cols].dropna()
    n = len(data)
    p_count = len(x_cols) + 1
    if n <= p_count:
        return {"n": n}
    X = np.column_stack([np.ones(n)] + [data[c].to_numpy(float) for c in x_cols])
    y = data[y_col].to_numpy(float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    hat = np.sum(X * (X @ xtx_inv), axis=1)
    omega = (resid / np.clip(1.0 - hat, 1e-12, None)) ** 2
    hc3_cov = xtx_inv @ (X.T @ (omega[:, None] * X)) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(hc3_cov), 0.0))
    dfree = max(n - p_count, 1)
    tcrit = float(t.ppf(0.975, dfree))
    out = {"n": int(n), "r2": float(r2)}
    names = ["intercept"] + x_cols
    for i, name in enumerate(names):
        tstat = beta[i] / se[i] if se[i] > 0 else np.nan
        out[f"{name}_coef"] = float(beta[i])
        out[f"{name}_hc3_se"] = float(se[i])
        out[f"{name}_ci_low"] = float(beta[i] - tcrit * se[i])
        out[f"{name}_ci_high"] = float(beta[i] + tcrit * se[i])
        out[f"{name}_p"] = float(2.0 * t.sf(abs(tstat), dfree)) if np.isfinite(tstat) else np.nan
    return out


def save_primary_scatter(df: pd.DataFrame, out_dir: Path, x_col: str, y_col: str, stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = df[[x_col, y_col]].dropna()
    x = data[x_col].to_numpy(float)
    y = data[y_col].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(x, y, s=24, alpha=0.68, edgecolor="white", linewidth=0.35, color="#365c8d")
    xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    yhat = stats["intercept"] + stats["slope"] * xgrid
    ax.plot(xgrid, yhat, color="#b83b3b", linewidth=2.2)
    label = (
        f"n = {int(stats['n'])}; slope = {stats['slope']:.3f} deg C per degree\n"
        f"HC3 95% CI [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}], "
        f"R2 = {stats['r2']:.3f}"
    )
    ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.set_xlabel("Occurrence-weighted absolute latitude (degrees)")
    ax.set_ylabel("Predicted normalized Topt (deg C)")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"figure2_species_topt_vs_abs_latitude.{ext}", dpi=300)
    plt.close(fig)


def save_secondary_traits(df: pd.DataFrame, out_dir: Path, effect_lookup: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success = df[df["fit_status"].eq("success")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True)
    specs = [
        ("E_eV", "Schoolfield E (eV)", "#357a38"),
        ("log_B0", "log(B0)", "#7b4da3"),
    ]
    x = success["abundance_weighted_absolute_latitude"].to_numpy(float)
    for ax, (col, ylabel, color) in zip(axes, specs):
        data = success[["abundance_weighted_absolute_latitude", col]].dropna()
        ax.scatter(
            data["abundance_weighted_absolute_latitude"],
            data[col],
            s=22,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.3,
            color=color,
        )
        st = effect_lookup[col]
        xgrid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        ax.plot(xgrid, st["intercept"] + st["slope"] * xgrid, color="#222222", linewidth=1.8)
        ax.text(
            0.03,
            0.97,
            f"n = {int(st['n'])}; slope = {st['slope']:.4f}\nHC3 p = {st['p']:.3g}, R2 = {st['r2']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
        )
        ax.set_xlabel("Occurrence-weighted absolute latitude (degrees)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"figure3_schoolfield_secondary_traits.{ext}", dpi=300)
    plt.close(fig)


def save_fba_audit(df: pd.DataFrame, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success = df[df["fit_status"].eq("success")].dropna(subset=["tpc_peak_temp_C", "observed_topt_C"]).copy()
    diff = success["observed_topt_C"] - success["tpc_peak_temp_C"]
    audit = {
        "positive_fba_species": int(len(success)),
        "max_abs_topt_difference_C": float(np.nanmax(np.abs(diff))) if len(diff) else np.nan,
        "mean_abs_topt_difference_C": float(np.nanmean(np.abs(diff))) if len(diff) else np.nan,
    }
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(success["tpc_peak_temp_C"], success["observed_topt_C"], s=23, alpha=0.65, color="#386cb0")
    lo = float(np.nanmin([success["tpc_peak_temp_C"].min(), success["observed_topt_C"].min()]))
    hi = float(np.nanmax([success["tpc_peak_temp_C"].max(), success["observed_topt_C"].max()]))
    ax.plot([lo, hi], [lo, hi], color="#b83b3b", linewidth=1.8)
    ax.set_xlabel("Normalized-curve Topt (deg C)")
    ax.set_ylabel("FBA-scaled curve Topt (deg C)")
    ax.text(
        0.03,
        0.97,
        f"n = {audit['positive_fba_species']}\nmax |difference| = {audit['max_abs_topt_difference_C']:.3g} deg C",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"figureS1_fba_topt_invariance_audit.{ext}", dpi=300)
    plt.close(fig)
    return audit


def save_attrition_figure(attrition: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.axis("off")
    y_positions = np.linspace(0.82, 0.18, len(attrition))
    for i, (_, row) in enumerate(attrition.iterrows()):
        y = y_positions[i]
        ax.text(0.02, y, str(row["stage"]), ha="left", va="center", fontsize=10.2, weight="bold")
        ax.text(0.45, y, str(row["n"]), ha="center", va="center", fontsize=10.2)
        ax.text(0.98, y, str(row["description"]), ha="right", va="center", fontsize=8.8)
        if i < len(attrition) - 1:
            ax.annotate("", xy=(0.45, y_positions[i + 1] + 0.04), xytext=(0.45, y - 0.04), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.text(0.02, 0.95, "Stage", fontsize=10, weight="bold")
    ax.text(0.45, 0.95, "n", fontsize=10, weight="bold", ha="center")
    ax.text(0.98, 0.95, "What this number means", fontsize=10, weight="bold", ha="right")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"figure1_workflow_attrition.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpc-summary", type=Path, default=DEFAULT_TPC_SUMMARY)
    parser.add_argument("--schoolfield", type=Path, default=DEFAULT_SCHOOLFIELD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tpc = pd.read_csv(args.tpc_summary)
    sf = pd.read_csv(args.schoolfield)
    tpc = ensure_numeric(
        tpc,
        [
            "mean_absolute_latitude",
            "min_absolute_latitude",
            "max_absolute_latitude",
            "abundance_weighted_absolute_latitude",
            "fasta_record_count",
            "predicted_ogt_C",
            "tpc_peak_temp_C",
            "Pmax",
            "E",
            "embedding_dim",
        ],
    )
    sf = ensure_numeric(
        sf,
        [
            "fba_peak_growth_rate_h_inv",
            "observed_topt_C",
            "observed_peak_rate_h_inv",
            "log_B0",
            "B0_h_inv",
            "E_eV",
            "Eh_eV",
            "Th_C",
            "fit_r2_log",
        ],
    )
    master = tpc.merge(sf, on=["proteome_id", "species"], how="left", validate="one_to_one")
    master["latitude_range_deg"] = master["max_absolute_latitude"] - master["min_absolute_latitude"]
    master["genus_cluster"] = master["species"].map(genus_from_species)
    master["positive_fba"] = master["fba_peak_growth_rate_h_inv"].fillna(0) > 0
    master.to_csv(args.out_dir / "species_level_master_table.csv", index=False)

    # Primary and sensitivity models.
    effect_rows: list[dict] = []
    model_specs = [
        ("tpc_peak_temp_C", "abundance_weighted_absolute_latitude", "primary"),
        ("tpc_peak_temp_C", "mean_absolute_latitude", "unweighted_occurrence_latitude_sensitivity"),
        ("predicted_ogt_C", "abundance_weighted_absolute_latitude", "secondary_ogt_context"),
        ("E_eV", "abundance_weighted_absolute_latitude", "secondary_schoolfield_positive_fba"),
        ("log_B0", "abundance_weighted_absolute_latitude", "exploratory_schoolfield_positive_fba"),
        ("B0_h_inv", "abundance_weighted_absolute_latitude", "exploratory_schoolfield_positive_fba"),
        ("fba_peak_growth_rate_h_inv", "abundance_weighted_absolute_latitude", "exploratory_fba_anchor"),
    ]
    for response, predictor, analysis_type in model_specs:
        df = master.copy()
        if response in {"E_eV", "log_B0", "B0_h_inv", "fba_peak_growth_rate_h_inv"}:
            df = df[df["fit_status"].eq("success")]
        stats = ols_hc3(df[predictor].to_numpy(), df[response].to_numpy())
        boot = cluster_bootstrap_slope(df, predictor, response, n_boot=args.bootstrap)
        row = {
            "response": response,
            "predictor": predictor,
            "analysis_type": analysis_type,
            **stats,
            **boot,
        }
        effect_rows.append(row)

    # Restricted-distribution sensitivity: bottom half of latitude ranges.
    restricted_cutoff = float(master["latitude_range_deg"].median(skipna=True))
    restricted = master[master["latitude_range_deg"] <= restricted_cutoff]
    stats = ols_hc3(restricted["abundance_weighted_absolute_latitude"].to_numpy(), restricted["tpc_peak_temp_C"].to_numpy())
    boot = cluster_bootstrap_slope(restricted, "abundance_weighted_absolute_latitude", "tpc_peak_temp_C", n_boot=args.bootstrap)
    effect_rows.append(
        {
            "response": "tpc_peak_temp_C",
            "predictor": "abundance_weighted_absolute_latitude",
            "analysis_type": f"restricted_latitude_range_le_median_{restricted_cutoff:.3f}",
            **stats,
            **boot,
        }
    )
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(args.out_dir / "table2_species_level_effects_hc3_bootstrap.csv", index=False)

    # Exploratory B0-E interaction in standardized variables.
    success = master[master["fit_status"].eq("success")].copy()
    success["z_log_B0"] = standardize(success["log_B0"])
    success["z_E_eV"] = standardize(success["E_eV"])
    success["z_lat"] = standardize(success["abundance_weighted_absolute_latitude"])
    success["z_E_x_z_lat"] = success["z_E_eV"] * success["z_lat"]
    interaction = multi_ols_hc3(success, "z_log_B0", ["z_E_eV", "z_lat", "z_E_x_z_lat"])
    pd.DataFrame([interaction]).to_csv(args.out_dir / "exploratory_logB0_E_latitude_interaction.csv", index=False)

    attrition = pd.DataFrame(
        [
            {
                "stage": "MicrobeAtlas soil occurrence records",
                "n": "processed upstream",
                "description": "97% OTU occurrence, relative abundance and coordinates used for taxon selection",
            },
            {
                "stage": "Matched reference proteomes",
                "n": int(master["proteome_id"].nunique()),
                "description": "species-level bacterial/archaeal taxa matched to UniProt and FASTA >= 1,000 proteins",
            },
            {
                "stage": "ESM2 embeddings and normalized TPCs",
                "n": int(master["proteome_id"].nunique()),
                "description": "one 1,280-dimensional proteome embedding and one normalized TPC per species",
            },
            {
                "stage": "FBA anchor",
                "n": f"{int(master['positive_fba'].sum())} positive / {int((~master['positive_fba']).sum())} zero",
                "description": "CarveMe + COBRApy under common M9-like medium; zero-growth species excluded from rate fitting",
            },
            {
                "stage": "Schoolfield trait fitting",
                "n": int(master["fit_status"].eq("success").sum()),
                "description": "species with positive FBA-scaled curves fitted for E and B0",
            },
        ]
    )
    attrition.to_csv(args.out_dir / "table1_workflow_attrition.csv", index=False)

    primary = effects[(effects["response"] == "tpc_peak_temp_C") & (effects["analysis_type"] == "primary")].iloc[0].to_dict()
    lookup = {row["response"]: row for row in effect_rows if row["predictor"] == "abundance_weighted_absolute_latitude"}
    save_attrition_figure(attrition, args.out_dir)
    save_primary_scatter(master, args.out_dir, "abundance_weighted_absolute_latitude", "tpc_peak_temp_C", primary)
    save_secondary_traits(master, args.out_dir, lookup)
    audit = save_fba_audit(master, args.out_dir)
    pd.DataFrame([audit]).to_csv(args.out_dir / "fba_scaling_topt_invariance_audit.csv", index=False)

    notes = [
        "Feedback-corrected species-level analysis.",
        f"Unique species/proteomes for normalized analysis: {master['proteome_id'].nunique()}",
        f"Positive FBA species for Schoolfield/FBA-dependent analyses: {int(master['positive_fba'].sum())}",
        f"Zero-growth FBA species: {int((~master['positive_fba']).sum())}",
        f"Primary Topt slope: {primary['slope']:.6f} deg C per degree absolute latitude",
        f"Primary HC3 95% CI: [{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]",
        f"Primary p: {primary['p']:.6g}; R2: {primary['r2']:.6f}",
        f"Primary 10th-90th latitude contrast: {primary['contrast_90_minus_10']:.6f} deg C",
        f"FBA scaling audit max abs Topt difference: {audit['max_abs_topt_difference_C']:.6g} deg C",
        "Low/mid/high latitude groups should be treated as descriptive only, not the primary inferential unit.",
    ]
    (args.out_dir / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
