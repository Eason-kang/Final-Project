#!/usr/bin/env python3
"""Fit Schoolfield-style parameters to FBA-scaled predicted TPCs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import linregress


K_BOLTZMANN = 8.617333262145e-5  # eV K-1
T_REF_C = 15.0
GROUP_ORDER = ["low_abs_latitude", "mid_abs_latitude", "high_abs_latitude"]
GROUP_LABELS = {
    "low_abs_latitude": "Low",
    "mid_abs_latitude": "Mid",
    "high_abs_latitude": "High",
}
GROUP_COLORS = {
    "low_abs_latitude": "#2ca58d",
    "mid_abs_latitude": "#4f6fd7",
    "high_abs_latitude": "#d94f4f",
}


def schoolfield_high(temp_c: np.ndarray, log_b0: float, e: float, eh: float, th_c: float) -> np.ndarray:
    """Schoolfield high-temperature inactivation model.

    B0 is the rate at T_REF_C in the absence of high-temperature inactivation.
    E and Eh are in eV; Th is in Celsius.
    """
    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    tref_k = T_REF_C + 273.15
    th_k = th_c + 273.15
    log_rate = log_b0 + (e / K_BOLTZMANN) * ((1.0 / tref_k) - (1.0 / temp_k))
    denom = 1.0 + np.exp(np.clip((eh / K_BOLTZMANN) * ((1.0 / th_k) - (1.0 / temp_k)), -700, 700))
    return np.exp(log_rate) / denom


def fit_one_species(group: pd.DataFrame) -> dict:
    group = group.sort_values("temperature_C")
    temp = group["temperature_C"].to_numpy(dtype=float)
    rate = group["absolute_growth_rate_h_inv"].to_numpy(dtype=float)
    max_rate = float(np.nanmax(rate)) if len(rate) else np.nan
    proteome_id = group["proteome_id"].iloc[0]
    species = group["species"].iloc[0]

    base = {
        "proteome_id": proteome_id,
        "species": species,
        "n_temperature_points": int(len(group)),
        "fba_peak_growth_rate_h_inv": float(group["fba_peak_growth_rate_h_inv"].iloc[0]),
        "observed_topt_C": float(temp[int(np.nanargmax(rate))]) if np.isfinite(max_rate) and max_rate > 0 else np.nan,
        "observed_peak_rate_h_inv": max_rate,
    }
    if not np.isfinite(max_rate) or max_rate <= 0:
        return {**base, "fit_status": "skipped_zero_fba_peak"}

    # Fit log rates and omit numerical tail zeros. This is equivalent to fitting
    # the positive part of the FBA-scaled predicted curve.
    mask = np.isfinite(rate) & (rate > max(max_rate * 1e-5, 1e-10))
    temp_fit = temp[mask]
    rate_fit = rate[mask]
    if len(rate_fit) < 12:
        return {**base, "fit_status": "skipped_too_few_positive_points"}

    topt_init = float(temp[int(np.nanargmax(rate))])
    idx_ref = int(np.argmin(np.abs(temp_fit - T_REF_C)))
    log_b0_init = float(np.log(max(rate_fit[idx_ref], 1e-8)))
    x0 = np.array([log_b0_init, 0.65, 2.5, min(max(topt_init + 8.0, 20.0), 95.0)], dtype=float)
    lower = np.array([np.log(1e-8), 0.01, 0.05, max(float(temp_fit.min()) + 1.0, 1.0)])
    upper = np.array([np.log(20.0), 3.0, 20.0, 120.0])

    def residual(params: np.ndarray) -> np.ndarray:
        pred = schoolfield_high(temp_fit, *params)
        pred = np.clip(pred, 1e-12, None)
        return np.log(pred) - np.log(rate_fit)

    try:
        result = least_squares(residual, x0=x0, bounds=(lower, upper), max_nfev=5000)
        params = result.x
        pred = schoolfield_high(temp_fit, *params)
        log_obs = np.log(rate_fit)
        log_pred = np.log(np.clip(pred, 1e-12, None))
        ss_res = float(np.sum((log_obs - log_pred) ** 2))
        ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            **base,
            "fit_status": "success" if result.success else "not_converged",
            "log_B0": float(params[0]),
            "B0_h_inv": float(np.exp(params[0])),
            "E_eV": float(params[1]),
            "Eh_eV": float(params[2]),
            "Th_C": float(params[3]),
            "fit_r2_log": r2_log,
            "fit_n_positive_points": int(len(rate_fit)),
            "fit_cost": float(result.cost),
            "fit_message": result.message,
        }
    except Exception as exc:  # noqa: BLE001
        return {**base, "fit_status": "failed", "fit_message": f"{type(exc).__name__}: {exc}"}


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def save_boxplot(df: pd.DataFrame, y_col: str, ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [df.loc[df["analysis_latitude_group"] == g, y_col].dropna().to_numpy() for g in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.boxplot(groups, tick_labels=[GROUP_LABELS[g] for g in GROUP_ORDER], showfliers=False, widths=0.55)
    rng = np.random.default_rng(42)
    for i, values in enumerate(groups, start=1):
        if len(values):
            ax.scatter(np.full(len(values), i) + rng.normal(0, 0.04, len(values)), values, s=10, alpha=0.35)
    ax.set_xlabel("Absolute-latitude group")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, x_col: str, y_col: str, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for group in GROUP_ORDER:
        sub = df[df["analysis_latitude_group"] == group]
        ax.scatter(sub[x_col], sub[y_col], s=18, alpha=0.65, label=GROUP_LABELS[group], color=GROUP_COLORS[group])
    data = df[[x_col, y_col]].dropna()
    if len(data) > 2 and data[x_col].nunique() > 1:
        res = linregress(data[x_col], data[y_col])
        x = np.linspace(float(data[x_col].min()), float(data[x_col].max()), 100)
        ax.plot(x, res.slope * x + res.intercept, color="black", lw=1.4)
        ax.text(
            0.02,
            0.98,
            f"slope={res.slope:.3g}, R2={res.rvalue**2:.3g}, p={res.pvalue:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_b0_e_plot(records: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for group in GROUP_ORDER:
        sub = records[records["analysis_latitude_group"] == group]
        ax.scatter(sub["E_eV"], sub["log_B0"], s=18, alpha=0.65, color=GROUP_COLORS[group], label=GROUP_LABELS[group])
    ax.set_xlabel("Activation energy E (eV)")
    ax.set_ylabel("log(B0)")
    ax.set_title("Fitted B0-E relationship from FBA-scaled TPCs")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(args.fba_curves)
    records = pd.read_csv(args.analysis_records)
    peaks = pd.read_csv(args.fba_peaks)

    curves = ensure_numeric(
        curves,
        ["temperature_C", "norm_shape", "predicted_ogt_C", "fba_peak_growth_rate_h_inv", "absolute_growth_rate_h_inv"],
    )
    records = ensure_numeric(
        records,
        [
            "group_abundance_weighted_absolute_latitude",
            "group_n_occurrences",
            "group_sum_relative_abundance",
            "predicted_ogt_C",
            "tpc_peak_temp_C",
            "fba_peak_growth_rate_h_inv",
        ],
    )

    fit_rows = [fit_one_species(group) for _, group in curves.groupby("proteome_id", sort=False)]
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(args.out_dir / "schoolfield_fitted_traits_by_species.csv", index=False)

    records_with_fits = records.merge(
        fits,
        left_on="uniprot_proteome_id",
        right_on="proteome_id",
        how="left",
        suffixes=("", "_fit"),
    )
    records_with_fits.to_csv(args.out_dir / "analysis_records_with_schoolfield_traits.csv", index=False)

    success_fits = fits[fits["fit_status"] == "success"].copy()
    success_records = records_with_fits[records_with_fits["fit_status"] == "success"].copy()

    group_summary = []
    for group in GROUP_ORDER:
        sub = success_records[success_records["analysis_latitude_group"] == group]
        row = {
            "analysis_latitude_group": group,
            "n_records": len(sub),
            "n_unique_species": sub["candidate_species"].nunique(),
            "mean_abs_latitude": sub["group_abundance_weighted_absolute_latitude"].mean(),
        }
        for col in ["B0_h_inv", "log_B0", "E_eV", "Eh_eV", "Th_C", "observed_topt_C", "fba_peak_growth_rate_h_inv"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_median"] = sub[col].median()
            row[f"{col}_sd"] = sub[col].std()
        group_summary.append(row)
    pd.DataFrame(group_summary).to_csv(args.out_dir / "schoolfield_latitude_group_summary.csv", index=False)

    model_rows = []
    for y_col in ["log_B0", "B0_h_inv", "E_eV", "observed_topt_C", "fba_peak_growth_rate_h_inv"]:
        data = success_records[["group_abundance_weighted_absolute_latitude", y_col]].dropna()
        if len(data) > 2 and data["group_abundance_weighted_absolute_latitude"].nunique() > 1:
            res = linregress(data["group_abundance_weighted_absolute_latitude"], data[y_col])
            model_rows.append(
                {
                    "response": y_col,
                    "predictor": "group_abundance_weighted_absolute_latitude",
                    "n": len(data),
                    "slope": res.slope,
                    "intercept": res.intercept,
                    "r": res.rvalue,
                    "r2": res.rvalue**2,
                    "p_value": res.pvalue,
                }
            )
    pd.DataFrame(model_rows).to_csv(args.out_dir / "schoolfield_latitude_linear_models.csv", index=False)

    save_boxplot(success_records, "observed_topt_C", "Observed Topt from absolute TPC (C)", "FBA-scaled Topt by latitude group", args.out_dir / "fba_scaled_topt_by_latitude_group.png")
    save_boxplot(success_records, "log_B0", "log(B0)", "Fitted log(B0) by latitude group", args.out_dir / "logB0_by_latitude_group.png")
    save_boxplot(success_records, "E_eV", "Fitted E (eV)", "Fitted activation energy E by latitude group", args.out_dir / "E_by_latitude_group.png")
    save_boxplot(success_records, "fba_peak_growth_rate_h_inv", "FBA peak growth rate (h^-1)", "FBA peak growth rate by latitude group", args.out_dir / "fba_peak_rate_by_latitude_group.png")
    save_scatter(
        success_records,
        "group_abundance_weighted_absolute_latitude",
        "log_B0",
        "Group-specific abundance-weighted absolute latitude (degrees)",
        "log(B0)",
        "log(B0) vs. absolute latitude",
        args.out_dir / "logB0_vs_absolute_latitude.png",
    )
    save_scatter(
        success_records,
        "group_abundance_weighted_absolute_latitude",
        "E_eV",
        "Group-specific abundance-weighted absolute latitude (degrees)",
        "Fitted E (eV)",
        "Fitted E vs. absolute latitude",
        args.out_dir / "E_vs_absolute_latitude.png",
    )
    save_b0_e_plot(success_records, args.out_dir / "logB0_vs_E_by_latitude_group.png")

    notes = [
        f"fba_peak_rows: {len(peaks)}",
        f"fba_unique_proteomes: {peaks['proteome_id'].nunique()}",
        f"fba_zero_growth_rate: {int((pd.to_numeric(peaks['fba_peak_growth_rate_h_inv'], errors='coerce') <= 0).sum())}",
        f"curve_unique_proteomes: {curves['proteome_id'].nunique()}",
        f"schoolfield_fit_success_species: {len(success_fits)}",
        f"analysis_records_with_successful_fit: {len(success_records)}",
        "note: B0/E are fitted to model-predicted FBA-scaled TPCs, not directly measured empirical growth curves.",
    ]
    (args.out_dir / "schoolfield_analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--fba-peaks", type=Path, default=root / "data/processed/fba_peak_growth_rates.csv")
    parser.add_argument("--fba-curves", type=Path, default=root / "data/processed/fba_scaled_absolute_tpcs_by_species.csv")
    parser.add_argument("--analysis-records", type=Path, default=root / "data/processed/selected_analysis_records_with_fba.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/schoolfield_fba_trait_analysis")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
