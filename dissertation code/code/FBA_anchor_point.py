#!/usr/bin/env python3
"""
FBA_anchor_point.py  --  GEM reconstruction and FBA-based peak growth rate.

Workflow
--------
    1. CarveMe reconstructs a genome-scale metabolic model (GEM) from a
       proteome FASTA (requires CarveMe and DIAMOND on PATH, and a CarveMe
       database; see https://carveme.readthedocs.io).
    2. COBRApy loads the GEM and applies the user-supplied medium (exchange
       reaction bounds).
    3. FBA is solved; the optimal growth rate (h-1) is returned as the
       absolute anchor for the normalised TPC shape.

Dependencies
------------
    pip install cobra
    pip install carveme      # also needs diamond and cplex/glpk solver

Usage
-----
    # Python API
    from FBA_anchor_point import get_peak_growth_rate
    rate = get_peak_growth_rate(
        fasta_path   = "genome.fna",
        medium       = {"EX_glc__D_e": 10.0, "EX_o2_e": 20.0, "EX_nh4_e": 10.0},
        temperature_c = 37.0,
    )

    # Command line
    python code/FBA_anchor_point.py \\
        --fasta   genome.fna \\
        --medium  examples/example_medium_ecoli.json \\
        --ogt     37.0
"""

import sys, os, subprocess, json, warnings, argparse, tempfile
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR  = Path(__file__).parent
REPO_DIR    = SCRIPT_DIR.parent

# ============================================================
# GEM reconstruction with CarveMe
# ============================================================

def reconstruct_gem(
    proteome_fasta: Path,
    output_xml: Path,
    universe: str = "bacteria",
    gapfill_medium: str = None,
    init_medium: str = None,
) -> Path:
    """Run CarveMe to build a GEM from a proteome FASTA.

    Parameters
    ----------
    proteome_fasta : protein sequences in FASTA format (amino acid)
    output_xml     : destination for the SBML GEM file
    universe       : CarveMe universe to use ('bacteria', 'gramneg', 'grampos', ...)

    Returns
    -------
    Path to the SBML file

    Notes
    -----
    Requires CarveMe (pip install carveme), DIAMOND, and a valid CarveMe
    universe database.  On first run, CarveMe downloads the database automatically.
    """
    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["carve", str(proteome_fasta),
           "--output", str(output_xml),
           "--universe", universe,
           "--solver", _carveme_solver()]
    if gapfill_medium:
        cmd.extend(["--gapfill", gapfill_medium])
    if init_medium:
        cmd.extend(["--init", init_medium])

    print(f"[CarveMe] Reconstructing GEM: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"CarveMe failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not output_xml.exists():
        raise FileNotFoundError(f"CarveMe did not produce output at {output_xml}")
    print(f"[CarveMe] GEM saved to: {output_xml}")
    return output_xml


def _cplex_available() -> bool:
    try:
        import cplex
        return True
    except ImportError:
        return False


def _carveme_solver() -> str:
    if _cplex_available():
        return "cplex"
    try:
        import gurobipy  # noqa: F401
        return "gurobi"
    except ImportError:
        pass
    try:
        import pyscipopt  # noqa: F401
        return "scip"
    except ImportError:
        return "glpk"

# ============================================================
# FBA with COBRApy
# ============================================================

def run_fba(gem_path: Path, medium: dict) -> float:
    """Load GEM from SBML and solve FBA under the supplied medium.

    Parameters
    ----------
    gem_path : path to SBML metabolic model (.xml / .json)
    medium   : dict mapping exchange reaction IDs (e.g. 'EX_glc__D_e') to
               maximum uptake rates in mmol gDW-1 h-1 (positive values).

    Returns
    -------
    float  -- optimal growth rate in h-1 (0.0 if infeasible)
    """
    try:
        import cobra
    except ImportError:
        raise ImportError("COBRApy not installed. Run:  pip install cobra")

    gem_path = Path(gem_path)
    if gem_path.suffix == ".json":
        model = cobra.io.load_json_model(str(gem_path))
    else:
        model = cobra.io.read_sbml_model(str(gem_path))

    with model:
        for rxn in model.exchanges:
            rxn.lower_bound = 0.0

        matched = []
        missing = []
        for rxn_id, ub in medium.items():
            try:
                rxn = model.reactions.get_by_id(rxn_id)
                rxn.lower_bound = -abs(float(ub))
                matched.append(rxn_id)
            except KeyError:
                missing.append(rxn_id)
                print(f"[FBA] Warning: exchange reaction '{rxn_id}' not in model -- skipped")
        if not matched:
            raise ValueError(
                "None of the supplied medium exchange reactions were found in the reconstructed model. "
                "Check whether the model uses BiGG-style exchange IDs or use a different medium."
            )
        print(f"[FBA] Medium reactions matched: {len(matched)} / {len(medium)}")

        solution = model.optimize()

        if solution.status != "optimal":
            print(f"[FBA] Solution status: {solution.status} -- returning 0.0")
            return 0.0

        growth_rate = float(solution.objective_value)
        print(f"[FBA] Growth rate = {growth_rate:.4f} h-1")
        return max(growth_rate, 0.0)

# ============================================================
# Utility: extract proteins from genome FASTA using Prodigal
# ============================================================

def _prodigal_proteins(genome_fasta: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prot_out = out_dir / "proteins.faa"
    cmd = ["prodigal", "-i", str(genome_fasta), "-a", str(prot_out),
           "-p", "meta", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Prodigal failed:\n{result.stderr}")
    return prot_out


def _is_nucleotide(fasta_path: Path) -> bool:
    text = fasta_path.read_text(errors="ignore").upper()
    non_hdr = "".join(l for l in text.splitlines() if not l.startswith(">"))
    if not non_hdr:
        return True
    dna = sum(non_hdr.count(c) for c in "ACGTN")
    return (dna / len(non_hdr)) > 0.80

# ============================================================
# High-level API
# ============================================================

def get_peak_growth_rate(fasta_path, medium: dict,
                         temperature_c: float = 37.0,
                         gem_path: Path = None,
                         universe: str = "bacteria",
                         gapfill_medium: str = None,
                         init_medium: str = None,
                         tmp_dir: Path = None) -> float:
    """Return the FBA-predicted peak growth rate for an organism under given medium.

    Parameters
    ----------
    fasta_path    : genome (nucleotide) or proteome (amino acid) FASTA
    medium        : dict {exchange_rxn_id: max_uptake_rate mmol/gDW/h}
    temperature_c : growth temperature for context (not directly used in FBA
                    unless the model contains temperature-sensitive constraints)
    gem_path      : pre-built SBML GEM to skip reconstruction (optional)
    universe      : CarveMe universe string (used only when gem_path is None)
    tmp_dir       : working directory for intermediate files

    Returns
    -------
    float  -- growth rate in h-1
    """
    fasta_path = Path(fasta_path)

    if tmp_dir is None:
        tmp_dir = fasta_path.parent / "_tmp_fba"
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if gem_path is None:
        if _is_nucleotide(fasta_path):
            print("[FBA] Nucleotide genome detected; running Prodigal first ...")
            proteome = _prodigal_proteins(fasta_path, tmp_dir)
        else:
            proteome = fasta_path

        gem_path = tmp_dir / (fasta_path.stem + "_gem.xml")
        reconstruct_gem(
            proteome,
            gem_path,
            universe=universe,
            gapfill_medium=gapfill_medium,
            init_medium=init_medium,
        )

    growth_rate = run_fba(gem_path, medium)
    return growth_rate

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct GEM with CarveMe and compute FBA growth rate")
    parser.add_argument("--fasta",    required=True, type=Path,
                        help="Genome FASTA (nucleotide) or proteome FASTA (amino acid)")
    parser.add_argument("--medium",   required=True, type=Path,
                        help="JSON file with exchange reaction IDs and uptake rates")
    parser.add_argument("--ogt",      default=37.0,  type=float,
                        help="Growth temperature (C, informational)")
    parser.add_argument("--gem",      default=None,  type=Path,
                        help="Pre-built SBML GEM path (skips CarveMe reconstruction)")
    parser.add_argument("--universe", default="bacteria",
                        help="CarveMe universe (bacteria / gramneg / grampos)")
    parser.add_argument("--gapfill_medium", default=None,
                        help="Optional CarveMe medium for gap filling, e.g. M9")
    parser.add_argument("--init_medium", default=None,
                        help="Optional CarveMe medium for model initialization, e.g. M9")
    parser.add_argument("--tmp_dir",  default=None, type=Path)
    args = parser.parse_args()

    if not args.fasta.exists():
        sys.exit(f"FASTA not found: {args.fasta}")
    if not args.medium.exists():
        sys.exit(f"Medium JSON not found: {args.medium}")

    with open(args.medium) as fh:
        medium_raw = json.load(fh)
    medium = {k: v for k, v in medium_raw.items() if not k.startswith("_")}

    rate = get_peak_growth_rate(
        fasta_path    = args.fasta,
        medium        = medium,
        temperature_c = args.ogt,
        gem_path      = args.gem,
        universe      = args.universe,
        gapfill_medium= args.gapfill_medium,
        init_medium   = args.init_medium,
        tmp_dir       = args.tmp_dir,
    )
    print(f"\nFBA peak growth rate: {rate:.4f} h-1 at OGT {args.ogt} C")
