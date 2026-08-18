#!/usr/bin/env python3
"""Select proteome-ready soil taxa across an absolute-latitude gradient.

This script implements the revised sampling design:

1. Start from MicrobeAtlas samples with valid latitude/longitude.
2. Restrict the main analysis to soil/terrestrial environmental samples.
3. Calculate absolute latitude, |latitude|.
4. Select samples evenly across low/mid/high absolute-latitude quantiles.
5. Extract abundant OTUs, map them to species names, then query NCBI/UniProt.

The downstream model should use the generated FASTA manifest, not individual
per-species plots, as the input for HPC batch jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_top150_species_temp_gradient import (  # noqa: E402
    DEFAULT_BIOM,
    DEFAULT_OTU_INFO,
    classify_tpc_ready,
    confidence_rank,
    download_uniprot_fasta,
    extract_top_otus,
    load_otu_annotations,
    parse_float,
    query_ncbi_taxid,
    query_uniprot_proteome,
    read_biom_sample_map,
    taxonomy_domain,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data/raw"
DEFAULT_LATLON = DEFAULT_DATA_DIR / "samples.info.latlon.parsed.tsv"
DEFAULT_ENV = DEFAULT_DATA_DIR / "samples.env.info.tsv"

DEFAULT_SOIL_INCLUDE = r"soil|forest soil|agricultural soil|grassland|cropland|desert soil|peat|permafrost"
DEFAULT_SOIL_EXCLUDE = (
    r"marine|ocean|sea water|seawater|freshwater|lake|river|stream|sediment|"
    r"gut|feces|faeces|stool|clinical|patient|human|animal|host-associated|"
    r"skin|oral|saliva|plant-associated|root|leaf|rhizosphere"
)


def read_tsv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def find_column(fieldnames: list[str], candidates: list[str], contains: list[str] | None = None) -> str:
    lower_to_original = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if contains:
        for name in fieldnames:
            low = name.lower()
            if all(token.lower() in low for token in contains):
                return name
    return ""


def read_latlon(path: Path) -> dict[str, dict]:
    rows = read_tsv_rows(path)
    if not rows:
        return {}
    fields = list(rows[0].keys())
    sample_col = find_column(fields, ["MAP_SID", "SampleID", "sample_id", "sample", "run", "Run", "id"])
    lat_col = find_column(fields, ["latitude", "lat", "Latitude", "Lat"], ["lat"])
    lon_col = find_column(fields, ["longitude", "lon", "lng", "Longitude", "Long"], ["lon"])
    if not sample_col or not lat_col:
        raise ValueError(
            f"Could not identify sample/latitude columns in {path}. "
            f"Fields seen: {fields[:20]}"
        )
    out: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get(sample_col, "")).strip()
        lat = parse_float(row.get(lat_col))
        lon = parse_float(row.get(lon_col)) if lon_col else None
        if not sid or lat is None or lat < -90 or lat > 90:
            continue
        if lon is not None and (lon < -180 or lon > 180):
            lon = None
        out[sid] = {
            "metadata_sample_id": sid,
            "latitude": lat,
            "longitude": lon if lon is not None else "",
            "absolute_latitude": abs(lat),
        }
    return out


def read_environment_text(path: Path) -> dict[str, str]:
    rows = read_tsv_rows(path)
    if not rows:
        return {}
    fields = list(rows[0].keys())
    sample_col = find_column(fields, ["MAP_SID", "SampleID", "sample_id", "sample", "run", "Run", "id"])
    if not sample_col:
        raise ValueError(f"Could not identify sample ID column in {path}. Fields seen: {fields[:20]}")
    text_by_sample: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sid = str(row.get(sample_col, "")).strip()
        if not sid:
            continue
        parts = []
        for key, value in row.items():
            if key == sample_col or value is None:
                continue
            parts.append(str(value))
        text = " ".join(parts)
        aliases = [sid]
        if "." in sid:
            aliases.append(sid.split(".")[-1])
        for alias in aliases:
            text_by_sample[alias].append(text)
    return {sid: " ".join(parts).lower() for sid, parts in text_by_sample.items()}


def select_soil_samples(
    latlon: dict[str, dict],
    env_text: dict[str, str],
    include_pattern: str,
    exclude_pattern: str,
) -> dict[str, dict]:
    include = re.compile(include_pattern, flags=re.IGNORECASE)
    exclude = re.compile(exclude_pattern, flags=re.IGNORECASE) if exclude_pattern else None
    selected = {}
    for sid, meta in latlon.items():
        text = env_text.get(sid, "")
        if not text:
            continue
        if not include.search(text):
            continue
        if exclude and exclude.search(text):
            continue
        out = dict(meta)
        out["environment_text_excerpt"] = text[:300]
        selected[sid] = out
    return selected


def assign_latitude_quantiles(samples: dict[str, dict]) -> dict[str, str]:
    items = sorted(samples.items(), key=lambda item: item[1]["absolute_latitude"])
    n = len(items)
    groups = {}
    for rank, (sid, _meta) in enumerate(items):
        frac = rank / max(n - 1, 1)
        if frac < 1 / 3:
            groups[sid] = "low_abs_latitude"
        elif frac < 2 / 3:
            groups[sid] = "mid_abs_latitude"
        else:
            groups[sid] = "high_abs_latitude"
    return groups


def evenly_select_by_abs_latitude(
    samples: dict[str, dict],
    available_ids: set[str],
    sample_count_per_group: int,
) -> list[str]:
    groups = assign_latitude_quantiles({sid: meta for sid, meta in samples.items() if sid in available_ids})
    selected = []
    for group in ["low_abs_latitude", "mid_abs_latitude", "high_abs_latitude"]:
        ids = [sid for sid, g in groups.items() if g == group]
        ids.sort(key=lambda sid: samples[sid]["absolute_latitude"])
        if len(ids) <= sample_count_per_group:
            chosen = ids
        else:
            chosen = []
            for i in range(sample_count_per_group):
                idx = round(i * (len(ids) - 1) / max(sample_count_per_group - 1, 1))
                chosen.append(ids[idx])
        selected.extend(chosen)
    return list(dict.fromkeys(selected))


def aggregate_candidates_by_latitude(rows: list[dict]) -> list[dict]:
    by_species: dict[str, dict] = {}
    for row in rows:
        species = row.get("candidate_species", "")
        if not species:
            continue
        rel = parse_float(row.get("relative_abundance")) or 0.0
        abs_lat = parse_float(row.get("absolute_latitude"))
        entry = by_species.setdefault(species, {
            "candidate_species": species,
            "taxonomy_domain": row.get("taxonomy_domain", ""),
            "match_confidence": row.get("match_confidence", ""),
            "confidence_reason": row.get("confidence_reason", ""),
            "n_occurrences": 0,
            "n_samples": set(),
            "sum_relative_abundance": 0.0,
            "max_relative_abundance": 0.0,
            "abundance_weighted_abs_lat_sum": 0.0,
            "abundance_weight_sum": 0.0,
            "absolute_latitudes": [],
            "latitude_groups": set(),
            "example_otu_id": row.get("otu_id", ""),
            "example_metadata_sample_id": row.get("metadata_sample_id", ""),
        })
        entry["n_occurrences"] += 1
        entry["n_samples"].add(row.get("metadata_sample_id", ""))
        entry["latitude_groups"].add(row.get("latitude_group", ""))
        entry["sum_relative_abundance"] += rel
        entry["max_relative_abundance"] = max(entry["max_relative_abundance"], rel)
        if abs_lat is not None:
            entry["absolute_latitudes"].append(abs_lat)
            entry["abundance_weighted_abs_lat_sum"] += rel * abs_lat
            entry["abundance_weight_sum"] += rel
    out = []
    for entry in by_species.values():
        abs_lats = entry.pop("absolute_latitudes")
        n_samples = len(entry.pop("n_samples"))
        groups = sorted(x for x in entry.pop("latitude_groups") if x)
        weight = entry.pop("abundance_weight_sum")
        weighted = entry.pop("abundance_weighted_abs_lat_sum")
        entry["n_samples"] = n_samples
        entry["latitude_groups_observed"] = ";".join(groups)
        entry["mean_relative_abundance_per_occurrence"] = entry["sum_relative_abundance"] / max(entry["n_occurrences"], 1)
        entry["mean_absolute_latitude"] = sum(abs_lats) / len(abs_lats) if abs_lats else ""
        entry["min_absolute_latitude"] = min(abs_lats) if abs_lats else ""
        entry["max_absolute_latitude"] = max(abs_lats) if abs_lats else ""
        entry["abundance_weighted_absolute_latitude"] = weighted / weight if weight > 0 else entry["mean_absolute_latitude"]
        out.append(entry)
    out.sort(key=lambda r: (r["n_samples"], r["sum_relative_abundance"], r["max_relative_abundance"]), reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Select soil taxa across absolute latitude for UTPC/FBA/rTPC.")
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--otu-info", type=Path, default=DEFAULT_OTU_INFO)
    parser.add_argument("--latlon", type=Path, default=DEFAULT_LATLON)
    parser.add_argument("--env-info", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/soil_latitude_species"))
    parser.add_argument("--proteome-dir", type=Path, default=Path("data/proteomes_soil_latitude"))
    parser.add_argument("--samples-per-latitude-group", type=int, default=300)
    parser.add_argument("--candidate-pool", type=int, default=120)
    parser.add_argument("--target-taxa-per-group", type=int, default=200)
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"], default="high")
    parser.add_argument("--allowed-domains", default="Bacteria,Archaea")
    parser.add_argument("--min-protein-count", type=int, default=1000)
    parser.add_argument("--max-protein-count", type=int, default=12000)
    parser.add_argument("--soil-include-regex", default=DEFAULT_SOIL_INCLUDE)
    parser.add_argument("--soil-exclude-regex", default=DEFAULT_SOIL_EXCLUDE)
    parser.add_argument("--api-sleep", type=float, default=0.34)
    parser.add_argument("--ncbi-email", default="")
    parser.add_argument("--query-apis", action="store_true")
    parser.add_argument("--download-proteomes", action="store_true")
    args = parser.parse_args()

    if not args.latlon.exists():
        raise FileNotFoundError(
            f"Missing latitude/longitude metadata: {args.latlon}\n"
            "Download MicrobeAtlas samples.info.latlon.parsed.tsv and place it in data/raw."
        )
    if not args.env_info.exists():
        raise FileNotFoundError(
            f"Missing environment metadata: {args.env_info}\n"
            "Download MicrobeAtlas samples.env.info.tsv and place it in data/raw."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowed_domains = {x.strip() for x in args.allowed_domains.split(",") if x.strip()}
    min_rank = confidence_rank(args.min_confidence)
    target_total = args.target_taxa_per_group * 3

    print("[1/8] Reading latitude and environment metadata...")
    latlon = read_latlon(args.latlon)
    env_text = read_environment_text(args.env_info)
    soil_samples = select_soil_samples(latlon, env_text, args.soil_include_regex, args.soil_exclude_regex)
    print(f"  samples with valid lat/lon: {len(latlon)}")
    print(f"  samples with env text: {len(env_text)}")
    print(f"  soil-like samples after include/exclude filters: {len(soil_samples)}")

    print("[2/8] Matching soil samples to BIOM sample IDs...")
    biom_map = read_biom_sample_map(args.biom, set(soil_samples))
    selected_ids = evenly_select_by_abs_latitude(soil_samples, set(biom_map), args.samples_per_latitude_group)
    groups = assign_latitude_quantiles({sid: soil_samples[sid] for sid in biom_map})
    communities = []
    for sid in selected_ids:
        biom_id, biom_idx = biom_map[sid]
        meta = soil_samples[sid]
        communities.append({
            "metadata_sample_id": sid,
            "biom_sample_id": biom_id,
            "biom_sample_index": biom_idx,
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "absolute_latitude": meta["absolute_latitude"],
            "latitude_group": groups[sid],
            "environment_text_excerpt": meta.get("environment_text_excerpt", ""),
            # Kept for compatibility with reused extract_top_otus.
            "temp_C": meta["absolute_latitude"],
        })
    sample_fields = [
        "metadata_sample_id", "biom_sample_id", "biom_sample_index", "latitude", "longitude",
        "absolute_latitude", "latitude_group", "environment_text_excerpt", "temp_C",
    ]
    write_csv(args.out_dir / "01_selected_soil_samples_by_abs_latitude.csv", communities, sample_fields)
    print(f"  soil samples present in BIOM: {len(biom_map)}")
    print(f"  selected samples: {len(communities)}")

    print("[3/8] Extracting top OTUs by relative abundance...")
    top_otus = extract_top_otus(args.biom, communities, args.candidate_pool)
    meta_by_sample = {row["metadata_sample_id"]: row for row in communities}
    for row in top_otus:
        meta = meta_by_sample[row["metadata_sample_id"]]
        row["latitude"] = meta["latitude"]
        row["longitude"] = meta["longitude"]
        row["absolute_latitude"] = meta["absolute_latitude"]
        row["latitude_group"] = meta["latitude_group"]
        row.pop("temp_C", None)
    write_csv(args.out_dir / "02_top_otus_from_soil_latitude_samples.csv", top_otus)
    print(f"  top OTU rows: {len(top_otus)}")

    print("[4/8] Joining OTU taxonomy annotations...")
    annotations = load_otu_annotations(args.otu_info, {row["otu_id"] for row in top_otus})
    annotated = []
    selected = []
    for row in top_otus:
        ann = annotations.get(row["otu_id"], {})
        out = dict(row)
        for key in [
            "Tax", "SpeciesRep", "GenomeCount", "TypeStrains", "Strains",
            "Genomes", "RepSpecies", "Taxaname", "GoldHit", "GoldID", "GoldScore",
            "candidate_species", "match_confidence", "confidence_reason",
        ]:
            out[key] = ann.get(key, "")
        out["taxonomy_domain"] = taxonomy_domain(out)
        annotated.append(out)
        if (
            confidence_rank(out.get("match_confidence", "low")) >= min_rank
            and out.get("candidate_species")
            and (not allowed_domains or out.get("taxonomy_domain") in allowed_domains)
        ):
            selected.append(out)
    write_csv(args.out_dir / "03_top_otus_with_taxonomy.csv", annotated)
    write_csv(args.out_dir / "04_high_confidence_species_occurrences.csv", selected)
    print(f"  annotated OTUs: {len(annotated)}")
    print(f"  high-confidence species occurrences: {len(selected)}")

    print("[5/8] Aggregating candidate species across selected latitude samples...")
    aggregated = aggregate_candidates_by_latitude(selected)
    write_csv(args.out_dir / "05_candidate_species_ranked_before_api.csv", aggregated)
    print(f"  unique candidate species before API: {len(aggregated)}")

    if not args.query_apis:
        print("[6/8] API lookup skipped. Re-run with --query-apis for taxid/proteome IDs.")
        return

    print("[6/8] Querying NCBI Taxonomy and UniProt Proteomes...")
    cache_path = args.out_dir / "api_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {"ncbi_taxid": {}, "uniprot_proteome": {}, "downloads": {}}

    mapped = []
    ready_by_group: dict[str, list[dict]] = defaultdict(list)
    seen_proteomes = set()
    for index, entry in enumerate(aggregated, start=1):
        species = entry["candidate_species"]
        if species in cache["ncbi_taxid"]:
            taxid_record = cache["ncbi_taxid"][species]
        else:
            taxid, note = query_ncbi_taxid(species, email=args.ncbi_email)
            taxid_record = {"ncbi_taxid": taxid, "ncbi_note": note}
            cache["ncbi_taxid"][species] = taxid_record
            time.sleep(args.api_sleep)

        taxid = taxid_record.get("ncbi_taxid", "")
        cache_key = taxid or species
        if cache_key in cache["uniprot_proteome"]:
            proteome_record = cache["uniprot_proteome"][cache_key]
        else:
            proteome, note = query_uniprot_proteome(taxid, species)
            proteome_record = dict(proteome)
            proteome_record["uniprot_note"] = note
            cache["uniprot_proteome"][cache_key] = proteome_record
            time.sleep(args.api_sleep)

        out = {**entry, **taxid_record, **proteome_record}
        availability, reason, tpc_ready = classify_tpc_ready(out, args.min_protein_count, args.max_protein_count)
        out["proteome_availability"] = availability
        out["proteome_filter_reason"] = reason
        out["tpc_ready"] = tpc_ready
        out["selection_rank_before_api"] = index
        mapped.append(out)

        proteome_id = out.get("uniprot_proteome_id", "")
        groups_seen = [g for g in str(out.get("latitude_groups_observed", "")).split(";") if g]
        primary_group = groups_seen[0] if groups_seen else "unknown"
        if (
            tpc_ready == "yes"
            and proteome_id
            and proteome_id not in seen_proteomes
            and len(ready_by_group[primary_group]) < args.target_taxa_per_group
        ):
            seen_proteomes.add(proteome_id)
            out["primary_latitude_group"] = primary_group
            ready_by_group[primary_group].append(out)
            print(
                f"  ready {sum(len(v) for v in ready_by_group.values()):3d}/{target_total}: "
                f"{primary_group} {proteome_id} {species}"
            )
        if index % 25 == 0:
            counts = {k: len(v) for k, v in ready_by_group.items()}
            print(f"  queried {index} species; ready by group={counts}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if all(len(ready_by_group[g]) >= args.target_taxa_per_group for g in [
            "low_abs_latitude", "mid_abs_latitude", "high_abs_latitude",
        ]):
            break

    ready_unique = []
    for group in ["low_abs_latitude", "mid_abs_latitude", "high_abs_latitude"]:
        ready_unique.extend(ready_by_group.get(group, []))

    write_csv(args.out_dir / "06_species_proteome_mapping_all_queried.csv", mapped)
    write_csv(args.out_dir / "07_top_unique_tpc_ready_proteomes.csv", ready_unique)
    print(f"  queried species: {len(mapped)}")
    print(f"  ready unique proteomes: {len(ready_unique)}")

    if not args.download_proteomes:
        print("[7/8] FASTA download skipped. Re-run with --download-proteomes if needed.")
        return

    print("[7/8] Downloading FASTA files for selected proteomes...")
    download_rows = []
    for i, row in enumerate(ready_unique, start=1):
        proteome_id = row["uniprot_proteome_id"]
        taxid = row.get("uniprot_taxid") or row.get("ncbi_taxid", "")
        path, note, record_count = download_uniprot_fasta(proteome_id, taxid, args.proteome_dir, args.api_sleep)
        out = dict(row)
        out["local_fasta_path"] = path
        out["download_note"] = note
        out["fasta_record_count"] = record_count
        out["fasta_ready"] = "yes" if record_count > 0 else "no"
        download_rows.append(out)
        print(f"  [{i}/{len(ready_unique)}] {proteome_id} records={record_count} {note}")
    write_csv(args.out_dir / "08_top_unique_tpc_ready_proteomes_with_fastas.csv", download_rows)

    print("[8/8] Writing final strict manifest...")
    strict_rows = []
    for row in download_rows:
        try:
            fasta_record_count = int(float(row.get("fasta_record_count", 0) or 0))
        except ValueError:
            fasta_record_count = 0
        if row.get("fasta_ready") == "yes" and fasta_record_count >= args.min_protein_count:
            strict_rows.append(row)
    write_csv(args.out_dir / "09_final_soil_latitude_species_manifest.csv", strict_rows)
    group_counts = defaultdict(int)
    for row in strict_rows:
        group_counts[row.get("primary_latitude_group", "unknown")] += 1
    print("\nDONE")
    print(f"Selected soil samples: {len(communities)}")
    print(f"High-confidence species occurrences: {len(selected)}")
    print(f"FASTA-ready proteomes: {len(strict_rows)}")
    print(f"FASTA-ready by latitude group: {dict(group_counts)}")
    print(f"Output dir: {args.out_dir}")
    print(f"FASTA dir: {args.proteome_dir}")


if __name__ == "__main__":
    main()
