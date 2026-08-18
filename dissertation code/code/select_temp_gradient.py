#!/usr/bin/env python3
"""Select ~150 high-confidence proteome-ready taxa across sample temp_C gradient.

This script is for the current analysis design:

1. Use sample-level temp_C as an environmental gradient.
2. Sample communities evenly across that gradient.
3. Extract high-abundance OTUs from those communities.
4. Keep species-level, Bacteria/Archaea, high-confidence taxa.
5. Query NCBI Taxonomy + UniProt Proteomes.
6. Rank unique proteomes by abundance/occurrence and keep top N.

The output preserves sample IDs, temp_C, OTU IDs, relative abundance and the
matched proteome, so predicted species TPCs can later be mapped back to the
communities where each taxon occurred.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIOM = ROOT / "data/raw/samples-otus.97.mapped.metag.minfilter.refilt.biom"
DEFAULT_OTU_INFO = ROOT / "data/raw/otus.info.tsv"
DEFAULT_MEASUREMENTS = ROOT / "data/raw/samples.measurements.tsv"

HIGH_LEVEL_TAXA = {
    "archaea",
    "bacteria",
    "eukaryota",
    "fungi",
    "metazoa",
    "proteobacteria",
    "actinobacteria",
    "firmicutes",
    "bacteroidetes",
    "acidobacteria",
}
BAD_SPECIES_TOKENS = {
    "sp.",
    "spp.",
    "bacterium",
    "archaeon",
    "uncultured",
    "environmental",
    "metagenome",
    "subclade",
    "group",
    "clade",
}


def set_large_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def parse_float(value: str | None) -> float | None:
    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_temperature_metadata(path: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("Meta_variable") != "temp_C":
                continue
            sample_id = row.get("SampleID", "").strip()
            temp = parse_float(row.get("Value"))
            if sample_id and temp is not None:
                values[sample_id].append(temp)
    return {sample_id: sum(xs) / len(xs) for sample_id, xs in values.items() if xs}


def normalize_biom_sample_id(sample_id: str) -> str:
    return sample_id.split(".")[-1]


def read_biom_sample_map(biom_path: Path, temp_sample_ids: set[str]) -> dict[str, tuple[str, int]]:
    import h5py

    found: dict[str, tuple[str, int]] = {}
    with h5py.File(biom_path, "r") as handle:
        sample_ids = handle["sample"]["ids"]
        for idx, raw in enumerate(sample_ids):
            biom_id = raw.decode() if isinstance(raw, bytes) else str(raw)
            metadata_id = normalize_biom_sample_id(biom_id)
            if metadata_id in temp_sample_ids and metadata_id not in found:
                found[metadata_id] = (biom_id, idx)
    return found


def evenly_select_by_temp(sample_temps: dict[str, float], available_ids: set[str], n: int) -> list[str]:
    ids = [sid for sid in sample_temps if sid in available_ids]
    ids.sort(key=lambda sid: sample_temps[sid])
    if len(ids) <= n:
        return ids
    selected = []
    for i in range(n):
        idx = round(i * (len(ids) - 1) / (n - 1))
        selected.append(ids[idx])
    return list(dict.fromkeys(selected))


def extract_top_otus(biom_path: Path, communities: list[dict], candidate_pool: int) -> list[dict]:
    import h5py

    rows = []
    with h5py.File(biom_path, "r") as handle:
        obs_ids = [raw.decode() if isinstance(raw, bytes) else str(raw) for raw in handle["observation"]["ids"][:]]
        data = handle["sample"]["matrix"]["data"]
        indices = handle["sample"]["matrix"]["indices"]
        indptr = handle["sample"]["matrix"]["indptr"]
        for community in communities:
            start = int(indptr[int(community["biom_sample_index"])])
            end = int(indptr[int(community["biom_sample_index"]) + 1])
            counts = data[start:end]
            obs_indices = indices[start:end]
            total = float(sum(int(x) for x in counts))
            pairs = sorted(((int(count), int(obs_idx)) for count, obs_idx in zip(counts, obs_indices)), reverse=True)
            for rank, (count, obs_idx) in enumerate(pairs[:candidate_pool], start=1):
                rows.append({
                    "metadata_sample_id": community["metadata_sample_id"],
                    "biom_sample_id": community["biom_sample_id"],
                    "temp_C": f"{community['temp_C']:.8g}",
                    "otu_id": obs_ids[obs_idx],
                    "otu_rank": rank,
                    "count": count,
                    "relative_abundance": f"{count / total:.8g}" if total > 0 else "0",
                    "sample_nonzero_otus": end - start,
                })
    return rows


def clean_candidate_name(name: str) -> str:
    name = name.strip().strip('"').strip("'")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*\[[^\]]+\]\s*", " ", name)
    name = name.replace(" et al.", "").strip()
    return re.sub(r"\s+", " ", name)


def looks_like_species(name: str) -> bool:
    name = clean_candidate_name(name)
    if not name or name.lower() in HIGH_LEVEL_TAXA:
        return False
    words = name.split()
    if words and words[0].lower() == "candidatus":
        words = words[1:]
    if len(words) < 2:
        return False
    if any(word.lower() in BAD_SPECIES_TOKENS for word in words):
        return False
    if words[0].lower() in HIGH_LEVEL_TAXA:
        return False
    return bool(re.match(r"^[A-Z][A-Za-z.-]+$", words[0])) and bool(re.match(r"^[a-z][A-Za-z0-9_.-]+$", words[1]))


def first_species_from_rep_species(value: str) -> str:
    for item in value.split(","):
        candidate = clean_candidate_name(item)
        if looks_like_species(candidate):
            return candidate
    return ""


def classify_candidate(row: dict) -> tuple[str, str, str]:
    rep_species = row.get("RepSpecies", "").strip()
    taxaname = clean_candidate_name(row.get("Taxaname", ""))
    tax = row.get("Tax", "")
    try:
        genome_count = int(row.get("GenomeCount") or 0)
    except ValueError:
        genome_count = 0
    if rep_species:
        species = first_species_from_rep_species(rep_species)
        if species:
            if genome_count > 0:
                return species, "high", "RepSpecies present and GenomeCount > 0"
            return species, "medium", "RepSpecies present but GenomeCount = 0"
    if looks_like_species(taxaname):
        if genome_count > 0:
            return taxaname, "high", "Taxaname is species-like and GenomeCount > 0"
        return taxaname, "medium", "Taxaname is species-like but GenomeCount = 0"
    tax_parts = [clean_candidate_name(x) for x in tax.split(";") if clean_candidate_name(x)]
    if tax_parts:
        last = tax_parts[-1]
        if last and last.lower() not in HIGH_LEVEL_TAXA and genome_count > 0:
            return last, "medium", "Genus-level candidate with GenomeCount > 0"
    return "", "low", "No reliable species-level or genome-supported candidate"


def load_otu_annotations(otu_info_path: Path, needed_otus: set[str]) -> dict[str, dict]:
    set_large_csv_field_limit()
    fields = [
        "OTU", "Tax", "SpeciesRep", "GenomeCount", "TypeStrains", "Strains",
        "Genomes", "RepSpecies", "Taxaname", "GoldHit", "GoldID", "GoldScore",
    ]
    annotations: dict[str, dict] = {}
    with otu_info_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            hit_ids = set(row.get("OTU", "").split(";")) & needed_otus
            if not hit_ids:
                continue
            compact = {field: row.get(field, "") for field in fields}
            candidate_species, confidence, reason = classify_candidate(compact)
            compact["candidate_species"] = candidate_species
            compact["match_confidence"] = confidence
            compact["confidence_reason"] = reason
            for otu_id in hit_ids:
                annotations.setdefault(otu_id, compact)
            if len(annotations) == len(needed_otus):
                break
    return annotations


def confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(confidence, 0)


def taxonomy_domain(row: dict) -> str:
    tax = row.get("Tax", "")
    if not tax:
        return ""
    return clean_candidate_name(tax.split(";")[0])


def http_json(url: str, timeout: int = 30) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "top150-temp-gradient-tpc/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def query_ncbi_taxid(species: str, email: str | None = None) -> tuple[str, str]:
    term = f'"{species}"[Scientific Name]'
    params = {"db": "taxonomy", "term": term, "retmode": "json", "retmax": "5"}
    if email:
        params["email"] = email
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    try:
        payload = http_json(url)
        ids = payload.get("esearchresult", {}).get("idlist", [])
        if ids:
            return ids[0], "exact_scientific_name_query"
    except Exception as exc:  # noqa: BLE001
        return "", f"ncbi_error:{type(exc).__name__}:{exc}"
    params["term"] = species
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    try:
        payload = http_json(url)
        ids = payload.get("esearchresult", {}).get("idlist", [])
        if ids:
            return ids[0], "fallback_name_query"
    except Exception as exc:  # noqa: BLE001
        return "", f"ncbi_error:{type(exc).__name__}:{exc}"
    return "", "no_ncbi_taxid_found"


def parse_uniprot_proteome_result(result: dict) -> dict:
    proteome_id = str(result.get("id") or result.get("upid") or "")
    protein_count = result.get("proteinCount") or result.get("protein_count") or ""
    proteome_type = result.get("proteomeType") or result.get("proteome_type") or ""
    taxon = result.get("taxonomy") or {}
    taxon_id = taxon.get("taxonId") if isinstance(taxon, dict) else ""
    organism = taxon.get("scientificName") if isinstance(taxon, dict) else ""
    return {
        "uniprot_proteome_id": proteome_id,
        "uniprot_proteome_type": str(proteome_type),
        "uniprot_protein_count": str(protein_count),
        "uniprot_taxid": str(taxon_id or ""),
        "uniprot_organism": str(organism or ""),
    }


def query_uniprot_proteome(taxid: str, species: str) -> tuple[dict, str]:
    queries = []
    if taxid:
        queries.extend([f"organism_id:{taxid}", f"taxonomy_id:{taxid}"])
    if species:
        queries.append(species)
    last_error = ""
    for query in queries:
        params = {"query": query, "format": "json", "size": "10"}
        url = "https://rest.uniprot.org/proteomes/search?" + urllib.parse.urlencode(params)
        try:
            payload = http_json(url)
            results = payload.get("results", [])
            if not results:
                continue
            parsed = [parse_uniprot_proteome_result(result) for result in results]
            parsed = [row for row in parsed if row["uniprot_proteome_id"]]
            if not parsed:
                continue
            parsed.sort(key=lambda row: (
                0 if "reference" in row["uniprot_proteome_type"].lower() else 1,
                -int(row["uniprot_protein_count"] or 0) if str(row["uniprot_protein_count"]).isdigit() else 0,
            ))
            return parsed[0], f"uniprot_query:{query}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"uniprot_error:{type(exc).__name__}:{exc}"
    return {}, last_error or "no_uniprot_proteome_found"


def normalize_name_tokens(name: str) -> list[str]:
    name = clean_candidate_name(name).lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    tokens = [token for token in name.split() if token and token not in {"strain", "subsp", "subspecies"}]
    if tokens and tokens[0] == "candidatus":
        tokens = tokens[1:]
    return tokens


def species_name_matches(candidate_species: str, uniprot_organism: str) -> bool:
    candidate = normalize_name_tokens(candidate_species)
    organism = normalize_name_tokens(uniprot_organism)
    if len(candidate) < 2 or len(organism) < 2:
        return False
    return candidate[:2] == organism[:2]


def classify_tpc_ready(row: dict, min_protein_count: int, max_protein_count: int) -> tuple[str, str, str]:
    taxid = str(row.get("ncbi_taxid", "") or "").strip()
    proteome_id = str(row.get("uniprot_proteome_id", "") or "").strip()
    proteome_type = str(row.get("uniprot_proteome_type", "") or "").strip()
    protein_count_raw = str(row.get("uniprot_protein_count", "") or "").strip()
    candidate_species = str(row.get("candidate_species", "") or "").strip()
    uniprot_organism = str(row.get("uniprot_organism", "") or "").strip()
    if not taxid:
        return "exclude_no_taxid", "No NCBI taxid returned for candidate taxon", "no"
    if not proteome_id:
        return "exclude_no_proteome", "NCBI taxid found, but UniProt returned no proteome ID", "no"
    try:
        protein_count = int(float(protein_count_raw))
    except ValueError:
        return "manual_check_missing_protein_count", "UniProt proteome found, but protein count is missing", "no"
    if protein_count < min_protein_count:
        return "manual_check_low_protein_count", f"Protein count {protein_count} below threshold {min_protein_count}", "no"
    if protein_count > max_protein_count:
        return "manual_check_high_protein_count", f"Protein count {protein_count} above threshold {max_protein_count}", "no"
    if candidate_species and uniprot_organism and not species_name_matches(candidate_species, uniprot_organism):
        return "manual_check_name_mismatch", f"Candidate species '{candidate_species}' does not match UniProt organism '{uniprot_organism}'", "no"
    if "excluded" in proteome_type.lower():
        return "manual_check_excluded_proteome", "UniProt proteome type is Excluded", "no"
    if "reference" in proteome_type.lower() and "non reference" not in proteome_type.lower():
        return "usable_reference", "Reference proteome with reasonable protein count and matching organism", "yes"
    return "usable_non_reference", "Non-reference proteome with reasonable protein count and matching organism", "yes"


def download_uniprot_fasta(proteome_id: str, taxid: str, out_dir: Path, sleep_seconds: float) -> tuple[str, str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{proteome_id}.fasta.gz"
    if out_path.exists() and out_path.stat().st_size > 0 and count_fasta_records(out_path) > 0:
        return str(out_path), "cached", count_fasta_records(out_path)
    queries = [f"proteome:{proteome_id}"]
    if taxid:
        queries.append(f"organism_id:{taxid}")
    last_note = ""
    for query in queries:
        params = {"compressed": "true", "format": "fasta", "query": query}
        url = "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "top150-temp-gradient-tpc/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            if not data.startswith(b"\x1f\x8b"):
                data = gzip.compress(data)
            out_path.write_bytes(data)
            n = count_fasta_records(out_path)
            time.sleep(sleep_seconds)
            if n > 0:
                return str(out_path), f"downloaded:{query}", n
            last_note = f"empty_download:{query}"
        except Exception as exc:  # noqa: BLE001
            last_note = f"download_error:{query}:{type(exc).__name__}:{exc}"
    return str(out_path) if out_path.exists() else "", last_note, count_fasta_records(out_path)


def count_fasta_records(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.startswith(">"))
    except OSError:
        return 0


def aggregate_candidates(rows: list[dict]) -> list[dict]:
    by_species: dict[str, dict] = {}
    for row in rows:
        species = row.get("candidate_species", "")
        if not species:
            continue
        rel = parse_float(row.get("relative_abundance")) or 0.0
        temp = parse_float(row.get("temp_C"))
        entry = by_species.setdefault(species, {
            "candidate_species": species,
            "taxonomy_domain": row.get("taxonomy_domain", ""),
            "match_confidence": row.get("match_confidence", ""),
            "confidence_reason": row.get("confidence_reason", ""),
            "n_occurrences": 0,
            "n_samples": set(),
            "sum_relative_abundance": 0.0,
            "max_relative_abundance": 0.0,
            "abundance_weighted_temp_sum": 0.0,
            "abundance_weight_sum": 0.0,
            "temps": [],
            "example_otu_id": row.get("otu_id", ""),
            "example_metadata_sample_id": row.get("metadata_sample_id", ""),
        })
        entry["n_occurrences"] += 1
        entry["n_samples"].add(row.get("metadata_sample_id", ""))
        entry["sum_relative_abundance"] += rel
        entry["max_relative_abundance"] = max(entry["max_relative_abundance"], rel)
        if temp is not None:
            entry["temps"].append(temp)
            entry["abundance_weighted_temp_sum"] += rel * temp
            entry["abundance_weight_sum"] += rel
    out = []
    for entry in by_species.values():
        temps = entry.pop("temps")
        n_samples = len(entry.pop("n_samples"))
        weight = entry.pop("abundance_weight_sum")
        weighted_temp = entry.pop("abundance_weighted_temp_sum")
        entry["n_samples"] = n_samples
        entry["mean_relative_abundance_per_occurrence"] = entry["sum_relative_abundance"] / max(entry["n_occurrences"], 1)
        entry["mean_source_temp_C"] = sum(temps) / len(temps) if temps else ""
        entry["min_source_temp_C"] = min(temps) if temps else ""
        entry["max_source_temp_C"] = max(temps) if temps else ""
        entry["abundance_weighted_source_temp_C"] = weighted_temp / weight if weight > 0 else entry["mean_source_temp_C"]
        out.append(entry)
    out.sort(key=lambda r: (r["n_samples"], r["sum_relative_abundance"], r["max_relative_abundance"]), reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--biom", type=Path, default=DEFAULT_BIOM)
    parser.add_argument("--otu-info", type=Path, default=DEFAULT_OTU_INFO)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/top150_temp_gradient_species"))
    parser.add_argument("--proteome-dir", type=Path, default=Path("data/proteomes_top150"))
    parser.add_argument("--sample-count", type=int, default=600)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--target-taxa", type=int, default=150)
    parser.add_argument("--min-confidence", choices=["high", "medium", "low"], default="high")
    parser.add_argument("--allowed-domains", default="Bacteria,Archaea")
    parser.add_argument("--min-protein-count", type=int, default=1000)
    parser.add_argument("--max-protein-count", type=int, default=12000)
    parser.add_argument("--api-sleep", type=float, default=0.34)
    parser.add_argument("--ncbi-email", default="")
    parser.add_argument("--query-apis", action="store_true")
    parser.add_argument("--download-proteomes", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    allowed_domains = {x.strip() for x in args.allowed_domains.split(",") if x.strip()}
    min_rank = confidence_rank(args.min_confidence)

    print("[1/7] Reading temperature metadata...")
    sample_temps = read_temperature_metadata(args.measurements)
    print(f"  samples with temp_C: {len(sample_temps)}")

    print("[2/7] Matching temp samples to BIOM sample IDs...")
    biom_map = read_biom_sample_map(args.biom, set(sample_temps))
    print(f"  temp samples present in BIOM: {len(biom_map)}")
    selected_ids = evenly_select_by_temp(sample_temps, set(biom_map), args.sample_count)
    communities = [
        {
            "metadata_sample_id": sid,
            "biom_sample_id": biom_map[sid][0],
            "biom_sample_index": biom_map[sid][1],
            "temp_C": sample_temps[sid],
        }
        for sid in selected_ids
    ]
    write_csv(args.out_dir / "01_selected_samples_by_temp_gradient.csv", communities)
    print(f"  selected samples: {len(communities)}")

    print("[3/7] Extracting top OTUs by relative abundance...")
    top_otus = extract_top_otus(args.biom, communities, args.candidate_pool)
    write_csv(args.out_dir / "02_top_otus_from_selected_samples.csv", top_otus)
    print(f"  top OTU rows: {len(top_otus)}")

    print("[4/7] Joining OTU taxonomy annotations...")
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

    print("[5/7] Aggregating candidate species across selected samples...")
    aggregated = aggregate_candidates(selected)
    write_csv(args.out_dir / "05_candidate_species_ranked_before_api.csv", aggregated)
    print(f"  unique candidate species before API: {len(aggregated)}")

    if not args.query_apis:
        print("[6/7] API lookup skipped. Re-run with --query-apis for taxid/proteome IDs.")
        return

    print("[6/7] Querying NCBI Taxonomy and UniProt Proteomes...")
    cache_path = args.out_dir / "api_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {"ncbi_taxid": {}, "uniprot_proteome": {}, "downloads": {}}

    mapped = []
    ready_unique = []
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
        if tpc_ready == "yes" and proteome_id and proteome_id not in seen_proteomes:
            seen_proteomes.add(proteome_id)
            ready_unique.append(out)
            print(f"  ready {len(ready_unique):3d}/{args.target_taxa}: {proteome_id} {species}")
        if index % 25 == 0:
            print(f"  queried {index} species; ready unique proteomes={len(ready_unique)}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(ready_unique) >= args.target_taxa:
            break

    write_csv(args.out_dir / "06_species_proteome_mapping_all_queried.csv", mapped)
    write_csv(args.out_dir / "07_top_unique_tpc_ready_proteomes.csv", ready_unique)
    print(f"  queried species: {len(mapped)}")
    print(f"  ready unique proteomes: {len(ready_unique)}")

    if not args.download_proteomes:
        print("[7/7] FASTA download skipped. Re-run with --download-proteomes if needed.")
        return

    print("[7/7] Downloading FASTA files for selected proteomes...")
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
    ready_fastas = sum(1 for row in download_rows if row["fasta_ready"] == "yes")
    print("\nDONE")
    print(f"Selected samples: {len(communities)}")
    print(f"High-confidence species occurrences: {len(selected)}")
    print(f"Ready unique proteomes: {len(ready_unique)}")
    print(f"FASTA-ready proteomes: {ready_fastas}")
    print(f"Output dir: {args.out_dir}")
    print(f"FASTA dir: {args.proteome_dir}")


if __name__ == "__main__":
    main()
