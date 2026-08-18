#!/usr/bin/env python3
"""Compute ESM2 species embeddings from a manifest of proteome FASTA files."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    sequence = sequence.upper().replace("*", "")
    return "".join(aa if aa in AMINO_ACIDS else "X" for aa in sequence)


def parse_fasta_gz(path: Path):
    header = None
    chunks: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, clean_sequence("".join(chunks))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, clean_sequence("".join(chunks))


def protein_id_from_header(header: str) -> str:
    first = header.split()[0]
    parts = first.split("|")
    if len(parts) >= 2:
        return parts[1]
    return first


def stable_sample(records: list[dict], n: int) -> list[dict]:
    if n <= 0 or len(records) <= n:
        return records
    keyed = []
    for record in records:
        digest = hashlib.sha1(record["header"].encode("utf-8")).hexdigest()
        keyed.append((digest, record))
    keyed.sort(key=lambda item: item[0])
    return [record for _, record in keyed[:n]]


def deduplicate_by_sequence(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        sequence = record["sequence"]
        if sequence in seen:
            continue
        seen.add(sequence)
        unique.append(record)
    return unique


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                clean_key = key.lstrip("\ufeff").strip()
                row[clean_key] = value.strip() if isinstance(value, str) else value
            rows.append(row)
    out = []
    for row in rows:
        fasta = Path(row["local_fasta_path"])
        if not fasta.is_absolute():
            fasta = Path.cwd() / fasta
        row["resolved_fasta_path"] = str(fasta)
        out.append(row)
    return out


def mean_pool(last_hidden_state, attention_mask):
    import torch

    pooled = []
    for hidden, mask in zip(last_hidden_state, attention_mask):
        length = int(mask.sum().item())
        if length <= 2:
            token_hidden = hidden[mask.bool()]
        else:
            token_hidden = hidden[1:length - 1]
        pooled.append(token_hidden.mean(dim=0))
    return torch.stack(pooled, dim=0)


def embed_sequences(sequences: list[str], tokenizer, model, batch_size: int, max_length: int, device, fp16: bool):
    import numpy as np
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            encoded = tokenizer(
                [seq[: max_length - 2] for seq in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.cpu().numpy().astype("float32"))
    matrix = np.concatenate(chunks, axis=0)
    return matrix


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/strict148_esm2_t33_20proteins"))
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max-proteins-per-species", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-fp16", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "species_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ESM2 model: {args.model_name}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    species_rows = []
    protein_rows = []
    skipped_rows = []

    for index, row in enumerate(rows, start=1):
        proteome_id = row["uniprot_proteome_id"]
        species = row.get("candidate_species") or row.get("Taxname") or row.get("uniprot_organism") or proteome_id
        fasta = Path(row["resolved_fasta_path"])
        cache_label = "full_unique" if args.max_proteins_per_species <= 0 else f"n{args.max_proteins_per_species}"
        cache_path = cache_dir / f"{proteome_id}_{cache_label}.npz"
        print(f"[{index}/{len(rows)}] {proteome_id} {species}")
        try:
            if cache_path.exists() and not args.force:
                data = np.load(cache_path, allow_pickle=True)
                species_vec = data["species_embedding"].astype("float32")
                sampled_records = data["sampled_records"].tolist()
                protein_count_fasta = int(data["protein_count_fasta"])
                protein_count_unique = int(data["protein_count_unique"]) if "protein_count_unique" in data else len(sampled_records)
            else:
                records = []
                for header, sequence in parse_fasta_gz(fasta):
                    if sequence:
                        records.append({
                            "header": header,
                            "protein_id": protein_id_from_header(header),
                            "sequence": sequence,
                            "length": len(sequence),
                        })
                if not records:
                    raise ValueError(f"No protein sequences found in {fasta}")
                protein_count_fasta = len(records)
                unique_records = deduplicate_by_sequence(records)
                protein_count_unique = len(unique_records)
                sampled = stable_sample(unique_records, args.max_proteins_per_species)
                matrix = embed_sequences(
                    [record["sequence"] for record in sampled],
                    tokenizer,
                    model,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    fp16=not args.no_fp16,
                )
                species_vec = matrix.mean(axis=0).astype("float32")
                sampled_records = [
                    {
                        "proteome_id": proteome_id,
                        "species_name": species,
                        "protein_id": record["protein_id"],
                        "length": record["length"],
                    }
                    for record in sampled
                ]
                np.savez_compressed(
                    cache_path,
                    species_embedding=species_vec,
                    sampled_records=np.array(sampled_records, dtype=object),
                    protein_count_fasta=protein_count_fasta,
                    protein_count_unique=protein_count_unique,
                )

            for protein_row in sampled_records:
                protein_rows.append(protein_row)

            out = {
                "proteome_id": proteome_id,
                "species_name": species,
                "ncbi_taxid": row.get("ncbi_taxid", ""),
                "uniprot_taxid": row.get("uniprot_taxid", ""),
                "source_temp_C": row.get("abundance_weighted_source_temp_C", ""),
                "embedding_model": args.model_name,
                "embedding_dim": len(species_vec),
                "protein_count_fasta": protein_count_fasta,
                "protein_count_unique": protein_count_unique,
                "protein_count_embedded": len(sampled_records),
                "fasta_path": str(fasta),
            }
            for i, value in enumerate(species_vec):
                out[f"z_{i:04d}"] = f"{float(value):.7g}"
            species_rows.append(out)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({
                "proteome_id": proteome_id,
                "species_name": species,
                "fasta_path": str(fasta),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  [skip] {type(exc).__name__}: {exc}")

        if species_rows:
            fields = list(species_rows[0])
            write_tsv(args.out_dir / "species_embeddings.tsv", species_rows, fields)
        if protein_rows:
            write_tsv(
                args.out_dir / "sampled_proteins.tsv",
                protein_rows,
                ["proteome_id", "species_name", "protein_id", "length"],
            )
        if skipped_rows:
            write_tsv(args.out_dir / "skipped_species.tsv", skipped_rows, list(skipped_rows[0]))

    summary = {
        "manifest": str(args.manifest),
        "model_name": args.model_name,
        "max_proteins_per_species": args.max_proteins_per_species,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "species_requested": len(rows),
        "species_embedded": len(species_rows),
        "species_skipped": len(skipped_rows),
        "embedding_dim": 1280 if species_rows else "",
    }
    write_tsv(args.out_dir / "run_summary.tsv", [summary], list(summary))
    print("\nDONE")
    for key, value in summary.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
