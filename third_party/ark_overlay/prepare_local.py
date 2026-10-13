"""Prepare complete local STaRK graphs for ARK without loading QA or calling models."""

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.summary_utils import (
    add_summary_amazon,
    add_summary_to_mag,
    add_summary_to_prime,
)

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
DATASETS = ("prime", "mag", "amazon")
NODE_SCHEMA = pa.schema([
    ("index", pa.int64()), ("name", pa.string()),
    ("type", pa.string()), ("summary", pa.string()),
])
EDGE_SCHEMA = pa.schema([
    ("start_node_index", pa.int64()), ("end_node_index", pa.int64()),
    ("type", pa.string()),
])
RULES = [
    "Every source node is retained; index is the explicit benchmark node_id.",
    "Use the pinned ARK summary_utils functions, including their text clipping rules.",
    "Full graph coverage does not imply identical text to the full-attribute export of prepare.py.",
    "Keep edge endpoint order, relation labels, reverse edges, and all nodes.",
    "Amazon alone removes identical (start, end, type) rows, matching ARK preprocessing.",
    "Amazon corpus already includes metadata nodes/edges; do not append its base graph again.",
    "Prime details and Amazon reviews/QA are serialized as in ARK preprocessing.",
    "Non-list Amazon reviews become an empty list instead of the upstream null-iteration error.",
    "Missing/null name and brand text use pandas-style nan; MAG count columns use float values as in the upstream mixed-node DataFrame.",
    "Parse string-valued Amazon structured fields with literal_eval instead of upstream eval.",
    "Null MAG abstracts become empty strings; other nulls retain corpus values.",
    "The source export already replaced non-finite metadata numbers with null; original distinctions cannot be recovered.",
    "This adapter reads no questions or answer labels and creates no embedding/model requests.",
    "Graph/GraphIndex build the original full-graph BM25 index separately at first load.",
]


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def node_row(dataset, record, compatibility):
    row = dict(record["attributes"])
    row["type"] = record["node_type"]
    if dataset == "prime":
        for field in ("name", "source"):
            if row.get(field) is None:
                row[field] = float("nan")
                compatibility["prime_null_text_to_pandas_nan"] += 1
        if row.get("details") is None:
            compatibility["prime_null_details_to_empty_object"] += 1
        row["details"] = json.dumps(row.get("details") or {})
        name = row["name"]
        summary = add_summary_to_prime(row)
    elif dataset == "amazon":
        if row["type"] == "product":
            for field in ("title", "brand"):
                if row.get(field) is None:
                    row[field] = float("nan")
                    compatibility["amazon_missing_or_null_" + field + "_to_pandas_nan"] += 1
            for field in ("details", "description", "feature"):
                if field not in row:
                    row[field] = None
                    compatibility["amazon_missing_" + field + "_to_none"] += 1
                if isinstance(row.get(field), str):
                    row[field] = ast.literal_eval(row[field])
                    compatibility["amazon_literal_string_" + field] += 1
            for field in ("review", "qa"):
                value = row.get(field)
                if field == "review" and not isinstance(value, list):
                    value = []
                    compatibility["amazon_nonlist_reviews_to_empty_list"] += 1
                row[field] = json.dumps(value if isinstance(value, list) else None)
        names = {"product": "title", "brand": "brand_name",
                 "category": "category_name", "color": "color_name"}
        name = row[names[row["type"]]]
        summary = add_summary_amazon(row)
    else:
        for field in ("PaperCount", "CitationCount"):
            if row["type"] != "paper":
                row[field] = float("nan") if row.get(field) is None else float(row[field])
        for field in ("DisplayName", "title", "Date", "OriginalVenue", "JournalDisplayName",
                      "ConferenceSeriesDisplayName", "ConferenceInstancesDisplayName"):
            if field not in row:
                row[field] = float("nan")
        if row["type"] == "paper" and row.get("abstract") is None:
            row["abstract"] = ""
            compatibility["mag_null_abstract_to_empty_string"] += 1
        names = {"paper": "title", "author": "DisplayName",
                 "institution": "DisplayName", "field_of_study": "DisplayName"}
        name = row[names[row["type"]]]
        summary = add_summary_to_mag(row)
    return {"index": record["node_id"], "name": str(name),
            "type": row["type"], "summary": summary}


def write_nodes(dataset, files, destination, expected, batch_size):
    count = 0
    types, compatibility = Counter(), Counter()
    batch = []
    with pq.ParquetWriter(destination, NODE_SCHEMA, compression="zstd") as writer:
        for path in files:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if type(record["node_id"]) is not int or record["node_id"] != count:
                        raise ValueError(f"Nonconsecutive benchmark node ID in {path}: expected {count}")
                    if record["node_type"] not in expected["schema"]["node_types"].values():
                        raise ValueError(f"Unknown node type at {dataset}:{count}")
                    try:
                        batch.append(node_row(dataset, record, compatibility))
                    except Exception as error:
                        raise ValueError(f"Cannot render ARK summary at {dataset}:{count}") from error
                    types[record["node_type"]] += 1
                    count += 1
                    if len(batch) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(batch, schema=NODE_SCHEMA))
                        batch.clear()
            print(f"{dataset}: rendered {count:,} nodes", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=NODE_SCHEMA))
    if count != expected["schema"]["nodes"]:
        raise ValueError(f"Node count mismatch: {count}")
    if "node_type_counts" in expected and dict(types) != expected["node_type_counts"]:
        raise ValueError("Node type counts differ from prepared manifest")
    return count, dict(types), dict(compatibility)


def write_edges(dataset, corpus, destination, expected, batch_size):
    counts = {}
    schema = expected["schema"]
    batch = []
    with pq.ParquetWriter(destination, EDGE_SCHEMA, compression="zstd") as writer:
        for kind in sorted(schema["relation_types"], key=int):
            path = corpus / "edges" / f"{int(kind):02d}.tsv"
            label, count = schema["relation_types"][kind], 0
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                if next(reader) != ["source_id", "target_id"]:
                    raise ValueError(f"Unexpected edge header: {path}")
                for fields in reader:
                    if len(fields) != 2:
                        raise ValueError(f"Malformed edge in {path}")
                    start, end = map(int, fields)
                    if not (0 <= start < schema["nodes"] and 0 <= end < schema["nodes"]):
                        raise ValueError(f"Edge references a missing node in {path}")
                    batch.append({"start_node_index": start, "end_node_index": end, "type": label})
                    count += 1
                    if len(batch) >= batch_size:
                        writer.write_table(pa.Table.from_pylist(batch, schema=EDGE_SCHEMA))
                        batch.clear()
            if count != schema["stored_edges_by_type"][kind]:
                raise ValueError(f"Edge count mismatch in {path}: {count}")
            counts[label] = count
            print(f"{dataset}: retained {count:,} source edges of type {label}", flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=EDGE_SCHEMA))
    return counts


def prepare_dataset(dataset, corpus, prepared_path, output_root, batch_size):
    expected = json.loads(prepared_path.read_text())
    schema = json.loads((corpus / "schema.json").read_text())
    if expected["dataset"] != dataset or expected["schema"] != schema:
        raise ValueError("Corpus schema differs from prepared manifest")
    node_files = sorted((corpus / "nodes").glob("*.jsonl"))
    files = node_files + sorted((corpus / "edges").glob("*.tsv"))
    files += [corpus / "schema.json", corpus / "candidate_ids.txt"]
    stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    provenance = {"dataset": dataset, "corpus_path": str(corpus.resolve()),
                  "prepared_path": str(prepared_path.resolve()),
                  "source_revision": expected.get("revision", "unknown"),
                  "source_files": [str(path.relative_to(corpus)) for path in files],
                  "implementation_files": ["prepare_local.py", "preprocessing/summary_utils.py"],
                  "ark_revision": subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "packages": {name: version(name) for name in ("pyarrow", "pandas", "numpy", "polars")}}
    destination = output_root / dataset
    manifest_path = destination / "manifest.json"
    if destination.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Partial or untracked output exists: {destination}; inspect before retrying")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Existing output is not complete: {manifest_path}")
        for name, info in manifest["outputs"].items():
            path = destination / name
            if not path.is_file():
                raise ValueError(f"Existing output missing: {path}")
            if pq.ParquetFile(path).metadata.num_rows != info["rows"]:
                raise ValueError(f"Existing output row count mismatch: {path}")
        print(f"{dataset}: verified existing manifest and outputs; skipping", flush=True)
        return manifest
    destination.mkdir(parents=True)
    nodes_path = destination / "nodes.parquet"
    edges_path = destination / "edges.parquet"
    node_count, node_types, compatibility = write_nodes(
        dataset, node_files, nodes_path, expected, batch_size)
    raw_edges = destination / "edges.source.parquet" if dataset == "amazon" else edges_path
    edge_counts = write_edges(dataset, corpus, raw_edges, expected, batch_size * 10)
    if dataset == "amazon":
        import polars as pl
        pl.scan_parquet(raw_edges).unique(
            subset=EDGE_SCHEMA.names, keep="first", maintain_order=True,
        ).sink_parquet(edges_path, compression="zstd")
        raw_edges.unlink()
    for path, stat in stats.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != stat:
            raise ValueError(f"Source changed during conversion: {path}")
    outputs = {}
    for path in (nodes_path, edges_path):
        parquet = pq.ParquetFile(path)
        outputs[path.name] = {"rows": parquet.metadata.num_rows,
                              "bytes": path.stat().st_size}
    manifest = {"status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
                "provenance": provenance, "adapter_rules": RULES,
                "nodes": node_count, "node_type_counts": node_types,
                "source_edge_rows_by_type": edge_counts,
                "source_edge_rows": sum(edge_counts.values()),
                "output_edge_rows": outputs["edges.parquet"]["rows"],
                "exact_duplicate_edges_removed": sum(edge_counts.values()) - outputs["edges.parquet"]["rows"],
                "compatibility_counts": compatibility, "outputs": outputs}
    temporary = manifest_path.with_suffix(".json.tmp")
    dump_json(temporary, manifest)
    temporary.replace(manifest_path)
    print(f"{dataset}: complete; {node_count:,} nodes, {manifest['output_edge_rows']:,} edges", flush=True)
    return manifest


def check():
    counts = Counter()
    prime = {"node_id": 0, "node_type": "drug",
             "attributes": {"name": "Example", "source": "fixture", "details": None}}
    assert node_row("prime", prime, counts)["summary"] == "- name: Example\n- type: drug\n- source: fixture\n"
    amazon = {"node_id": 1, "node_type": "brand", "attributes": {"brand_name": "Example"}}
    assert node_row("amazon", amazon, counts)["summary"] == "brand name: Example"
    product = {"node_id": 0, "node_type": "product", "attributes": {
        "title": "Example", "brand": "", "details": None, "description": [],
        "feature": [], "review": None, "qa": None}}
    assert node_row("amazon", product, counts)["summary"] == "- product: Example\n- reviews: \n"
    mag = {"node_id": 2, "node_type": "author",
           "attributes": {"DisplayName": "Example", "PaperCount": -1, "CitationCount": -1}}
    assert node_row("mag", mag, counts)["summary"] == "- author name: Example\n"
    mag["attributes"]["PaperCount"] = 16
    assert node_row("mag", mag, counts)["summary"] == "- author name: Example\n- author paper count: 16.0\n"
    del product["attributes"]["brand"]
    assert "- brand: nan\n" in node_row("amazon", product, counts)["summary"]
    with tempfile.TemporaryDirectory(prefix="ark-prepare-check-") as temporary:
        root = Path(temporary)
        corpus = root / "corpus"
        (corpus / "nodes").mkdir(parents=True)
        (corpus / "edges").mkdir()
        nodes = corpus / "nodes/0000.jsonl"
        second = dict(prime, node_id=1)
        nodes.write_text(json.dumps(prime) + "\n" + json.dumps(second) + "\n")
        (corpus / "edges/00.tsv").write_text("source_id\ttarget_id\n0\t1\n0\t1\n1\t0\n")
        (corpus / "candidate_ids.txt").write_text("0\n1\n")
        schema = {"nodes": 2, "node_types": {"0": "drug"},
                  "relation_types": {"0": "related"}, "stored_edges_by_type": {"0": 3}}
        dump_json(corpus / "schema.json", schema)
        prepared = {"dataset": "prime", "schema": schema, "node_type_counts": {"drug": 2}}
        prepared_path = root / "prepared.json"
        dump_json(prepared_path, prepared)
        first = prepare_dataset("prime", corpus, prepared_path, root / "out", 1)
        assert first["nodes"] == 2 and first["output_edge_rows"] == 3
        assert prepare_dataset("prime", corpus, prepared_path, root / "out", 1) == first
        edges = pq.read_table(root / "out/prime/edges.parquet").to_pylist()
        assert [row["start_node_index"] for row in edges] == [0, 0, 1]
    print("Offline checks passed: summaries, IDs, duplicate/direction retention, verified reuse.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--check", action="store_true", help="Run only small offline fixture checks")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    if args.check:
        check()
        return
    for dataset in dict.fromkeys(args.datasets):
        prepare_dataset(dataset, PROJECT / "experiments/stark/corpus" / dataset,
                        PROJECT / "experiments/stark/prepared" / f"{dataset}.json",
                        ROOT / "data/graphs", args.batch_size)


if __name__ == "__main__":
    main()
