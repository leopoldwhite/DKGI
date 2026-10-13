"""Export the full STaRK knowledge bases to files (nodes JSONL, edges TSV, schema, DATA_FORMAT.txt); no API calls."""

import argparse
import ast
from collections import Counter
import csv
import gc
import json
import math
from pathlib import Path
import pickle
import random

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "STaRK"
SEED = 20260924
SHARD_SIZE = 10000
DATASETS = ("prime", "mag", "amazon")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def clean(value):
    """Keep all metadata fields; encode non-finite numbers as JSON null."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    raise TypeError(f"Unsupported metadata type: {type(value)}")


def choose(rows, count):
    ordered = sorted(rows, key=lambda row: int(row["query_id"]))
    return random.Random(SEED).sample(ordered, count) if count else ordered


def prepare(name):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    report_path = ROOT / "prepared" / f"{name}.json"
    if report_path.exists():
        print(f"Already prepared: {name}; keeping the existing manifest.", flush=True)
        return
    corpus = ROOT / "corpus" / name
    if corpus.exists():
        raise RuntimeError(f"Partial export exists; inspect before continuing: {corpus}")
    source_report = json.loads((SOURCE / "verification.json").read_text())
    expected = source_report["datasets"][name]
    graph = SOURCE / expected["graph_path"]
    source_files = {str(path.relative_to(SOURCE)): path.stat().st_size for path in sorted(graph.glob("*")) if path.is_file()}
    with (graph / "node_type_dict.pkl").open("rb") as stream:
        node_names = pickle.load(stream)
    with (graph / "edge_type_dict.pkl").open("rb") as stream:
        edge_names = pickle.load(stream)
    node_types = torch.load(graph / "node_types.pt", map_location="cpu", weights_only=True).numpy()
    candidate_types = ["product"] if name == "amazon" else ["paper"] if name == "mag" else list(node_names.values())
    candidates = {i for i, kind in enumerate(node_types) if node_names[int(kind)] in candidate_types}
    assert len(node_types) == expected["nodes"] and candidates
    (corpus / "nodes").mkdir(parents=True)
    (corpus / "edges").mkdir()
    (corpus / "candidate_ids.txt").write_text("".join(f"{i}\n" for i in sorted(candidates)))
    print(f"{name}: exporting {len(node_types):,} nodes; {len(candidates):,} candidates", flush=True)
    with (graph / "node_info.pkl").open("rb") as stream:
        nodes = pickle.load(stream)
    assert len(nodes) == len(node_types)
    sample_ids = set(random.Random(SEED).sample(range(len(nodes)), 100))
    samples = {}
    for start in range(0, len(nodes), SHARD_SIZE):
        with (corpus / "nodes" / f"{start // SHARD_SIZE:04d}.jsonl").open("w", encoding="utf-8") as out:
            for node_id in range(start, min(start + SHARD_SIZE, len(nodes))):
                record = {"node_id": node_id, "node_type": node_names[int(node_types[node_id])],
                          "is_candidate": node_id in candidates, "attributes": clean(nodes[node_id])}
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if node_id in sample_ids:
                    samples[node_id] = record
        if start % 100000 == 0:
            print(f"{name}: wrote nodes through {min(start + SHARD_SIZE, len(nodes)):,}", flush=True)
    checked, node_count = 0, 0
    for path in sorted((corpus / "nodes").glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                assert record["node_id"] == node_count
                node_count += 1
                if record["node_id"] in samples:
                    assert record == samples[record["node_id"]]
                    checked += 1
    assert checked == len(samples) and node_count == len(nodes)
    del nodes, samples
    gc.collect()
    edges = torch.load(graph / "edge_index.pt", map_location="cpu", weights_only=True).numpy()
    edge_types = torch.load(graph / "edge_types.pt", map_location="cpu", weights_only=True).numpy()
    assert edges.shape == (2, expected["stored_edges"]) and len(edge_types) == edges.shape[1]
    assert edges.min() >= 0 and edges.max() < len(node_types)
    counts = {}
    for kind, label in edge_names.items():
        subset = edges[:, edge_types == kind].T
        counts[str(kind)] = len(subset)
        np.savetxt(corpus / "edges" / f"{kind:02d}.tsv", subset, fmt="%d", delimiter="\t",
                   header="source_id\ttarget_id", comments="")
    assert sum(counts.values()) == edges.shape[1]
    del edges, edge_types
    schema = {"dataset": name, "nodes": len(node_types), "candidate_count": len(candidates),
              "candidate_types": candidate_types, "node_types": node_names, "relation_types": edge_names,
              "stored_edges_by_type": counts, "node_shard_size": SHARD_SIZE,
              "representation": "Full source node attributes; non-finite numbers become null. Stored edge rows preserved.",
              "graph_semantics": "Edge files retain source endpoint order and duplicates. The official default SKB treats adjacency as undirected; inspect both endpoints for neighbors. Relation labels and entity types retain their meaning."}
    save(corpus / "schema.json", schema)
    (corpus / "DATA_FORMAT.txt").write_text(
        "Node records: nodes/NNNN.jsonl, shard number = node_id // 10000.\n"
        "Each line has node_id (benchmark ID), node_type, is_candidate, and full attributes.\n"
        "IDs inside attributes can be external source IDs; return node_id in predictions.\n"
        "Graph edges: edges/RR.tsv, source_id and target_id columns with a header.\n"
        "RR is the relation type ID from schema.json. Stored edge rows are not truncated.\n"
        "For adjacency, inspect either endpoint; the official default graph is undirected.\n"
        "Preserve relation meaning when interpreting directions or entity roles.\n"
        "Only IDs in candidate_ids.txt are valid retrieval outputs. Other nodes are usable evidence.\n"
        "Use shell tools or Python's json/csv modules to read these files.\n"
    )
    private = ROOT / "private" / name
    private.mkdir(parents=True, mode=0o700)
    (ROOT / "private").chmod(0o700)
    qa = SOURCE / "data/qa" / name
    with (qa / "stark_qa/stark_qa.csv").open() as stream:
        all_rows = {int(row["id"]): row for row in csv.DictReader(stream)}
    split_counts, coverage_issues = {}, {}
    for split in ("train", "val", "test", "test-0.1", "human_generated_eval"):
        if split == "human_generated_eval":
            path = qa / "stark_qa/stark_qa_human_generated_eval.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        else:
            path = qa / "split" / f"{split}.index"
            rows = [all_rows[int(value)] for value in path.read_text().split()]
        public_path = ROOT / "public" / name / f"{split}.jsonl"
        public_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_issues[split] = []
        with public_path.open("w") as out, (private / f"{split}.jsonl").open("w") as gold:
            for row in rows:
                labels = ast.literal_eval(row["answer_ids"])
                assert labels and all(type(i) is int and 0 <= i < len(node_types) for i in labels)
                unavailable = set(labels) - candidates
                if unavailable:
                    coverage_issues[split].append({"query_id": str(row["id"]),
                        "outside_candidate_ids": sorted(unavailable),
                        "node_types": [node_names[int(node_types[i])] for i in sorted(unavailable)],
                        "candidate_coverage": len(set(labels) & candidates) / len(set(labels))})
                public = {"dataset": name, "split": split, "query_id": str(row["id"]), "query": row["query"],
                          "corpus_dir": str(corpus)}
                out.write(json.dumps(public, ensure_ascii=False) + "\n")
                gold.write(json.dumps({"query_id": str(row["id"]), "answer_ids": labels}) + "\n")
        (private / f"{split}.jsonl").chmod(0o600)
        split_counts[split] = len(rows)
    save(private / "candidate_coverage_issues.json", coverage_issues)
    source_files.update({str(path.relative_to(SOURCE)): path.stat().st_size for path in qa.rglob("*") if path.is_file()})
    corpus_files = {str(path.relative_to(corpus)): path.stat().st_size for path in corpus.rglob("*") if path.is_file()}
    report = {"dataset": name, "revision": source_report["revision"], "seed": SEED,
              "schema": schema, "qa_counts": split_counts, "node_roundtrip_samples": checked,
              "source_files_bytes": source_files, "corpus_files_bytes": corpus_files,
              "corpus_bytes": sum(path.stat().st_size for path in corpus.rglob("*") if path.is_file()),
              "node_type_counts": dict(Counter(node_names[int(kind)] for kind in node_types)),
              "gold_outside_candidates_queries": {key: len(value) for key, value in coverage_issues.items()},
              "all_gold_exist_in_graph": True}
    save(report_path, report)
    print(f"{name}: preparation complete ({report['corpus_bytes']:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    for name in dict.fromkeys(args.datasets):
        prepare(name)


if __name__ == "__main__":
    main()
