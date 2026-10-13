"""Check the downloaded STaRK release (extracted files, QA splits, graphs) and print the summary that prepare.py reads.

    python3 experiments/stark/verify_data.py > STaRK/verification.json
"""

import ast
import csv
import json
import pickle
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "STaRK"
DATA = ROOT / "data"
EXPECTED = {
    "amazon": (9100, 81, 1035542),
    "mag": (13323, 84, 1872968),
    "prime": (11204, 98, 129375),
}


def main():
    torch.set_num_threads(4)
    extracted_files = 0
    for archive in sorted(DATA.glob("skb/*/*.zip")):
        target = archive.parent
        if archive.name == "idx_title_abs.zip":
            target = target / "processed" / "paper100M_cache"
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                if entry.is_dir():
                    continue
                path = target / entry.filename
                assert path.stat().st_size == entry.file_size, path
                extracted_files += 1
        print(f"Extracted files verified: {archive.relative_to(DATA)}", file=sys.stderr, flush=True)

    report = {"revision": "88269e23e90587f99476c5dd74e235a0877e69be",
              "extracted_files_verified": extracted_files, "datasets": {}}
    for name, (qa_count, human_count, node_count) in EXPECTED.items():
        graph = DATA / "skb" / name / "processed"
        if name == "amazon":
            graph = graph / "cache" / "brand-category-color"
        node_types = torch.load(graph / "node_types.pt", map_location="cpu", weights_only=True)
        edges = torch.load(graph / "edge_index.pt", map_location="cpu", weights_only=True)
        edge_types = torch.load(graph / "edge_types.pt", map_location="cpu", weights_only=True)
        assert len(node_types) == node_count
        assert edges.shape == (2, len(edge_types))
        assert 0 <= edges.min().item() <= edges.max().item() < node_count
        with (graph / "node_info.pkl").open("rb") as stream:
            node_info = pickle.load(stream)
        assert len(node_info) == node_count
        assert all(i in node_info for i in range(node_count))
        del node_info

        qa_root = DATA / "qa" / name
        for filename, expected in [("stark_qa.csv", qa_count),
                                   ("stark_qa_human_generated_eval.csv", human_count)]:
            with (qa_root / "stark_qa" / filename).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            ids = {int(row["id"]) for row in rows}
            assert len(rows) == len(ids) == expected, (name, filename)
            for row in rows:
                answers = ast.literal_eval(row["answer_ids"])
                assert row["query"].strip() and answers
                assert all(isinstance(i, int) and 0 <= i < node_count for i in answers)
            if filename == "stark_qa.csv":
                query_ids = ids
        splits = {}
        for split in ["train", "val", "test", "test-0.1"]:
            values = [int(i) for i in (qa_root / "split" / f"{split}.index").read_text().split()]
            assert len(values) == len(set(values))
            splits[split] = set(values)
        assert splits["train"] | splits["val"] | splits["test"] == query_ids
        assert sum(len(splits[s]) for s in ["train", "val", "test"]) == len(query_ids)
        assert splits["test-0.1"] <= splits["test"]
        report["datasets"][name] = {
            "synthesized_queries": qa_count, "human_queries": human_count,
            "splits": {key: len(value) for key, value in splits.items()},
            "nodes": node_count, "stored_edges": edges.shape[1],
            "graph_path": str(graph.relative_to(ROOT)),
        }
        del edges, edge_types, node_types
        print(f"QA and KG verified: {name}", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
