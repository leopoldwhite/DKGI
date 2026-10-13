"""Load one prepared ARK graph with its BM25 index (copied into the ARK checkout; used by the graph-tool service)."""

from importlib.metadata import version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_graph(dataset):
    from src.core.graph import Graph
    directory = ROOT / "data/graphs" / dataset
    prepared = read(directory / "manifest.json")
    expected = {"dataset": dataset, "tantivy_version": version("tantivy"), "num_docs": prepared["outputs"]["nodes.parquet"]["rows"]}
    receipt = directory / "index_manifest.json"
    existing = (directory / "index").exists()
    if existing:
        if not receipt.exists():
            raise ValueError(f"Untracked or partial index: {directory / 'index'}; inspect before rebuilding")
        pinned = read(receipt)
        if any(pinned.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stale BM25 index: {dataset}")
    graph = Graph(dataset, directory)
    if graph.index.index.searcher().num_docs != graph.nodes_df.height:
        raise ValueError(f"Incomplete BM25 index: {dataset}")
    if not existing:
        save(receipt, {**expected, "status": "complete"})
    return graph
