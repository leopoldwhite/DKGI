"""Exact offline name indexes, constructed without labels."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
EXPECTED = {"mag": 700244, "prime": 129375}


def norm(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def names_for(node, dataset):
    attrs = node["attributes"]
    if dataset == "mag":
        return [attrs.get("title"), attrs.get("OriginalTitle")], []
    details = attrs.get("details") or {}
    aliases = details.get("alias") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return [attrs.get("name"), details.get("name"), *aliases], aliases


def build(dataset):
    target = ROOT / "index" / f"{dataset}.json"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return json.loads(target.read_text())["metadata"]
    index, covered, alias_count, sources = defaultdict(set), set(), 0, {}
    for path in sorted((ROOT.parent / "corpus" / dataset / "nodes").glob("*.jsonl")):
        lines = 0
        with path.open("rb") as source:
            for line in source:
                lines += 1
                node = json.loads(line)
                if dataset == "mag" and not node["is_candidate"]:
                    continue
                names, aliases = names_for(node, dataset)
                alias_count += sum(isinstance(a, str) and bool(norm(a)) for a in aliases)
                for name in names:
                    if isinstance(name, str) and norm(name):
                        index[norm(name)].add(node["node_id"])
                        covered.add(node["node_id"])
        sources[str(path)] = lines
    assert len(covered) == EXPECTED[dataset], (dataset, len(covered))
    metadata = {"dataset": dataset, "covered_nodes": len(covered), "unique_names": len(index),
                "conflicting_names": sum(len(ids) > 1 for ids in index.values()),
                "alias_entries": alias_count, "source_lines": sources,
                "mapping": "NFKC, casefold, nonalphanumeric to space, collapse whitespace; all exact matches"}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "names": {k: sorted(v) for k, v in sorted(index.items())}}, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*")
    for dataset in parser.parse_args().datasets or list(EXPECTED):
        if dataset not in EXPECTED:
            parser.error("Unknown dataset")
        print(json.dumps(build(dataset), ensure_ascii=False), flush=True)
