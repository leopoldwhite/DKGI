"""Score generated names by rank, expanding every exact-name collision."""

import json
from name_index import norm


def parse(prediction, probe):
    parsed = json.loads(prediction)
    key = "names" if probe == "A0-Name" else "node_ids"
    if not isinstance(parsed, dict) or set(parsed) != {key} or not isinstance(parsed[key], list):
        raise ValueError("Expected exactly one ranked answer list")
    if len(parsed[key]) > 20:
        raise ValueError("More than 20 answers")
    kind = str if probe == "A0-Name" else int
    if any(type(value) is not kind for value in parsed[key]):
        raise ValueError("Answer has the wrong type")
    return parsed[key]


def score(names, gold, index):
    gold = set(gold)
    if not gold:
        raise ValueError("Empty gold")
    matches = [set(index.get(norm(name), [])) for name in names]
    hits = [bool(ids & gold) for ids in matches]
    combined = set().union(*matches[:20]) if matches else set()
    return {"hit_at_1": float(any(hits[:1])), "hit_at_5": float(any(hits[:5])),
            "hit_at_20": float(any(hits[:20])), "recall_at_20": len(combined & gold) / len(gold),
            "mrr": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
            "returned_count": len(names), "unmatched_count": sum(not ids for ids in matches),
            "ambiguous_count": sum(len(ids) > 1 for ids in matches),
            "mapped_ids": [sorted(ids) for ids in matches]}
