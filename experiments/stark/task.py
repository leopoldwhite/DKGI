"""STaRK task: the base prompt, scoring of a ranked ID list, and retry classification of failed attempts."""

import importlib.util
import json
from pathlib import Path
import re

import prepare

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("agent_base", ROOT.parent / "agent_base/base.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ARM = {"harness": "codex", "context_level": "native"}


def score(prediction, labels, candidates):
    parsed = json.loads(prediction)
    if not isinstance(parsed, dict) or set(parsed) != {"node_ids"}:
        raise ValueError("Expected exactly {node_ids: [...]} as JSON")
    ranked = parsed["node_ids"]
    if not isinstance(ranked, list) or len(ranked) > 100:
        raise ValueError("node_ids must be a list of at most 100 entries")
    gold = set(labels)
    if not gold or any(type(i) is not int or i < 0 for i in gold):
        raise ValueError("Gold must contain nonnegative integer node IDs")
    seen, hits, invalid = set(), [], []
    for rank, node_id in enumerate(ranked, 1):
        if type(node_id) is not int or node_id not in candidates or node_id in seen:
            invalid.append(rank)
            hits.append(False)
        else:
            seen.add(node_id)
            hits.append(node_id in gold)
    result = {"mrr_at_100": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
              "returned_count": len(ranked), "invalid_or_duplicate_ranks": invalid,
              "candidate_coverage": len(gold & candidates) / len(gold),
              "unreachable_gold_count": len(gold - candidates)}
    for k in (1, 3, 5, 10, 20, 50, 100):
        result[f"hit_at_{k}"] = float(any(hits[:k]))
        result[f"recall_at_{k}"] = sum(hits[:k]) / len(gold)
    return result


def prompt_for(row, attempt):
    return (
        "Answer the retrieval query using only the supplied local STaRK knowledge base. "
        "Use the textual attributes and graph relations as evidence. Return candidate node IDs "
        "ranked by relevance, most relevant first. Use the benchmark node_id field, not external "
        "IDs within attributes. Only nodes marked is_candidate=true are valid answers; other "
        "node types may be used as evidence. Return at most 100 unique integer IDs observed in "
        "the corpus. Do not invent or pad IDs; return an empty list if none are supported.\n\n"
        'Return only JSON with exactly this structure: {"node_ids": [123, 456]}. '
        "Do not include Markdown, explanations, or citations outside the JSON.\n\n"
        f"Query: {row['query']}\n\nCorpus directory: {row['corpus_dir']}\n"
        "Read DATA_FORMAT.txt and schema.json for the file format, entity types, relation "
        "types, and graph interpretation. Nodes and edges are complete local text files. "
        "Use shell tools, rg, or Python 3 standard-library modules to search and read them. "
        f"The corpus is read-only. Writable scratch directory: {attempt / 'tmp'}\n"
    )


def retry_reason(result, attempt):
    if result.get("status") == "parse_failed":
        return "invalid_output" if result.get("error_type") in ("ValueError", "JSONDecodeError") else None
    if result.get("error_phase") != "agent":
        return None
    if result.get("error_type") == "TimeoutExpired":
        return "timeout"
    messages = [result.get("error", "")]
    errors = attempt / "stderr.txt"
    if errors.exists():
        with errors.open("rb") as source:
            source.seek(max(0, errors.stat().st_size - 16384))
            messages.append(source.read().decode("utf-8", errors="replace"))
    events = attempt / "stdout.jsonl"
    if events.exists():
        with events.open() as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") in ("error", "turn.failed"):
                    messages.append(json.dumps(event))
    # The final error takes precedence over earlier errors during SDK retries.
    for message in reversed(messages):
        message = message.lower()
        if re.search(r"\b(400|401|403|404|405|422)\b|invalid_api_key|unauthori[sz]ed|"
                     r"insufficient[ _]quota|quota[ _](exceeded|exhausted)|model_not_found|unsupported|unknown model|invalid.request|"
                     r"unexpected argument|unrecognized argument", message):
            return None
        # An upstream authentication error relayed by an intermediate proxy is transient, not a client 401.
        if re.search(r"\b5\d\d\b", message):
            return "transient_agent_error"
        if "authentication" in message:
            return None
        if re.search(r"\b(408|425|429)\b|rate.?limit|overload|temporar|timed? ?out|timeout|"
                     r"connection|connect error|stream|transport|socket|\beof\b|no final answer|"
                     r"did not complete successfully", message):
            return "transient_agent_error"
    return None
