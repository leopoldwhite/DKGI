"""Closed-book probe summary (results.json, summary.md), the contamination rule fixed before the run
(A0-ID H@1 >= 1%), and the H@1 of an agent configuration stratified by the probe, from optional per-query result
files (--dci)."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from name_index import norm

ROOT = Path(__file__).resolve().parent
COUNTS = {"amazon": 1642, "mag": 2665, "prime": 2801}
GROUPS = [(p, d) for p in ("A0-ID", "A0-Name") for d in COUNTS if p == "A0-ID" or d != "amazon"]


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return statistics.mean(values) if values else None


def pct(value):
    return "n/a" if value is None else f"{100 * value:.3f}%"


def h1(row):
    value = row.get("metrics", row).get("hit_at_1")
    if value not in (0, 1):
        if row.get("status") in {"failed", "parse_failed"}:
            return 0
        raise ValueError("Per-query result row lacks binary hit_at_1")
    return int(value)


def dci_rows(path):
    data = load(path)
    if isinstance(data, dict):
        data = data.get("per_query", data.get("results"))
    if not isinstance(data, list):
        raise ValueError("Result file must be a row list or contain per_query/results")
    mapped = {}
    for row in data:
        key = row["dataset"], str(row["query_id"])
        if key in mapped:
            raise ValueError("Duplicate query key: " + repr(key))
        mapped[key] = h1(row)
    return mapped


def stratify(records, dci, criterion):
    result = {}
    for dataset in COUNTS:
        probe = "A0-ID" if dataset == "amazon" else "A0-Name"
        cohort = [r for r in records if r["probe"] == probe and r["dataset"] == dataset and (dataset, r["query_id"]) in dci]
        table = {"a0_wrong_dci_wrong": 0, "a0_wrong_dci_correct": 0, "a0_correct_dci_wrong": 0, "a0_correct_dci_correct": 0}
        for row in cohort:
            a0 = bool(row.get("metrics", {}).get(criterion, 0))
            correct = dci[dataset, row["query_id"]]
            table[f"a0_{'correct' if a0 else 'wrong'}_dci_{'correct' if correct else 'wrong'}"] += 1
        known = table["a0_correct_dci_wrong"] + table["a0_correct_dci_correct"]
        unknown = table["a0_wrong_dci_wrong"] + table["a0_wrong_dci_correct"]
        result[dataset] = {"probe": probe, "criterion": criterion, "paired": len(cohort), "planned": COUNTS[dataset],
                           "known_n": known, "unknown_n": unknown,
                           "known_dci_hit_at_1": table["a0_correct_dci_correct"] / known if known else None,
                           "unknown_dci_hit_at_1": table["a0_wrong_dci_correct"] / unknown if unknown else None,
                           "overall_dci_hit_at_1": (table["a0_correct_dci_correct"] + table["a0_wrong_dci_correct"]) / len(cohort) if cohort else None,
                           "table": table}
    return result


def audit_hits(records, signal_datasets, dci_sources):
    hits = [r for r in records if r["probe"] == "A0-ID" and r["dataset"] in signal_datasets and r.get("metrics", {}).get("hit_at_1", 0)]
    audit = []
    for dataset in signal_datasets:
        train_gold = rows(ROOT.parent / "private" / dataset / "train.jsonl")
        counts = Counter(node_id for r in train_gold for node_id in set(r["answer_ids"]))
        comparison = [(split, row, set(norm(row["query"]).split())) for split in ("train", "val")
                      for row in rows(ROOT.parent / "public" / dataset / f"{split}.jsonl")]
        test = {r["query_id"]: r for r in rows(ROOT.parent / "public" / dataset / "test.jsonl")}
        gold = {r["query_id"]: r["answer_ids"] for r in rows(ROOT.parent / "private" / dataset / "test.jsonl")}
        for hit in (r for r in hits if r["dataset"] == dataset):
            query = test[hit["query_id"]]["query"]
            tokens = set(norm(query).split())
            similar = sorted(((len(tokens & other) / len(tokens | other) if tokens | other else 1, split, r) for split, r, other in comparison),
                             key=lambda item: (-item[0], item[1], int(item[2]["query_id"])))[:5]
            audit.append({"dataset": dataset, "query_id": hit["query_id"], "query": query,
                          "gold": gold[hit["query_id"]], "top_prediction": hit["parsed"][0],
                          "train_answer_occurrences": {str(n): counts[n] for n in gold[hit["query_id"]]},
                          "train_question_count": len(train_gold),
                          "nearest_train_val": [{"split": split, "query_id": r["query_id"], "query": r["query"],
                                                 "token_jaccard": similarity, "exact_normalized_duplicate": norm(query) == norm(r["query"])}
                                                for similarity, split, r in similar]})
    excluded = {(r["dataset"], r["query_id"]) for r in hits}
    robustness = {}
    for name, dci in dci_sources.items():
        robustness[name] = {dataset: {"excluded": sum(key in excluded for key in dci if key[0] == dataset),
                                      "retained": sum(key not in excluded for key in dci if key[0] == dataset),
                                      "original_hit_at_1": mean([v for key, v in dci.items() if key[0] == dataset]),
                                      "retained_hit_at_1": mean([v for key, v in dci.items() if key[0] == dataset and key not in excluded])}
                            for dataset in COUNTS}
    return {"hits": audit, "robustness": robustness,
            "audit_method": "Report raw training-answer frequencies and top five exact-normalized-token Jaccard neighbors exhaustively over train/val. No similarity threshold changes the prespecified contamination decision."}


def analyze(dci_paths):
    generated = datetime.now(timezone.utc).isoformat()
    state = load(ROOT / "state.json") if (ROOT / "state.json").exists() else {"status": "not_started"}
    raw_records = [load(p) for p in sorted((ROOT / "runs").glob("*/*/test/*/result.json"))]
    # Attempts ended by a stop signal (exit 143) without a scored result are not model failures; they are left out
    # until the queue resumes them.
    interrupted = [r for r in raw_records if (r.get("returncode") == 143 and r.get("status") != "scored")
                   or r.get("status") == "interrupted"]
    interrupted_keys = {(r["probe"], r["dataset"], r["query_id"]) for r in interrupted}
    records = [r for r in raw_records if r.get("terminal") and (r["probe"], r["dataset"], r["query_id"]) not in interrupted_keys]
    complete = len(records) == 12574
    summaries = []
    for probe, dataset in GROUPS:
        group = [r for r in records if r["probe"] == probe and r["dataset"] == dataset]
        metrics = [r.get("metrics", {}) for r in group]
        returned = sum(m.get("returned_count", 0) for m in metrics)
        summary = {"probe": probe, "dataset": dataset, "planned": COUNTS[dataset], "completed": len(group),
                   "failed": sum(r["status"] != "scored" for r in group),
                   "hit_at_1": mean([m.get("hit_at_1", 0) for m in metrics]),
                   "hit_at_5": mean([m.get("hit_at_5", 0) for m in metrics]),
                   "recall_at_20": mean([m.get("recall_at_20", 0) for m in metrics]),
                   "mrr": mean([m.get("mrr", m.get("mrr_at_100", 0)) for m in metrics]),
                   "returned_count": returned, "known_cost_usd": sum(r.get("cost_usd", 0) for r in group),
                   "median_wall_seconds": mean([]) if not group else statistics.median(r.get("total_wall_seconds", 0) for r in group)}
        for label, numerator in (("unmatched_rate", "unmatched_count"), ("ambiguity_rate", "ambiguous_count"), ("legal_candidate_rate", "legal_candidate_count")):
            summary[label] = sum(m.get(numerator, 0) for m in metrics) / returned if returned else None
        summary["failure_rate"] = summary["failed"] / len(group) if group else None
        summaries.append(summary)
    payload = {"generated_at": generated, "complete": complete, "planned": 12574, "completed": len(records),
               "failure_policy": "Terminal failures score zero; pending jobs excluded from interim means; no final decision until complete.",
               "stopped_not_counted": len(interrupted), "summary": summaries, "results": records}
    save(ROOT / "results.json", payload)
    table = ["# Probe Results", "", f"Status: {'complete' if complete else 'INCOMPLETE'}; {len(records)}/12574 terminal jobs. Generated {generated}.", "",
             "Failures count as zero. Interim metrics use completed jobs only and are not final conclusions.", "",
             "| Probe | Dataset | Completed / Planned | Failed | H@1 | H@5 | R@20 | MRR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        table.append(f"| {s['probe']} | {s['dataset']} | {s['completed']} / {s['planned']} | {s['failed']} | {pct(s['hit_at_1'])} | {pct(s['hit_at_5'])} | {pct(s['recall_at_20'])} | {pct(s['mrr'])} |")
    table += ["", "| Probe | Dataset | Unmatched names | Ambiguous names | Legal candidate IDs |", "|---|---|---:|---:|---:|"]
    for s in summaries:
        table.append(f"| {s['probe']} | {s['dataset']} | {pct(s['unmatched_rate']) if s['probe']=='A0-Name' else 'n/a'} | {pct(s['ambiguity_rate']) if s['probe']=='A0-Name' else 'n/a'} | {pct(s['legal_candidate_rate']) if s['probe']=='A0-ID' else 'n/a'} |")
    attempts = []
    for directory in sorted((ROOT / "runs").glob("*/*/test/*/attempt_*")):
        path = directory / "result.json"
        if path.exists():
            attempts.append(load(path))
    cost = sum(r.get("attempt_cost_usd") or r.get("observed_cost_usd_lower_bound") or 0 for r in attempts)
    input_tokens = sum(r.get("input_tokens") or 0 for r in attempts)
    output_tokens = sum(r.get("output_tokens") or 0 for r in attempts)
    incomplete_usage = sum(not r.get("usage_complete", False) for r in attempts)
    payload.update(known_cost_usd=cost, input_tokens=input_tokens, output_tokens=output_tokens,
                   saved_attempts=len(attempts), incomplete_usage_attempts=incomplete_usage)
    save(ROOT / "results.json", payload)
    table += ["", f"Observed official-price equivalent: ${cost:.6f}; recorded input tokens {input_tokens:,}, output tokens {output_tokens:,}.",
              f"Attempts with incomplete usage: {incomplete_usage}. Running attempts and unreported failed-response tokens are not included; this is a lower bound when present.",
              f"Prices per million tokens (list price): input $0.10, cached read $0.01, cache write $0.125, output $0.50."]
    if interrupted:
        table += ["", f"{len(interrupted)} stopped jobs are excluded from interim metrics until they are resumed."]
    table += ["", "| Index | Covered nodes | Unique names | Conflicting names | Alias entries |", "|---|---:|---:|---:|---:|"]
    for dataset in ("mag", "prime"):
        metadata = load(ROOT / "index" / f"{dataset}.json")["metadata"]
        table.append(f"| {dataset} | {metadata['covered_nodes']} | {metadata['unique_names']} | {metadata['conflicting_names']} | {metadata['alias_entries']} |")
    subset = {}
    for dataset in COUNTS:
        ids = {r["query_id"] for r in rows(ROOT.parent / "public" / dataset / "test-0.1.jsonl")}
        for probe, d in GROUPS:
            if d == dataset:
                chosen = [r for r in records if r["probe"] == probe and r["dataset"] == dataset and r["query_id"] in ids]
                subset[f"{probe}:{dataset}"] = {"planned": len(ids), "completed": len(chosen),
                                                **{metric: mean([r.get("metrics", {}).get(metric, 0) for r in chosen]) for metric in ("hit_at_1", "hit_at_5", "recall_at_20")}}
    save(ROOT / "test_0.1.json", subset)
    (ROOT / "summary.md").write_text("\n".join(table) + "\n")
    dci_sources = {str(p): dci_rows(p) for p in dci_paths}
    stratified = {name: {criterion: stratify(records, dci, criterion) for criterion in ("hit_at_1", "hit_at_20")} for name, dci in dci_sources.items()}
    save(ROOT / "stratified.json", {"generated_at": generated, "complete": complete, "sources": stratified})
    lines = ["# Agent H@1 by closed-book probe stratum", "", f"Probe status: {'complete' if complete else 'INCOMPLETE'}. Only available paired rows enter interim tables.", "",
             "MAG and Prime use A0-Name; Amazon uses A0-ID. Agent H@1 in the unknown group is the prespecified corpus-reading gain.",
             "The 2x2 cells are raw question counts. Empty groups have undefined rates (n/a)."]
    for name, criteria in stratified.items():
        lines += ["", f"## {name}"]
        for criterion, datasets in criteria.items():
            lines += ["", f"Known criterion: {criterion}.", "", "| Dataset | Paired | Known n | Unknown n | Known agent H@1 | Unknown agent H@1 | All agent H@1 | A0 wrong/agent wrong | A0 wrong/agent correct | A0 correct/agent wrong | A0 correct/agent correct |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for dataset, s in datasets.items():
                t = s["table"]
                lines.append(f"| {dataset} | {s['paired']} | {s['known_n']} | {s['unknown_n']} | {pct(s['known_dci_hit_at_1'])} | {pct(s['unknown_dci_hit_at_1'])} | {pct(s['overall_dci_hit_at_1'])} | {t['a0_wrong_dci_wrong']} | {t['a0_wrong_dci_correct']} | {t['a0_correct_dci_wrong']} | {t['a0_correct_dci_correct']} |")
    (ROOT / "stratified.md").write_text("\n".join(lines) + "\n")
    signals = [s["dataset"] for s in summaries if s["probe"] == "A0-ID" and (s["hit_at_1"] or 0) >= .01] if complete else []
    decision = ["# Contamination Decision", "", f"Status: {'complete' if complete else 'INCOMPLETE'} ({len(records)}/12574 terminal jobs).", "",
                "Prespecified rule: A0-ID H@1 >= 1% in any dataset signals contamination. Prompts, mappings and this threshold were not tuned on test results."]
    for s in summaries:
        if s["probe"] == "A0-ID":
            decision.append(f"- {s['dataset']}: A0-ID H@1 {pct(s['hit_at_1'])}; {s['completed']}/{s['planned']} complete.")
    if not complete:
        decision += ["", "No contamination conclusion is available until all jobs reach final states."]
    elif signals:
        audit = audit_hits(records, signals, dci_sources)
        save(ROOT / "contamination_hits.json", audit)
        decision += ["", "Contamination signal detected in: " + ", ".join(signals) + ".",
                     "All H@1 hit questions, raw training-answer frequencies, nearest train/val questions and the agent H@1 with those questions excluded are in contamination_hits.json."]
    else:
        numbers = ", ".join(f"{s['dataset']} {pct(s['hit_at_1'])}" for s in summaries if s["probe"] == "A0-ID")
        decision += ["", f"No contamination signal under the 1% A0-ID H@1 threshold ({numbers})."]
    if complete and any((s["failure_rate"] or 0) > .02 for s in summaries):
        decision += ["", "Warning: at least one group exceeds 2% terminal failures; review is required."]
    (ROOT / "DECISION.md").write_text("\n".join(decision) + "\n")
    print(json.dumps({"complete": complete, "completed": len(records), "planned": 12574, "known_cost_usd": cost, "signals": signals}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dci", type=Path, nargs="*", default=[],
                        help="optional JSON files with per-query result rows of an agent configuration "
                             "(dataset, query_id, hit_at_1)")
    analyze(parser.parse_args().dci)
