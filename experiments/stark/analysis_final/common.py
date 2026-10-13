"""Shared loaders, per-query feature extraction and writers for the analyses.

Read-only with respect to every experiment directory. Nothing here calls a model or
executes a recorded command. Outputs go to analysis_final/out/ only.

Result layouts handled:
  * run trees   <root>/<ds>/<split-dir>/<qid>/result.json  (Full DKGI, tool variants, Codex, validation rounds)
  * raw jsonl   one row per final result (prompt-selection round on val)

Model-visible tool output:
  * Pi      attempt/artifacts/events.jsonl, provider_request_context payload.input function_call_output
            (last request; union over all requests when the run compacted)
  * Codex   attempt/model_tool_outputs.jsonl
  * Codex without a recorded model input: stdout.jsonl aggregated_output is the raw command output,
            a superset of what the model saw -> visibility = "upper_bound" (seen = upper bound,
            retrieval-miss = lower bound).
"""

import ast
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # model-written Python parsed by the AST classifiers

HERE = Path(__file__).resolve().parent
STARK = HERE.parent
OUT = HERE / "out"
CACHE = OUT / "cache"
sys.path.insert(0, str(STARK))

import analyze_behavior as base  # noqa: E402  (existing shared shell/AST parser)
import align_intent_distribution as align  # noqa: E402  (aligned nine-category classifier)
import analyze_graph_behavior as graph  # noqa: E402  (graph operations inside Python)

DATASETS = ("amazon", "mag", "prime")
DS_LABEL = {"amazon": "Amazon", "mag": "MAG", "prime": "Prime", "all": "All"}
INTENTS = align.ORDER
INTENT_LABEL = align.TITLES
GRAPH_LABELS = tuple(graph.LABELS)
ERROR_CATS = ("correct", "ranking", "seen_not_selected", "retrieval_miss", "empty_format", "unknown_visibility")
ERROR_LABEL = {"correct": "Correct (H@1)", "ranking": "Ranking (gold in list, not first)",
               "seen_not_selected": "Seen, not selected", "retrieval_miss": "Retrieval miss (never seen)",
               "empty_format": "Empty / format / failed run", "unknown_visibility": "Wrong, visibility unknown"}

L3_MARK = re.compile(r"\[\.\.\.truncated, ([0-9]+) chars omitted\]")
H6_MARK = re.compile(r"\[\.\.\. harness output limit: .*?; ([0-9]+) bytes omitted in between")
PI_FULL_MARK = re.compile(r"Full output: ")
CODEX_MARK = re.compile(r"…([0-9]+) tokens truncated…")
FOUND = re.compile(r"found ([0-9]+) results")
CODEX_EST_LIMIT = 40_000  # Codex without recorded model input: Codex truncates at ~10k approx tokens (~40 KB), head+tail

# Bump when the per-query feature logic changes; cached rows with another version are re-extracted.
FEATURE_VERSION = "2026-09-29-d"

# Figure/table groupings: main-text figures use the 710 subset with the tool variants.
MAIN_710 = ("A4", "H", "A1", "A2", "A3", "A5", "A6")
FULL_SET = ("pi_final_full",)

SEL_710 = STARK / "trials/selection.json"
SEL_FULL = STARK / "pi_full_test/selection.json"


# --------------------------------------------------------------------------- configurations

def _cfg(name, label, tier, harness, kind, source, split, **extra):
    return {"name": name, "label": label, "tier": tier, "harness": harness, "kind": kind,
            "source": source, "split": split, **extra}


def configurations():
    """Registry of every configuration the analyses know about. Missing data -> empty coverage."""
    PI = STARK / "pi_full_test/runs/pi_final"
    ABL = STARK / "trials"
    H = STARK / "trials/codex_h/codex_final"
    T11 = STARK / "prompt_val/results_raw.jsonl"
    R1 = STARK / "prompt_val_r1"
    cfgs = [
        _cfg("pi_final_full", "Full DKGI (Pi, full test)", "paper", "pi", "file", ("tree", PI, "test"), "test",
             expected=SEL_FULL, expected_split="test"),
        _cfg("pi_final_human", "Full DKGI (Pi, human-generated set)", "paper", "pi", "file", ("tree", PI, "human_generated_eval"),
             "human_generated_eval", expected=SEL_FULL, expected_split="human_generated_eval"),
        _cfg("A4", "Full DKGI (test subset)", "paper", "pi", "file", ("tree", PI, "test"), "test", subset=SEL_710,
             expected=SEL_710, cache_as="pi_final_full"),
        _cfg("H", "Full DKGI (Codex, test subset)", "paper", "codex", "file", ("tree", H, "test"), "test", subset=SEL_710,
             expected=SEL_710),
    ]
    for arm, label, kind in (("A1", "Retrieval only", "tool"), ("A2", "Graph tools", "tool"),
                             ("A3", "Shell only", "file"), ("A5", "Graph tools + code", "tool_code"),
                             ("A6", "No edges", "file")):
        cfgs.append(_cfg(arm, label, "paper", "pi", kind, ("tree", ABL / arm, "test"), "test", expected=SEL_710))
    # Stronger backbone (tier "extra": used by final_numbers.py; other reports list it in coverage tables and CSVs only).
    cfgs += [
        _cfg("astra_pi_full", "Full DKGI (Pi) + GPT-6-Astra (full test)", "extra", "pi", "file",
             ("tree", STARK / "astra_runs/runs/pi_astra", "test"), "test", expected=SEL_FULL, expected_split="test"),
        _cfg("astra_pi_710", "Full DKGI (Pi) + GPT-6-Astra (test subset)", "extra", "pi", "file",
             ("tree", STARK / "astra_runs/runs/pi_astra", "test"), "test", subset=SEL_710, expected=SEL_710),
        _cfg("astra_codex_710", "Full DKGI (Codex) + GPT-6-Astra (test subset)", "extra", "codex", "file",
             ("tree", STARK / "astra_runs/runs/codex_astra", "test"), "test", subset=SEL_710, expected=SEL_710),
        # Codex + GPT-6-Luna on the full test (preset codex_luna_full; the 710 of "H" are part of the same tree).
        _cfg("codex_luna_full", "Full DKGI (Codex) + GPT-6-Luna (full test)", "extra", "codex", "file", ("tree", H, "test"), "test",
             expected=SEL_FULL, expected_split="test"),
        # One program: one python3 start per query, Shell only allowlist otherwise.
        _cfg("X1", "One program (test subset)", "extra", "pi", "file", ("tree", ABL / "O1", "test"), "test",
             expected=SEL_710),
    ]
    # Validation configurations (val split): prompt-selection round (t11_*) and confirmatory round (r1_*).
    for group in ("codex_p0", "codex_p1", "codex_p2", "pi_p0", "pi_p1", "pi_p2"):
        h = "pi" if group.startswith("pi") else "codex"
        cfgs.append(_cfg(f"t11_{group}", f"Prompt selection {group} (val)", "dev", h, "file",
                         ("raw", T11, group), "val"))
    for group in ("codex_p0", "codex_p1r1", "pi_p0", "pi_p1r1"):
        h = "pi" if group.startswith("pi") else "codex"
        cfgs.append(_cfg(f"r1_{group}", f"Confirmatory {group} (val)", "dev", h, "file",
                         ("tree", R1 / "runs" / group, "val"), "val"))
    return cfgs


def config_by_name():
    return {c["name"]: c for c in configurations()}


# --------------------------------------------------------------------------- basic readers

def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_obj(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                return json.loads(value)
            except ValueError:
                return {}
    return value or {}


def load_selection(path, split=None):
    if not path or not Path(path).exists():
        return None
    rows = read_json(path)
    rows = rows["questions"] if isinstance(rows, dict) else rows
    return {(r["dataset"], str(r["query_id"])) for r in rows if split is None or r.get("split") in (None, split)}


_GOLD = {}


def gold(dataset, split):
    split = "test" if split == "test-0.1" else split
    key = (dataset, split)
    if key not in _GOLD:
        path = STARK / "private" / dataset / f"{split}.jsonl"
        _GOLD[key] = {str(r["query_id"]): [int(x) for x in r["answer_ids"]] for r in read_jsonl(path)} if path.exists() else {}
    return _GOLD[key]


def iter_results(cfg):
    """Yield (result_dict, source_path) for the final result of each query in a configuration."""
    src = cfg["source"]
    subset = load_selection(cfg.get("subset")) if cfg.get("subset") else None
    if src[0] == "eval":
        if not Path(src[1]).exists():
            return
        for row in read_json(src[1])["per_query"]:
            if subset is not None and (row["dataset"], str(row["query_id"])) not in subset:
                continue
            path = Path(row["result_path"])
            if path.exists():
                yield read_json(path), str(path)
    elif src[0] == "raw":
        if not Path(src[1]).exists():
            return
        for row in read_jsonl(src[1]):
            if row.get("group") == src[2]:
                yield row, str(row.get("attempt_path") or row.get("attempt"))
    elif src[0] == "tree":
        root, splitdir = Path(src[1]), src[2]
        for ds in DATASETS:
            d = root / ds / splitdir
            if not d.is_dir():
                continue
            for qdir in sorted(d.iterdir(), key=lambda p: p.name):
                path = qdir / "result.json"
                if not path.exists():
                    continue
                if subset is not None and (ds, qdir.name) not in subset:
                    continue
                try:
                    yield read_json(path), str(path)
                except ValueError:
                    continue  # partially written file of a running query


# --------------------------------------------------------------------------- model-visible outputs

def _text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(b.get("text", "") for b in output if isinstance(b, dict))
    return "" if output is None else str(output)


def pi_calls(attempt, union):
    """Tool calls and model-visible outputs from Pi provider_request_context events."""
    path = Path(attempt) / "artifacts/events.jsonl"
    if not path.exists():
        return None
    raws = []
    with open(path, "rb") as f:
        for raw in f:
            if b'"provider_request_context"' in raw[:200]:
                if union:
                    raws.append(raw)
                else:
                    raws = [raw]
    if not raws:
        return None
    calls, outs = {}, {}
    for raw in raws:
        items = json.loads(raw).get("payload", {}).get("input", [])
        for it in items:
            t = it.get("type")
            if t == "function_call" and it["call_id"] not in calls:
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls[it["call_id"]] = {"name": it.get("name"), "args": args}
            elif t == "function_call_output":
                outs.setdefault(it["call_id"], _text(it.get("output")))
    return [{"call_id": k, **v, "output": outs.get(k), "visible": k in outs} for k, v in calls.items()]


def codex_commands(attempt):
    """Shell commands from Codex stdout.jsonl (deduplicated by item id; tolerant of partial traces)."""
    path = Path(attempt) / "stdout.jsonl"
    cmds = {}
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            if '"command_execution"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            it = e.get("item", {})
            if it.get("type") != "command_execution":
                continue
            c = cmds.setdefault(it["id"], {"command": it.get("command", ""), "completed": False, "output": None,
                                           "exit_code": None})
            if e.get("type") == "item.completed":
                c.update(completed=True, output=it.get("aggregated_output", ""), exit_code=it.get("exit_code"))
    return list(cmds.values())


def codex_visible(attempt):
    path = Path(attempt) / "model_tool_outputs.jsonl"
    if not path.exists():
        return None
    outs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            outs.setdefault(r["call_id"], _text(r.get("output")))
    return outs


# --------------------------------------------------------------------------- per-query features

def id_regex(ids):
    ids = sorted({int(i) for i in ids})
    if not ids:
        return None
    return re.compile(r"(?<![0-9])(?:%s)(?![0-9])" % "|".join(map(str, ids)))


def record_regex(ids):
    ids = sorted({int(i) for i in ids})
    if not ids:
        return None
    alt = "|".join(map(str, ids))
    # JSONL record key or tool-output record header "[123] name"
    return re.compile(r"(?:node_id\"?\s*[:=]\s*(?:%s)(?![0-9]))|(?:\[(?:%s)\])" % (alt, alt))


def classify_command(command):
    """Aligned nine-category intent + nodes/edges read + graph operation labels for one shell command."""
    got = align.classify(command)
    normalized = got["normalized_command"]
    wrapped = "/usr/bin/bash -lc " + shlex.quote(normalized)
    ops, py, glabels = set(), False, []
    try:
        parsed = base.classify(wrapped)
        ops = set(parsed["operations"])
        py = "python" in parsed["tools"]
    except Exception:  # noqa: BLE001  (unparseable shell: keep category only)
        pass
    if py:
        try:
            glabels = graph.classify_command(wrapped)["labels"]
        except Exception:  # noqa: BLE001
            glabels = []
    return {"intent": got["category"], "python": py or got["category"] == "python",
            "nodes": "nodes" in ops, "edges": "edges" in ops, "graph": glabels}


PRICE = {"input": 0.10, "cached": 0.01, "output": 0.50}  # USD per million tokens, gpt-6-luna list price


def price_equivalent(result):
    """Official-price equivalent from recorded usage (lower bound; used only when no cost was recorded)."""
    u = result.get("total_usage_raw") or result.get("usage_raw") or {}
    inp, cached, out = u.get("input_tokens"), u.get("cached_input_tokens") or 0, u.get("output_tokens")
    if inp is None or out is None:
        return None
    return ((inp - cached) * PRICE["input"] + cached * PRICE["cached"] + out * PRICE["output"]) / 1e6


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def features(cfg, result, src):
    """One flat record per query. Never raises: extraction problems are stored in 'extract_error'."""
    ds = result.get("dataset")
    qid = str(result.get("query_id"))
    split = cfg["split"]
    status = result.get("status")
    terminal = bool(result.get("terminal", status in ("scored", "failed")))
    m = parse_obj(result.get("metrics"))
    try:
        pred = [int(x) for x in json.loads(result.get("prediction") or "{}").get("node_ids", [])]
        parse_ok = True
    except (ValueError, TypeError, AttributeError):
        pred, parse_ok = [], False
    g = gold(ds, split).get(qid, [])
    gs = set(g)
    pred = list(dict.fromkeys(pred))  # duplicates keep their first rank
    rank = next((i + 1 for i, p in enumerate(pred) if p in gs), None)
    scored = status == "scored"
    # Metrics are recomputed from the prediction and the split's own gold file (private/<ds>/<split>.jsonl).
    # The recorded hit_at_1 is kept for comparison; disagreements are counted in coverage().
    hit1 = float(scored and rank == 1)
    rec = {"config": cfg["name"], "tier": cfg["tier"], "dataset": ds, "split": split, "query_id": qid,
           "status": status, "terminal": terminal, "scored": scored, "parse_ok": parse_ok,
           "hit1": hit1, "hit5": float(scored and rank is not None and rank <= 5),
           "r20": (len(gs & set(pred[:20])) / len(gs) if scored and gs else 0.0),
           "mrr": (1.0 / rank if scored and rank is not None and rank <= 100 else 0.0),
           "recorded_hit1": float(m.get("hit_at_1", 0.0)) if scored else 0.0,
           "returned": len(pred), "gold_n": len(g), "gold_rank": rank, "top1": pred[0] if pred else None,
           "input_tokens": _num(result.get("total_input_tokens", result.get("input_tokens"))),
           "cached_tokens": _num(result.get("cached_input_tokens")),
           "output_tokens": _num(result.get("total_output_tokens", result.get("output_tokens"))),
           "wall_seconds": _num(result.get("total_wall_seconds", result.get("total_agent_seconds", result.get("wall_time_seconds")))),
           "cost_usd": _num(result.get("cost_usd", result.get("attempt_cost_usd"))) if result.get("cost_usd", result.get("attempt_cost_usd")) is not None else price_equivalent(result),
           "tool_calls_result": _num(result.get("tool_calls")), "compactions": _num(result.get("compaction_starts")) or 0,
           "src": src, "finished_at": result.get("finished_at"), "extract_error": None,
           "feature_version": FEATURE_VERSION}
    attempt = result.get("attempt_path") or result.get("attempt")
    try:
        rec.update(_trajectory(cfg, attempt, g, pred, rec))
    except Exception as e:  # noqa: BLE001  (one bad trace never stops the analysis)
        rec["extract_error"] = repr(e)[:300]
        rec.setdefault("visibility", "missing")
    return rec


def _trajectory(cfg, attempt, g, pred, rec):
    out = {"visibility": "missing", "n_outputs": 0, "visible_chars": 0, "gold_seen_id": None, "gold_seen_record": None,
           "top1_seen": None, "gold_first_call": None, "gold_seen_sim": None, "gold_seen_sim_id": None,
           "gold_first_call_id": None, "trunc_l3_by_tool": {}, "trunc_l3": 0, "trunc_h6": 0, "trunc_pi_full": 0,
           "trunc_codex": 0, "trunc_est_large": 0, "omitted_units": 0, "bash_calls": 0, "python_calls": 0,
           "read_calls": 0, "other_tool_calls": 0, "calls": 0, "nodes_calls": 0, "edges_calls": 0,
           "py_edges_calls": 0, "py_nodes_calls": 0, "intents": {}, "graph_ops": {}, "tools_by_name": {},
           "service_calls": 0, "neighbors_calls": 0, "neighbors_capped": 0}
    if not attempt or not Path(attempt).exists():
        return out
    grx, rrx = id_regex(g), record_regex(g)
    trx = id_regex(pred[:1])
    commands, texts, names = [], [], []
    h = cfg["harness"]
    if h == "pi":
        calls = pi_calls(attempt, union=rec["compactions"] > 0)
        if calls is None:
            return out
        out["visibility"] = "recorded"
        for c in calls:
            name = c["name"]
            out["tools_by_name"][name] = out["tools_by_name"].get(name, 0) + 1
            if name == "bash":
                commands.append(c["args"].get("command", ""))
            elif name == "read":
                out["read_calls"] += 1
            else:
                out["other_tool_calls"] += 1
            texts.append(c["output"] if c["visible"] else None)
            names.append(name)
        out["calls"] = len(calls)
    elif h in ("codex", "codex_raw"):
        cmds = codex_commands(attempt) or []
        commands = [c["command"] for c in cmds]
        out["calls"] = len(cmds)
        vis = codex_visible(attempt) if h == "codex" else None
        if vis is not None:
            out["visibility"] = "recorded"
            texts = list(vis.values())
            names = ["code_cell"] * len(texts)
            out["model_calls"] = len(vis)
        elif cmds:
            out["visibility"] = "upper_bound"
            texts = [c["output"] for c in cmds]
            names = ["shell"] * len(texts)
            out["trunc_est_large"] = sum(1 for t in texts if t and len(t.encode()) > CODEX_EST_LIMIT)
            # Sensitivity only: approximate Codex's head+tail truncation (~half of the limit at each end).
            half = CODEX_EST_LIMIT // 2
            sim = [t if len(t) <= CODEX_EST_LIMIT else t[:half] + t[-half:] for t in texts if t]
            out["gold_seen_sim"] = bool(rrx and any(rrx.search(t) for t in sim))
            out["gold_seen_sim_id"] = bool(grx and any(grx.search(t) for t in sim))
    first, first_rec = None, None
    for i, t in enumerate(texts):
        if t is None:
            continue
        out["n_outputs"] += 1
        out["visible_chars"] += len(t)
        for rx, key in ((L3_MARK, "trunc_l3"), (H6_MARK, "trunc_h6"), (CODEX_MARK, "trunc_codex")):
            found = rx.findall(t)
            if found:
                out[key] += 1
                if key == "trunc_l3":
                    tool = names[i] if i < len(names) else "unknown"
                    out["trunc_l3_by_tool"][tool] = out["trunc_l3_by_tool"].get(tool, 0) + 1
                out["omitted_units"] += sum(int(x) for x in found) * (4 if key == "trunc_codex" else 1)
        if PI_FULL_MARK.search(t):
            out["trunc_pi_full"] += 1
        if grx is not None and first is None and grx.search(t):
            first = i
        if rrx is not None and first_rec is None and rrx.search(t):
            first_rec = i
    if out["visibility"] != "missing":
        joined = [t for t in texts if t]
        out["gold_seen_id"] = first is not None
        out["gold_first_call_id"] = first
        out["gold_seen_record"] = first_rec is not None
        out["gold_first_call"] = first_rec
        out["top1_seen"] = bool(trx and any(trx.search(t) for t in joined)) if pred else None
    intents, gops = Counter(), Counter()
    for cmd in commands:
        c = classify_command(cmd)
        intents[c["intent"]] += 1
        out["bash_calls"] += 1
        out["python_calls"] += c["python"]
        out["nodes_calls"] += c["nodes"]
        out["edges_calls"] += c["edges"]
        if c["python"]:
            out["py_edges_calls"] += "edge_read" in c["graph"] or c["edges"]
            out["py_nodes_calls"] += c["nodes"]
        gops.update(c["graph"])
    out["intents"] = dict(intents)
    out["graph_ops"] = dict(gops)
    tr = Path(attempt) / "tool_requests.jsonl"
    if tr.exists():
        with open(tr) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("phase") != "agent":
                    continue
                out["service_calls"] += 1
                req = r.get("request", {})
                if req.get("name") == "neighbors":
                    out["neighbors_calls"] += 1
                    k = req.get("arguments", {}).get("k", 20)
                    mm = FOUND.search((r.get("response") or {}).get("text", "") or "")
                    if mm and k and int(mm[1]) > int(k):
                        out["neighbors_capped"] += 1
    return out


def _work(args):
    cfg, result, src = args
    return features(cfg, result, src)


def load_features(cfg, use_cache=True, processes=None):
    """Per-query records for one configuration.

    Cache: out/cache/<source>.jsonl, reused when a row's source path, status and finish time are unchanged
    (plain value comparison). Configurations that share a source (full test and its 710 subset) share the cache.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / f"{cfg.get('cache_as', cfg['name'])}.jsonl"
    cached = {}
    if use_cache and cache_path.exists():
        for r in read_jsonl(cache_path):
            cached[r["src"]] = r
    todo, rows = [], []
    for result, src in iter_results(cfg):
        c = cached.get(src)
        if (c is not None and c.get("status") == result.get("status") and c.get("finished_at") == result.get("finished_at")
                and c.get("feature_version") == FEATURE_VERSION):
            rows.append(c)
        else:
            todo.append((cfg, result, src))
    if todo:
        if len(todo) > 20:
            from multiprocessing import Pool
            with Pool(processes or min(48, os.cpu_count() or 4)) as pool:
                fresh = list(pool.imap_unordered(_work, todo, chunksize=4))
        else:
            fresh = [_work(t) for t in todo]
        rows.extend(fresh)
        cached.update({r["src"]: r for r in fresh})
        with open(cache_path, "w") as f:
            for r in cached.values():
                f.write(json.dumps(r) + "\n")
    rows = [{**r, "config": cfg["name"], "tier": cfg["tier"]} for r in rows]
    rows.sort(key=lambda r: (r["dataset"], int(r["query_id"]) if r["query_id"].isdigit() else r["query_id"]))
    return rows


def coverage(cfg, rows):
    exp = load_selection(cfg.get("expected"), cfg.get("expected_split")) if cfg.get("expected") else None
    out = {}
    for ds in DATASETS:
        sel = [r for r in rows if r["dataset"] == ds]
        expected = len([k for k in exp if k[0] == ds]) if exp is not None else None
        out[ds] = {"expected": expected, "present": len(sel), "terminal": sum(r["terminal"] for r in sel),
                   "scored": sum(r["scored"] for r in sel),
                   "failed": sum(r["terminal"] and not r["scored"] for r in sel),
                   "pending": sum(not r["terminal"] for r in sel),
                   "visibility": dict(Counter(r.get("visibility") for r in sel if r["terminal"])),
                   "extract_errors": sum(bool(r.get("extract_error")) for r in sel),
                   "metric_mismatch": sum(r["scored"] and r["hit1"] != r.get("recorded_hit1", r["hit1"]) for r in sel)}
    return out


# --------------------------------------------------------------------------- error categories

def error_category(r, seen_key="gold_seen_record"):
    """Error category of one terminal query (first matching rule wins)."""
    if r["hit1"] >= 1.0:
        return "correct"
    if not r["scored"] or not r["parse_ok"] or r["returned"] == 0:
        return "empty_format"
    if r["gold_rank"] is not None:
        return "ranking"
    seen = r.get(seen_key)
    if seen is None:
        return "unknown_visibility"
    return "seen_not_selected" if seen else "retrieval_miss"


# --------------------------------------------------------------------------- statistics

def paired_difference(pairs_by_ds):
    """pairs_by_ds: {ds: [(a, b), ...]} -> (mean of b - a, None, None); pooled over datasets, weighted by n.
    Point estimate only."""
    total = sum(len(v) for v in pairs_by_ds.values())
    if not total:
        return (None, None, None)
    point = 0.0
    for pairs in pairs_by_ds.values():
        if pairs:
            point += sum(b - a for a, b in pairs) / total
    return (float(point), None, None)


def median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


# --------------------------------------------------------------------------- writers

def fmt(x, pct=False, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if pct:
        return f"{100 * x:.{digits}f}"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["empty"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def md_table(header, rows, align_right=None):
    align_right = align_right if align_right is not None else [i > 0 for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if a else "---" for a in align_right) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def tex_escape(s):
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
             "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(table.get(ch, ch) for ch in str(s))


class Raw(str):
    """A LaTeX cell that is already escaped."""


def latex_table(header, rows, caption, label, colspec=None, note=None, star=False):
    """booktabs table snippet for the ACL template (\\usepackage{booktabs})."""
    env = "table*" if star else "table"
    colspec = colspec or ("l" + "r" * (len(header) - 1))
    esc = lambda c: c if isinstance(c, Raw) else tex_escape(c)  # noqa: E731
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(esc(h) for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(esc(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\smallskip\\footnotesize {tex_escape(note)}")
    lines += [f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def coverage_md(cov_by_cfg, cfgs):
    rows = []
    for c in cfgs:
        cov = cov_by_cfg[c["name"]]
        cells = []
        for ds in DATASETS:
            x = cov[ds]
            exp = x["expected"]
            cells.append(f"{x['terminal']}/{exp if exp is not None else '?'}")
        vis = Counter()
        for ds in DATASETS:
            vis.update(cov[ds]["visibility"])
        rows.append([c["label"], *cells, sum(cov[d]["pending"] for d in DATASETS),
                     sum(cov[d]["failed"] for d in DATASETS), ", ".join(f"{k}:{v}" for k, v in sorted(vis.items(), key=str)) or "-",
                     sum(cov[d]["metric_mismatch"] for d in DATASETS), sum(cov[d]["extract_errors"] for d in DATASETS)])
    return md_table(["Configuration", "Amazon", "MAG", "Prime", "Pending", "Failed", "Visibility",
                     "Recorded H@1 disagrees", "Extract errors"], rows)


# --------------------------------------------------------------------------- figures

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#a3a29d"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False})
    return plt


def save_fig(fig, name):
    """name may contain a subfolder, e.g. 'appendix/behavior_intents_dev'."""
    (OUT / "figures" / name).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")


def placeholder_fig(name, title, message, width=3.3, height=1.6):
    plt = mpl()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=9, weight="bold", color=INK)
    ax.text(0.5, 0.35, message, ha="center", va="center", fontsize=7.5, color=INK2, wrap=True)
    save_fig(fig, name)
    plt.close(fig)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def load_all(names=None, use_cache=True, tiers=None):
    cfgs = [c for c in configurations() if (names is None or c["name"] in names) and (tiers is None or c["tier"] in tiers)]
    data, cov = {}, {}
    for c in cfgs:
        rows = load_features(c, use_cache=use_cache)
        data[c["name"]] = rows
        cov[c["name"]] = coverage(c, rows)
        print(f"  {c['name']}: {sum(r['terminal'] for r in rows)} terminal / {len(rows)} present", flush=True)
    return cfgs, data, cov


def common_args(parser):
    parser.add_argument("--no-cache", action="store_true", help="re-extract every trajectory")
    parser.add_argument("--configs", nargs="*", help="restrict to these configuration names")
    return parser
