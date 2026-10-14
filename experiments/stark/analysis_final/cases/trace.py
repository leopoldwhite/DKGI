"""Dump one STaRK test query across configurations: query, gold, outcome and the model-visible trajectory.

Read-only on every run tree and on the corpus. No model or API calls.
Writes only under analysis_final/cases/traces/.

Usage (from experiments/stark):
    python3 analysis_final/cases/trace.py mag 9090            # all configurations, writes traces/mag_9090/*.txt
    python3 analysis_final/cases/trace.py mag 9090 --cfg luna --max-out 1500 --stdout

Sources of model-visible output (same rule as analysis_final/common.py):
  * Pi (Full DKGI and the ablation variants): attempt/artifacts/events.jsonl, provider_request_context
    native Responses or Messages tool records, union over all requests.
  * Codex: attempt/stdout.jsonl command_execution items (command and raw aggregated output; the raw output
    is a superset of what the model saw) plus attempt/model_tool_outputs.jsonl when present.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARK = HERE.parents[1]
sys.path.insert(0, str(STARK / "harness_fixes"))
from pi_context import provider_items  # noqa: E402

# Paper names; internal ids only as keys.
CONFIGS = {
    "luna": ("Full DKGI (GPT-6-Luna)", "pi", STARK / "pi_full_test/runs/pi_final"),
    "astra": ("Full DKGI (GPT-6-Astra)", "pi", STARK / "astra_runs/runs/pi_astra"),
    "A5": ("Graph tools + code", "pi", STARK / "trials/A5"),
    "A2": ("Graph tools", "pi", STARK / "trials/A2"),
    "A3": ("Shell only", "pi", STARK / "trials/A3"),
    "A1": ("Retrieval only", "pi", STARK / "trials/A1"),
    "A6": ("No edges", "pi", STARK / "trials/A6"),
    "H": ("Full DKGI (Codex, GPT-6-Luna)", "codex", STARK / "trials/codex_h/codex_final"),
    "codex_astra": ("Full DKGI (Codex, GPT-6-Astra)", "codex", STARK / "astra_runs/runs/codex_astra"),
}
ORDER = ["luna", "astra", "A5", "A2", "A3", "A1", "A6", "H", "codex_astra"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def query_row(ds, qid):
    for split in ("test-0.1", "test"):
        p = STARK / "public" / ds / f"{split}.jsonl"
        for r in read_jsonl(p):
            if str(r["query_id"]) == str(qid):
                return r, split
    raise KeyError((ds, qid))


def gold(ds, qid):
    for r in read_jsonl(STARK / "private" / ds / "test.jsonl"):
        if str(r["query_id"]) == str(qid):
            return [int(x) for x in r["answer_ids"]]
    raise KeyError((ds, qid))


def node_records(ds, ids):
    """Return {id: record} for the given node ids from corpus/<ds>/nodes/NNNN.jsonl (shard = id // 10000)."""
    want = {int(i) for i in ids}
    shards = {}
    for i in want:
        shards.setdefault(i // 10000, set()).add(i)
    out = {}
    for s, ss in shards.items():
        p = STARK / "corpus" / ds / "nodes" / f"{s:04d}.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                m = re.match(r'\{"node_id":\s*(\d+)', line)
                if m and int(m.group(1)) in ss:
                    out[int(m.group(1))] = json.loads(line)
    return out


def node_title(rec):
    if not rec:
        return "?"
    a = rec.get("attributes") or rec
    for k in ("title", "name", "DisplayName", "OriginalTitle"):
        v = a.get(k) if isinstance(a, dict) else None
        if v:
            return str(v)[:120]
    return json.dumps(rec)[:120]


def result_of(cfg, ds, qid):
    root = CONFIGS[cfg][2]
    p = root / ds / "test" / str(qid) / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def prediction_ids(r):
    pred = r.get("prediction")
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except ValueError:
            m = re.search(r"\{.*\}", pred, re.S)
            pred = json.loads(m.group(0)) if m else {}
    ids = (pred or {}).get("node_ids") or []
    seen, out = set(), []
    for x in ids:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def outcome(cfg, ds, qid, g):
    r = result_of(cfg, ds, qid)
    if r is None:
        return None
    ids = prediction_ids(r)
    gs = set(g)
    ranks = [i + 1 for i, x in enumerate(ids) if x in gs]
    m = r.get("metrics") or {}
    return {"cfg": cfg, "label": CONFIGS[cfg][0], "status": r.get("status"), "hit1_recorded": m.get("hit_at_1"),
            "hit1": 1.0 if ranks and ranks[0] == 1 else 0.0, "first_gold_rank": ranks[0] if ranks else None,
            "gold_in_list": len(ranks), "returned": len(ids), "top": ids[:5],
            "tool_calls": r.get("tool_calls"), "attempt": r.get("attempt_path"), "seconds": r.get("elapsed_seconds") or r.get("duration_seconds")}


def _text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(b.get("text", "") for b in output if isinstance(b, dict))
    return "" if output is None else str(output)


def pi_trace(attempt):
    path = Path(attempt) / "artifacts/events.jsonl"
    calls, outs = {}, {}
    n_exec = 0
    with open(path, "rb") as f:
        for raw in f:
            head = raw[:200]
            if b'"tool_execution_start"' in head:
                n_exec += 1
            if b'"provider_request_context"' not in head:
                continue
            for it in provider_items(json.loads(raw).get("payload", {})):
                t = it.get("type")
                if t == "function_call" and it["call_id"] not in calls:
                    try:
                        args = json.loads(it.get("arguments") or "{}")
                    except ValueError:
                        args = {"raw": it.get("arguments")}
                    calls[it["call_id"]] = {"name": it.get("name"), "args": args}
                elif t == "function_call_output":
                    outs.setdefault(it["call_id"], _text(it.get("output")))
    # The final call(s) after the last provider request have no recorded output in any request context;
    # take them from tool_execution_end so the trace is complete.
    ends = {}
    with open(path, "rb") as f:
        for raw in f:
            if b'"tool_execution_end"' in raw[:200]:
                e = json.loads(raw)
                ends[e.get("toolCallId")] = e
    for cid, e in ends.items():
        base = cid.split("|")[0]
        key = cid if cid in calls else (base if base in calls else None)
        if key is None:
            calls[cid] = {"name": e.get("toolName"), "args": {}}
            key = cid
        if key not in outs:
            res = e.get("result") or {}
            outs[key] = _text(res.get("content")) if isinstance(res, dict) else _text(res)
    dur = {}
    conv = Path(attempt) / "artifacts/conversation_full.json"
    if conv.exists():
        try:
            for m in json.loads(conv.read_text()).get("messages", []):
                te = m.get("tool_execution") or {}
                if m.get("role") == "toolResult" and te.get("duration_seconds") is not None:
                    tid = m.get("toolCallId") or ""
                    dur[tid] = dur[tid.split("|")[0]] = te["duration_seconds"]
        except ValueError:
            pass
    rows = [{"name": v["name"], "args": v["args"], "output": outs.get(k), "visible": k in outs,
             "seconds": dur.get(k, dur.get(k.split("|")[0]))} for k, v in calls.items()]
    return rows, n_exec


def codex_trace(attempt):
    path = Path(attempt) / "stdout.jsonl"
    cmds = {}
    with open(path) as f:
        for line in f:
            if '"command_execution"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            it = e.get("item", {})
            c = cmds.setdefault(it["id"], {"name": "shell", "args": {"command": it.get("command", "")}, "output": None,
                                           "visible": False})
            if e.get("type") == "item.completed":
                c["output"] = it.get("aggregated_output", "")
                c["exit_code"] = it.get("exit_code")
    return list(cmds.values()), len(cmds)


def unwrap(cmd):
    m = re.match(r"^/usr/bin/bash -lc (['\"])(.*)\1$", cmd, re.S)
    if m:
        s = m.group(2)
        return s.replace("'\"'\"'", "'") if m.group(1) == "'" else s
    return cmd


def short(s, n):
    if s is None:
        return "<no output recorded>"
    if len(s) <= n:
        return s
    h = n * 2 // 3
    return s[:h] + f"\n[... {len(s) - n} chars omitted in this dump ...]\n" + s[-(n - h):]


def render(cfg, ds, qid, g, max_out=2500, max_cmd=None):
    o = outcome(cfg, ds, qid, g)
    if o is None:
        return None
    harness = CONFIGS[cfg][1]
    rows, n_exec = (pi_trace if harness == "pi" else codex_trace)(o["attempt"])
    lines = [f"# {CONFIGS[cfg][0]}  [{cfg}]  {ds} {qid}",
             f"status={o['status']} H@1={o['hit1']} (recorded {o['hit1_recorded']}) first_gold_rank={o['first_gold_rank']} "
             f"gold_in_list={o['gold_in_list']}/{len(g)} returned={o['returned']} tool_calls={o['tool_calls']} "
             f"(trace calls {len(rows)}, exec events {n_exec})",
             f"attempt: {o['attempt']}", ""]
    for i, r in enumerate(rows, 1):
        a = r["args"]
        if r["name"] in ("bash", "shell"):
            body = unwrap(a.get("command", ""))
        elif r["name"] == "read":
            body = json.dumps(a)
        else:
            body = json.dumps(a, ensure_ascii=False)
        if max_cmd:
            body = short(body, max_cmd)
        lines.append(f"=== call {i} [{r['name']}] ===")
        lines.append(body)
        out = r["output"]
        lines.append(f"--- output ({0 if out is None else len(out)} chars{'' if r['visible'] else ', raw'}"
                     f"{'' if r.get('exit_code') in (None, 0) else ', exit ' + str(r.get('exit_code'))}) ---")
        lines.append(short(out, max_out))
        lines.append("")
    return "\n".join(lines), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("qid")
    ap.add_argument("--cfg", nargs="*", default=ORDER)
    ap.add_argument("--max-out", type=int, default=2500)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    ds, qid = args.dataset, args.qid
    q, split = query_row(ds, qid)
    g = gold(ds, qid)
    recs = node_records(ds, g)
    head = [f"{ds} {qid} (public split file: {split})", f"Query: {q['query']}",
            "Gold: " + "; ".join(f"{i} {node_title(recs.get(i))}" for i in g), ""]
    outdir = HERE / "traces" / f"{ds}_{qid}"
    summary = list(head)
    for cfg in args.cfg:
        got = render(cfg, ds, qid, g, args.max_out)
        if got is None:
            summary.append(f"{cfg:12s} missing")
            continue
        text, o = got
        summary.append(f"{cfg:12s} {o['label']:40s} H@1={o['hit1']:.0f} rank={o['first_gold_rank']} "
                       f"calls={o['tool_calls']} top5={o['top']}")
        if args.stdout:
            print(text)
        else:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{cfg}.txt").write_text("\n".join(head) + "\n" + text + "\n")
    if not args.stdout:
        (outdir / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
