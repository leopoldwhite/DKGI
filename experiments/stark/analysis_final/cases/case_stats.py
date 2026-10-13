"""Full-test frequencies of the behaviours illustrated by the case studies in the appendix.

Read-only on every run tree. No model or API calls. Writes only analysis_final/cases/out/.

Usage (from experiments/stark):
    python3 analysis_final/cases/case_stats.py            # all statistics (~3-6 min, 24 processes)
    python3 analysis_final/cases/case_stats.py --no-saved # skip the slow saved-output scan (710 subset)

Definitions (all on what the model saw; Pi conversation_full.json tool results are byte-identical to the
provider-visible outputs in events.jsonl, checked on the case queries):
  * Python call      a bash call whose command runs python/python3 (heredoc or -c).
  * revision         a Python program whose difflib ratio to some earlier Python program of the same query
                     is >= 0.5 (first 4,000 characters); it joins that program's lineage. A lineage with k
                     programs has k versions.
  * error            tool result flagged isError, or containing a Python traceback.
  * repair           the next Python call after an erroring one is a revision of it (ratio >= 0.5);
                     repaired = that next call has no error.
  * truncated        the result carries the single-stage output cap notice
                     ("[... harness output limit: ...; Complete output saved to ...]").
  * reopen saved     a later call names a harness-saved complete output (tmp/full_outputs/out-*.log).
  * full scan        Python code that iterates over all node shards (glob/listdir/Path.glob over nodes).
  * shard lookup     Python code that computes a shard from an id (// 10000).
  * rg | python      the command pipes rg/grep output into Python.
  * multi-hop        the AST classifier (analyze_graph_behavior) labels the program chained_expansion: a later
                     edge filter uses IDs produced by an earlier edge filter in the same program.
  * grows to multi-hop  a lineage whose first version is not multi-hop and a later version is.
  * one hop, then chained  the first edge-reading program of the query is not multi-hop and a later one is
                     (query level; does not require textual similarity).
  * set operation    the AST classifier labels a program intersection or difference.
  * carries ids      Python code contains an integer of 4-8 digits that is not in the query text and
                     appeared in an earlier tool output of the same query.
  * neighbors capped (Graph tools) the result says "Showing k of N results" with N > k. The share is given
                     over all neighbors calls and over the calls that returned a result (tool results flagged
                     isError excluded).
  * paired           Luna and Astra on the same full-test queries, grouped by which model is correct at H@1.
"""

import argparse
from collections import Counter, defaultdict
import difflib
import json
import os
from pathlib import Path
import re
import sys
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
STARK = HERE.parents[1]
OUTDIR = HERE / "out"
sys.path.insert(0, str(STARK / "analysis_final"))
sys.path.insert(0, str(HERE))

DATASETS = ("amazon", "mag", "prime")
ROOTS = {
    "luna": STARK / "pi_full_test/runs/pi_final",
    "astra": STARK / "astra_runs/runs/pi_astra",
    "A2": STARK / "trials/A2",
    "A3": STARK / "trials/A3",
    "A5": STARK / "trials/A5",
}
LABEL = {"luna": "Full DKGI (GPT-6-Luna)", "astra": "Full DKGI (GPT-6-Astra)", "A2": "Graph tools",
         "A3": "Shell only", "A5": "Graph tools + code"}

PY_RE = re.compile(r'(^|[\s;&|(`])python3?(\s|$)')
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1\b", re.S)
PIPE_TO_PY_RE = re.compile(r'\b(rg|grep)\b[^\n]*\|\s*python3?\b')
FULLSCAN_RE = re.compile(r'glob\.glob\(\s*f?["\'][^"\']*nodes/\*|glob\(\s*f?["\'][^"\']*nodes/\*|'
                         r'os\.listdir\(\s*[^)]*nodes|Path\([^)]*nodes[^)]*\)\.glob|\.glob\(\s*["\']\*\.jsonl|'
                         r'nodes["\']?\s*\)\.glob|for\s+\w+\s+in\s+range\(\s*\d+\s*\)\s*:[^\n]*\n?[^\n]*nodes/')
SHARD_RE = re.compile(r'//\s*10000')
EDGE_FILE_RE = re.compile(r'edges/(\d{1,2})\.tsv')
FULLOUT_RE = re.compile(r'full_outputs/out-')
H6_RE = re.compile(r"\[\.\.\. harness output limit:")
SAVED_RE = re.compile(r"Complete output saved to (\S+?\.log)")
INT_RE = re.compile(r'(?<![\w.])\d{4,8}(?![\w.])')
CAP_RE = re.compile(r"\(Showing (\d+) of (\d+) results")
INTERP_RE = re.compile(r'(^|[\s;&|(`])(awk|perl|python3?|node|jq)\b')
COMM_RE = re.compile(r'(^|[\s;&|(`])comm\b')

_classify = None


def graph_labels(cmd):
    global _classify
    if _classify is None:
        import common  # noqa: E402  (analysis_final/common.py; AST classifier of graph operations)
        _classify = common.classify_command
    try:
        return set(_classify(cmd)["graph"])
    except Exception:  # noqa: BLE001
        return set()


def py_code(cmd):
    m = HEREDOC_RE.search(cmd)
    return m.group(2) if m else cmd


def sim(a, b):
    a, b = a[:4000], b[:4000]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return 0.0
    return sm.ratio()


def results(root, subset=None):
    for ds in DATASETS:
        d = root / ds / "test"
        if not d.is_dir():
            continue
        for q in sorted(os.listdir(d)):
            if subset is not None and (ds, q) not in subset:
                continue
            p = d / q / "result.json"
            if p.exists():
                yield str(p)


def selection_710():
    rows = json.loads((STARK / "trials/selection.json").read_text())
    rows = rows["questions"] if isinstance(rows, dict) else rows
    return {(r["dataset"], str(r["query_id"])) for r in rows}


_GOLD = {}


def gold(ds, q):
    if ds not in _GOLD:
        _GOLD[ds] = {}
        with open(STARK / "private" / ds / "test.jsonl") as f:
            for line in f:
                r = json.loads(line)
                _GOLD[ds][str(r["query_id"])] = [int(x) for x in r["answer_ids"]]
    return _GOLD[ds].get(q, [])


def load_conv(attempt):
    c = json.loads((Path(attempt) / "artifacts/conversation_full.json").read_text())
    query, calls, res = "", [], {}
    for m in c.get("messages", []):
        if m["role"] == "user" and not query:
            query = "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")
        elif m["role"] == "assistant":
            calls += [b for b in m["content"] if b.get("type") == "toolCall"]
        elif m["role"] == "toolResult":
            t = "".join(x.get("text", "") for x in m["content"] if x.get("type") == "text")
            res[m.get("toolCallId")] = (t, bool(m.get("isError")))
    return query, calls, res


def prediction(r):
    pred = r.get("prediction")
    try:
        pred = json.loads(pred) if isinstance(pred, str) else (pred or {})
    except ValueError:
        pred = {}
    out = []
    for x in pred.get("node_ids") or []:
        try:
            x = int(x)
        except (TypeError, ValueError):
            continue
        if x not in out:
            out.append(x)
    return out


def analyse(job):
    cfg, res_path = job
    r = json.loads(Path(res_path).read_text())
    ds, q = r["dataset"], str(r["query_id"])
    att = r.get("attempt_path")
    row = {"cfg": cfg, "dataset": ds, "qid": q, "status": r.get("status")}
    g = gold(ds, q)
    pred = prediction(r)
    ranks = [i + 1 for i, x in enumerate(pred) if x in set(g)]
    row.update(hit1=int(bool(ranks) and ranks[0] == 1), rank=ranks[0] if ranks else None, returned=len(pred))
    try:
        query, calls, res = load_conv(att)
    except (OSError, ValueError, TypeError) as e:
        row["err"] = f"conv:{type(e).__name__}"
        return row
    q_ints = set(INT_RE.findall(query.split("Query:", 1)[-1]))
    out_ints = set()
    py = []          # python programs: dicts
    fam = []
    feats = Counter()
    trunc_any = False
    saved_paths = []
    for i, b in enumerate(calls):
        name = b.get("name")
        a = b.get("arguments") or {}
        cmd = a.get("command") or ""
        text, is_err = res.get(b.get("id"), ("", False))
        feats["calls"] += 1
        feats[f"tool_{name}"] += 1
        if H6_RE.search(text):
            trunc_any = True
            feats["trunc_calls"] += 1
            saved_paths += SAVED_RE.findall(text)
        blob = cmd if name == "bash" else json.dumps(a)
        if FULLOUT_RE.search(blob):
            feats["reopen_saved"] += 1
        if cfg == "A2" and name == "neighbors":
            feats["neighbors"] += 1
            if is_err:
                feats["neighbors_failed"] += 1
            m = CAP_RE.search(text)
            if m and int(m.group(2)) > int(m.group(1)):
                feats["neighbors_capped"] += 1
        if cfg == "A3" and name == "bash":
            if INTERP_RE.search(re.sub(r"'[^']*'|\"[^\"]*\"", "", cmd)):
                feats["interp_attempt"] += 1
            if COMM_RE.search(cmd):
                feats["comm"] += 1
        if name == "bash" and PY_RE.search(cmd):
            code = py_code(cmd)
            err = is_err or "Traceback (most recent call last)" in text
            edges = set(EDGE_FILE_RE.findall(code))
            labels = graph_labels(cmd) if cfg in ("luna", "astra") else set()
            multihop = "chained_expansion" in labels
            edge_prog = bool(edges) or "edge_read" in labels
            best, bj = 0.0, None
            for j, p in enumerate(py):
                s = sim(code, p["code"])
                if s > best:
                    best, bj = s, j
            f = fam[bj] if best >= 0.5 else (max(fam) + 1 if fam else 0)
            fam.append(f)
            ints = set(INT_RE.findall(code)) - q_ints
            rec = {"code": code, "err": err, "trunc": bool(H6_RE.search(text)), "sim": best, "fam": f,
                   "multihop": multihop, "edges": len(edges), "edge_prog": edge_prog,
                   "setop": bool(labels & {"intersection", "difference"}), "diff": "difference" in labels,
                   "fullscan": bool(FULLSCAN_RE.search(code)), "shard": bool(SHARD_RE.search(code)),
                   "pipe": bool(PIPE_TO_PY_RE.search(cmd)), "carry": bool(ints & out_ints),
                   "empty": text.strip() in ("", "(no output)")}
            py.append(rec)
        out_ints.update(INT_RE.findall(text[:200000]))
    n = len(py)
    row.update(n_calls=feats["calls"], n_py=n, trunc_any=trunc_any, trunc_calls=feats["trunc_calls"],
               reopen_saved=feats["reopen_saved"] > 0, saved_paths=saved_paths)
    for k in ("neighbors", "neighbors_capped", "neighbors_failed", "interp_attempt", "comm"):
        if feats[k]:
            row[k] = feats[k]
    if cfg == "A5":
        p = Path(att) / "tool_requests.jsonl"
        if p.exists():
            with open(p) as f:
                row["service_calls"] = sum(1 for line in f if '"phase":"agent"' in line or '"phase": "agent"' in line)
        row["cells"] = feats["tool_bash"]
    if n:
        sizes = Counter(fam)
        errs = [k for k, p in enumerate(py) if p["err"]]
        rep = rep_ok = 0
        for k in errs:
            if k + 1 < n and sim(py[k + 1]["code"], py[k]["code"]) >= 0.5:
                rep += 1
                rep_ok += not py[k + 1]["err"]
        grows = False
        first = {}
        for p in py:
            if p["fam"] not in first:
                first[p["fam"]] = p
            elif p["multihop"] and not first[p["fam"]]["multihop"]:
                grows = True
        edge_progs = [p for p in py if p["edge_prog"]]
        one_hop_then_chained = bool(edge_progs) and not edge_progs[0]["multihop"] and any(
            p["multihop"] for p in edge_progs[1:])
        lin = defaultdict(list)
        for p in py:
            lin[p["fam"]].append(p)
        multi = [v for v in lin.values() if len(v) >= 2]
        row.update(
            py_revisions=sum(p["sim"] >= 0.5 for p in py), lineages=len(sizes), longest_lineage=max(sizes.values()),
            py_err=len(errs), py_repair=rep, py_repair_ok=rep_ok,
            py_trunc=sum(p["trunc"] for p in py), py_fullscan=sum(p["fullscan"] for p in py),
            py_shard=sum(p["shard"] for p in py), py_pipe=sum(p["pipe"] for p in py), py_carry=sum(p["carry"] for p in py),
            py_multihop=sum(p["multihop"] for p in py), grows_multihop=grows,
            one_hop_then_chained=one_hop_then_chained, py_edge=len(edge_progs), py_setop=sum(p["setop"] for p in py),
            py_diff=sum(p["diff"] for p in py), py_empty=sum(p["empty"] for p in py),
            lin_first_trunc=sum(v[0]["trunc"] for v in multi), lin_last_trunc=sum(v[-1]["trunc"] for v in multi),
            lin_first_err=sum(v[0]["err"] for v in multi), lin_last_err=sum(v[-1]["err"] for v in multi),
            lin_n=len(multi))
    tmp = Path(att) / "tmp"
    if tmp.is_dir():
        row["scratch_files"] = sum(1 for x in tmp.iterdir() if x.name != "full_outputs")
    return row


def scan_saved(job):
    """Wrong answers on the 710 subset: is a gold id in a saved complete output while never visible?"""
    row = job
    g = gold(row["dataset"], row["qid"])
    if not g or not row.get("saved_paths"):
        return {**row, "gold_saved_only": False}
    pat = re.compile((r'(?<!\d)(%s)(?!\d)' % "|".join(map(str, g))).encode())
    found = False
    for p in row["saved_paths"]:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            got = 0
            while got < 300_000_000:  # read at most 300 MB per saved file
                b = f.read(8_000_000)
                if not b:
                    break
                got += len(b)
                if pat.search(b):
                    found = True
                    break
        if found:
            break
    return {**row, "gold_saved_only": found}


def gold_visible(cfg, ds, q):
    r = json.loads((ROOTS[cfg] / ds / "test" / q / "result.json").read_text())
    _, calls, res = load_conv(r["attempt_path"])
    g = gold(ds, q)
    pat = re.compile(r'(?<!\d)(%s)(?!\d)' % "|".join(map(str, g)))
    return any(pat.search(res.get(b.get("id"), ("", False))[0]) for b in calls)


def gv_job(row):
    return {**row, "gold_visible": gold_visible(row["cfg"], row["dataset"], row["qid"])}


def pct(a, b):
    return None if not b else round(100.0 * a / b, 1)


def summarize(rows):
    S = {}
    by = defaultdict(list)
    for r in rows:
        if r.get("err"):
            continue
        by[(r["cfg"], r["dataset"])].append(r)
        by[(r["cfg"], "all")].append(r)
    for (cfg, ds), L in sorted(by.items()):
        n = len(L)
        npy = sum(r.get("n_py", 0) for r in L)
        perr = sum(r.get("py_err", 0) for r in L)
        lin_n = sum(r.get("lin_n", 0) for r in L)
        s = {"n_queries": n, "H@1": pct(sum(r["hit1"] for r in L), n),
             "py_calls": npy, "py_per_query": round(npy / n, 2) if n else None,
             "q_trunc_any": pct(sum(r["trunc_any"] for r in L), n),
             "q_reopen_saved": pct(sum(r["reopen_saved"] for r in L), n),
             "q_trunc_not_reopened": pct(sum(r["trunc_any"] and not r["reopen_saved"] for r in L), n),
             "q_empty_prediction": sum(r["returned"] == 0 for r in L),
             "q_scratch_files": pct(sum(r.get("scratch_files", 0) > 0 for r in L), n)}
        if npy:
            s.update({
                "py_revision_share": pct(sum(r.get("py_revisions", 0) for r in L), npy),
                "q_with_py": pct(sum(r.get("n_py", 0) > 0 for r in L), n),
                "q_lineage_ge4": pct(sum(r.get("longest_lineage", 0) >= 4 for r in L), n),
                "q_py_error": pct(sum(r.get("py_err", 0) > 0 for r in L), n),
                "py_error_share": pct(perr, npy),
                "err_followed_by_repair": pct(sum(r.get("py_repair", 0) for r in L), perr),
                "err_repaired": pct(sum(r.get("py_repair_ok", 0) for r in L), perr),
                "py_fullscan_share": pct(sum(r.get("py_fullscan", 0) for r in L), npy),
                "fullscan_per_query": round(sum(r.get("py_fullscan", 0) for r in L) / n, 2),
                "q_shard_lookup": pct(sum(r.get("py_shard", 0) > 0 for r in L), n),
                "py_pipe_share": pct(sum(r.get("py_pipe", 0) for r in L), npy),
                "q_pipe": pct(sum(r.get("py_pipe", 0) > 0 for r in L), n),
                "py_carry_share": pct(sum(r.get("py_carry", 0) for r in L), npy),
                "q_carry": pct(sum(r.get("py_carry", 0) > 0 for r in L), n),
                "q_multihop": pct(sum(r.get("py_multihop", 0) > 0 for r in L), n),
                "q_grows_multihop": pct(sum(bool(r.get("grows_multihop")) for r in L), n),
                "q_one_hop_then_chained": pct(sum(bool(r.get("one_hop_then_chained")) for r in L), n),
                "q_py_edges": pct(sum(r.get("py_edge", 0) > 0 for r in L), n),
                "q_setop": pct(sum(r.get("py_setop", 0) > 0 for r in L), n),
                "q_difference": pct(sum(r.get("py_diff", 0) > 0 for r in L), n),
                "py_empty_share": pct(sum(r.get("py_empty", 0) for r in L), npy),
                "lineages_multi": lin_n,
                "lineage_first_trunc": pct(sum(r.get("lin_first_trunc", 0) for r in L), lin_n),
                "lineage_last_trunc": pct(sum(r.get("lin_last_trunc", 0) for r in L), lin_n),
                "lineage_first_err": pct(sum(r.get("lin_first_err", 0) for r in L), lin_n),
                "lineage_last_err": pct(sum(r.get("lin_last_err", 0) for r in L), lin_n),
            })
        if cfg == "A2":
            nb = sum(r.get("neighbors", 0) for r in L)
            nb_failed = sum(r.get("neighbors_failed", 0) for r in L)
            s.update(neighbors_calls=nb, neighbors_failed=nb_failed,
                     neighbors_capped_share=pct(sum(r.get("neighbors_capped", 0) for r in L), nb),
                     neighbors_capped_share_of_returned=pct(sum(r.get("neighbors_capped", 0) for r in L),
                                                            nb - nb_failed),
                     q_neighbors_capped=pct(sum(r.get("neighbors_capped", 0) > 0 for r in L), n),
                     calls_per_query=round(sum(r["n_calls"] for r in L) / n, 1))
        if cfg == "A3":
            s.update(q_interp_attempt=pct(sum(r.get("interp_attempt", 0) > 0 for r in L), n),
                     q_comm=pct(sum(r.get("comm", 0) > 0 for r in L), n),
                     calls_per_query=round(sum(r["n_calls"] for r in L) / n, 1))
        if cfg == "A5":
            cells = sum(r.get("cells", 0) for r in L)
            sc = sum(r.get("service_calls", 0) for r in L)
            s.update(cells=cells, service_calls=sc, service_per_cell=round(sc / cells, 1) if cells else None,
                     calls_per_query=round(sum(r["n_calls"] for r in L) / n, 1))
        if "gold_saved_only" in L[0]:
            wrong = [r for r in L if not r["hit1"]]
            hid = [r for r in wrong if r.get("gold_saved_only") and not r.get("gold_visible")]
            s.update(wrong=len(wrong), wrong_gold_only_in_saved=len(hid),
                     wrong_gold_only_in_saved_reopened=sum(r["reopen_saved"] for r in hid))
        S[f"{cfg}|{ds}"] = s
    return S


def paired(rows):
    """Luna and Astra on the same full-test queries, split by which model answers correctly (H@1)."""
    by = defaultdict(dict)
    for r in rows:
        if r["cfg"] in ("luna", "astra") and not r.get("err"):
            by[(r["dataset"], r["qid"])][r["cfg"]] = r
    groups = defaultdict(list)
    for k, d in by.items():
        if len(d) == 2:
            lab = {(1, 1): "both_correct", (0, 1): "astra_only_correct", (1, 0): "luna_only_correct",
                   (0, 0): "both_wrong"}[(d["luna"]["hit1"], d["astra"]["hit1"])]
            groups[lab].append(d)
            groups["all"].append(d)
    out = {}
    for lab, L in sorted(groups.items()):
        n = len(L)
        g = {"n_queries": n}
        for cfg in ("luna", "astra"):
            R = [d[cfg] for d in L]
            npy = sum(r.get("n_py", 0) for r in R)
            g[cfg] = {"q_reopen_saved": pct(sum(r["reopen_saved"] for r in R), n),
                      "q_pipe": pct(sum(r.get("py_pipe", 0) > 0 for r in R), n),
                      "py_pipe_share": pct(sum(r.get("py_pipe", 0) for r in R), npy),
                      "py_fullscan_share": pct(sum(r.get("py_fullscan", 0) for r in R), npy),
                      "q_shard_lookup": pct(sum(r.get("py_shard", 0) > 0 for r in R), n)}
        out[lab] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-saved", action="store_true")
    ap.add_argument("--processes", type=int, default=24)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sel = selection_710()
    jobs = [(c, p) for c in ("luna", "astra") for p in results(ROOTS[c])]
    jobs += [(c, p) for c in ("A2", "A3", "A5") for p in results(ROOTS[c], sel)]
    with Pool(args.processes) as pool:
        rows = pool.map(analyse, jobs, chunksize=16)
    full = [r for r in rows if r["cfg"] in ("luna", "astra")]
    sub = [dict(r, cfg=r["cfg"] + "_710") for r in full if (r["dataset"], r["qid"]) in sel]
    abl = [r for r in rows if r["cfg"] in ("A2", "A3", "A5")]
    if not args.no_saved:
        wrong = [r for r in sub if not r["hit1"] and not r.get("err")]
        with Pool(args.processes) as pool:
            wrong = pool.map(gv_job, [dict(r, cfg=r["cfg"].replace("_710", "")) for r in wrong], chunksize=4)
            wrong = pool.map(scan_saved, wrong, chunksize=2)
        flag = {(r["cfg"] + "_710", r["dataset"], r["qid"]): r for r in wrong}
        for r in sub:
            w = flag.get((r["cfg"], r["dataset"], r["qid"]))
            r["gold_saved_only"] = bool(w and w["gold_saved_only"])
            r["gold_visible"] = bool(w and w["gold_visible"]) if w else True
    stats = summarize(full + sub + abl)
    errors = Counter(r["err"] for r in rows if r.get("err"))
    meta = {"generated": "analysis_final/cases/case_stats.py", "jobs": len(jobs), "extraction_errors": dict(errors),
            "configs": {k: str(v.relative_to(STARK)) for k, v in ROOTS.items()},
            "subset_710": "trials/selection.json", "saved_scan": not args.no_saved}
    pairs = paired(full)
    (OUTDIR / "case_stats.json").write_text(json.dumps({"meta": meta, "stats": stats, "paired": pairs}, indent=1))
    with open(OUTDIR / "case_features.jsonl", "w") as f:
        for r in full + abl:
            f.write(json.dumps({k: v for k, v in r.items() if k != "saved_paths"}) + "\n")
    print(json.dumps(meta, indent=1))
    keys = ["n_queries", "H@1", "py_per_query", "q_trunc_any", "q_reopen_saved", "q_trunc_not_reopened",
            "q_lineage_ge4", "q_py_error", "err_followed_by_repair", "err_repaired", "py_fullscan_share",
            "q_shard_lookup", "py_pipe_share", "q_pipe", "q_py_edges", "q_multihop", "q_grows_multihop", "q_one_hop_then_chained", "q_setop", "q_difference",
            "q_carry", "py_revision_share", "lineage_first_trunc", "lineage_last_trunc", "lineage_first_err",
            "lineage_last_err", "q_empty_prediction", "q_scratch_files", "neighbors_capped_share",
            "neighbors_capped_share_of_returned", "neighbors_failed",
            "q_neighbors_capped", "q_interp_attempt", "q_comm", "service_per_cell", "calls_per_query", "wrong",
            "wrong_gold_only_in_saved", "wrong_gold_only_in_saved_reopened"]
    lines = ["# Case-study frequencies (generated by case_stats.py; do not edit)", "",
             "Definitions: see the module docstring. Percentages are of queries (q_*) or of Python calls (py_*).", ""]
    cols = sorted(stats)
    lines.append("| metric | " + " | ".join(c.replace("|", " ") for c in cols) + " |")
    lines.append("|---|" + "---:|" * len(cols))
    for k in keys:
        vals = [stats[c].get(k) for c in cols]
        if all(v is None for v in vals):
            continue
        lines.append(f"| {k} | " + " | ".join("" if v is None else str(v) for v in vals) + " |")
    lines += ["", "## Luna and Astra on the same queries (full test, split by which model is correct at H@1)", "",
              "| group | n | metric | Luna | Astra |", "|---|---:|---|---:|---:|"]
    for lab in ("all", "both_correct", "astra_only_correct", "luna_only_correct", "both_wrong"):
        g = pairs.get(lab)
        if not g:
            continue
        for k in g["luna"]:
            lines.append(f"| {lab} | {g['n_queries']} | {k} | {g['luna'][k]} | {g['astra'][k]} |")
    (OUTDIR / "case_stats.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUTDIR / "case_stats.json", OUTDIR / "case_stats.md")


if __name__ == "__main__":
    main()
