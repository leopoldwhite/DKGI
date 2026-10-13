"""Behaviour study: tool coding and tool calling within one query (offline, existing trajectories).

    python3 analysis_final/coevolution.py              # reuse the per-query cache where still valid
    python3 analysis_final/coevolution.py --no-cache   # re-extract every trajectory
    python3 analysis_final/coevolution.py --configs H codex_full   # re-extract / report a subset

What a "tool" is here. The agent never gets a tool-building API. A *program* is one inline Python source
(``python3 -c`` or a heredoc) issued inside one shell call. Programs are written into the call, run once, and
are not saved or imported later (section 2 of the report checks this). The harness is fixed. "Co-evolution"
below means only that, within one query, the programs the agent writes and the tool calls it makes change in
response to each other's outputs: later programs are edited copies of earlier ones, embed ids that outputs of
earlier turns showed, and change after harness feedback such as the single-stage output cap (association only).

Read-only with respect to every run tree. No model or API call. Writes only to
analysis_final/out/coevolution/ (cache/ with its .gitignore, figures/, tex/, REPORT.md, coevolution.json,
lineage_pairs_sample.md).

Sources per harness:
  * Pi:    attempt/artifacts/events.jsonl. Calls and model-visible outputs come from the provider_request_context
           events (last request; union over requests if the run compacted), as in common.pi_calls. isError comes
           from tool_execution_end; call windows (for mapping graph-service requests to calls) from message_end
           timestamps. Graph-service requests (Graph tools, Graph tools + code): attempt/tool_requests.jsonl.
  * Codex: attempt/codex_trace/*/trace.jsonl (exec_command invocations and results, in order) linked to the
           model-visible code-cell outputs in attempt/model_tool_outputs.jsonl. Fallback without a trace:
           stdout.jsonl command_execution items (raw output; marked visibility="raw").

Config "codex_full" (Codex + GPT-6-Luna on the full test: trials/codex_h/codex_final, preset codex_luna_full) reads
the terminal results and reports their n (the cache is keyed by result path, status and finish time; plain
comparisons only).
"""

import argparse
from collections import Counter, defaultdict
import difflib
import json
import math
import os
from pathlib import Path
import re
import time

import common as C

base, graph = C.base, C.graph

OUTDIR = C.OUT / "coevolution"
CACHE = OUTDIR / "cache"
# Plain-text version of the extraction logic; cached rows with another value are re-extracted.
VERSION = "2026-10-02-j"

SIM_CAP = 4000          # characters compared per program (difflib ratio)
THRESHOLDS = (0.5, 0.8)

# --------------------------------------------------------------------------- configurations

def configurations():
    S = C.STARK
    PI = S / "pi_full_test/runs/pi_final"
    AST = S / "astra_runs/runs/pi_astra"
    ABL = S / "trials"
    H = S / "trials/codex_h/codex_final"

    def cfg(name, paper, short, harness, kind, root, source, subset=None, expected=None, note=""):
        return {"name": name, "paper": paper, "short": short, "harness": harness, "kind": kind, "root": root,
                "source": source, "subset": subset, "expected": expected, "note": note}

    return [
        cfg("luna_full", "Full DKGI, Pi + GPT-6-Luna (full test)", "Luna", "pi", "file", PI, "luna_full",
            expected=7108),
        cfg("astra_full", "Full DKGI, Pi + GPT-6-Astra (full test)", "Astra", "pi", "file", AST, "astra_full",
            expected=7108),
        cfg("A4", "Full DKGI", "Full DKGI", "pi", "file", PI, "luna_full", subset=C.SEL_710, expected=710),
        cfg("A5", "Graph tools + code", "Graph tools + code", "pi", "tool_code", ABL / "A5", "A5", expected=710),
        cfg("A2", "Graph tools", "Graph tools", "pi", "tool", ABL / "A2", "A2", expected=710),
        cfg("A3", "Shell only", "Shell only", "pi", "shell", ABL / "A3", "A3", expected=710),
        cfg("A6", "No edges", "No edges", "pi", "file", ABL / "A6", "A6", expected=710),
        cfg("H", "Full DKGI (Codex)", "Codex", "codex", "file", H, "codex_luna", subset=C.SEL_710, expected=710),
        cfg("codex_full", "Full DKGI (Codex), Codex + GPT-6-Luna (full test)", "Codex full", "codex",
            "file", H, "codex_luna", expected=7108, note="partial: only results terminal at run time"),
    ]


def iter_results(cfg):
    """(result, path) for every terminal result.json of a configuration (pending queries are skipped)."""
    subset = C.load_selection(cfg["subset"]) if cfg.get("subset") else None
    for ds in C.DATASETS:
        d = Path(cfg["root"]) / ds / "test"
        if not d.is_dir():
            continue
        for qdir in sorted(d.iterdir(), key=lambda p: p.name):
            if subset is not None and (ds, qdir.name) not in subset:
                continue
            path = qdir / "result.json"
            if not path.exists():
                continue
            try:
                r = C.read_json(path)
            except ValueError:
                continue  # being written
            if not r.get("terminal", r.get("status") in ("scored", "failed")):
                continue
            yield r, str(path)


# --------------------------------------------------------------------------- shell / program parsing

WRAP = re.compile(r"^\s*(?:/usr/bin/|/bin/)?(?:bash|sh|zsh|dash)\s+-\w*c\s")
PY_EXE = re.compile(r"python(?:\d+(?:\.\d+)?)?")
PY_RE = re.compile(r"(?:^|[\s;&|(`])(?:/usr/bin/)?python3?(?:\.\d+)?(?=\s|$)")
SEARCH_RE = re.compile(r"(?:^|[\s;&|(`])(?:rg|grep|egrep|fgrep)(?=\s)")
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\s*\1\s*(?:\n|$)", re.S)
PYC_RE = re.compile(r"python3?\s+(?:-\w+\s+)*-c\s+(['\"])(.*?)(?<!\\)\1", re.S)
SCRIPT_EXT = re.compile(r"\.(?:py|sh|bash|pl)$")
RUN_SCRIPT_RE = re.compile(r"(?:python3?(?:\.\d+)?\s+(?:-u\s+)?|(?:^|[\s;&|])(?:bash|sh|source|\.)\s+|(?:^|[\s;&|])\./)"
                           r"(?!-)([^\s|;&<>'\"]+\.(?:py|sh|bash|pl))\b")
PY_WRITE_RE = re.compile(r"open\(\s*(?:f?['\"])([^'\"]+)['\"]\s*,\s*(?:mode\s*=\s*)?['\"][wax]|"
                         r"Path\(\s*f?['\"]([^'\"]+)['\"]\s*\)\.write_(?:text|bytes)")
PY_IMPORT_LOCAL_RE = re.compile(r"\bexec\(\s*open\(|\brunpy\b|\bimportlib\b|spec_from_file_location")
DEF_RE = re.compile(r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\(")

SHARD_RE = re.compile(r"//\s*10000|/\s*10000\b|:04d|%04d|\bshard\b|nodes/\{|zfill\(4\)|nodes/[0-9]{4}\.jsonl")
NODE_SCAN_RE = re.compile(r"glob\(\s*[^)]*nodes|nodes/\*|listdir\([^)]*nodes|iterdir\(|scandir\(|os\.walk\(|"
                          r"Path\([^)]*nodes[^)]*\)\.glob|for\s+\w+\s+in\s+range\(\s*\d+\s*\)")
STDIN_RE = re.compile(r"sys\.stdin|fileinput")
EDGE_RE = re.compile(r"edges/|edges['\"]|\.tsv\b")
NODE_RE = re.compile(r"nodes/|nodes['\"]|\.jsonl\b")
FULLOUT_RE = re.compile(r"full_outputs/out-")
SORT_RE = re.compile(r"\bsorted\(|\.sort\(|nlargest|nsmallest|most_common|\bscore\w*\s*(?:\+?=|\[)|\brank\w*\s*=")
SETOP_RE = re.compile(r"\.intersection(?:_update)?\(|\.difference(?:_update)?\(|\.issubset\(|\.isdisjoint\(|"
                      r"\.symmetric_difference\(|&=|-=|\bset\([^)]*\)\s*[&-]\s*|\w\s*&\s*\w|\bnot\s+in\s+\w+_?(?:set|ids|seen)\b")
VERIFY_LOOKUP_RE = re.compile(r"candidate_ids|is_candidate|node_id['\"]?\]?\s*(?:==|in\b)|\bin\s+(?:ids|want|wanted|targets|"
                              r"cands|candidates|check)\b")
KB_CALL_RE = re.compile(r"\.(search|get_node|neighbors|get_schema)\(")

# Field projection / output limit, counted for shell commands only (rg, grep, other shell; not Python). A Python
# pattern for key access matched almost every program, so it would only re-count "runs Python".
SH_PROJ_RE = re.compile(r"(?:^|\s)(?:rg|grep)\s[^|;&]*?(?:\s-[a-zA-Z]*o[a-zA-Z]*\b|--only-matching)|\bcut\s+-[cfd]|"
                        r"\bjq\b|\bawk\b[^|;&]*\{\s*print\s+\$|sed\s+-[nE]+\s+['\"]s")
LIMIT_RE = re.compile(r"\|\s*(?:head|tail)\b|\s-m\s*\d|--max-count|--max-columns|\s-M\s*\d|\bwc\b|\s-c\b|--count|"
                      r"\[\s*:\s*\d+\s*\]|islice|\bbreak\b|\blen\(|Counter\(|most_common")

H6_MARK = re.compile(r"\[\.\.\. harness output limit: [^\n]*?of ([0-9]+) lines \(([0-9]+) bytes\)")
CODEX_MARK = C.CODEX_MARK
# The interpreter itself was missing (`python` where only `python3` exists): the program did not run. Kept as a
# program (it was written) but counted apart from failures and left out of failure and repair rates.
PY_NOTFOUND_RE = re.compile(r"\bpython[\d.]*: (?:command not found|not found)")
# Result lines of a graph-service neighbors response ("0. -rel-> [ID] name ..."); the header line
# "Search around [SEED] ..." echoes the seed and is not a result.
NB_RESULT_RE = re.compile(r"(?m)^\s*\d+\.\s[^\n\[]*\[(\d+)\]")
EXIT_RE = re.compile(r"Command exited with code ([0-9]+)")
TRACEBACK = "Traceback (most recent call last)"
PY_ERR_RE = re.compile(r"(?m)^(?:SyntaxError|IndentationError|NameError|KeyError|TypeError|ValueError|IndexError|"
                       r"AttributeError|FileNotFoundError|ModuleNotFoundError|ImportError|ZeroDivisionError|"
                       r"JSONDecodeError|UnboundLocalError)\b")
TIMEOUT_RE = re.compile(r"Command timed out|timed out after", re.I)
ERR_NAME_RE = re.compile(r"(?m)^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Exception|Interrupt))\b")

INT_RE = re.compile(r"(?<![\w.\-/])\d{4,8}(?![\w.])")
STRICT_PATTERNS = (re.compile(r"\"node_id\"\s*:\s*(\d+)"), re.compile(r"\[(\d{1,8})\]"),
                   re.compile(r"(?m)(?:^|[:\s])(\d{1,8})\t(\d{1,8})(?=\t|\s*$)"),
                   re.compile(r"(?m)^\s*(\d{3,8})(?=[\s,:|)\]]|$)"), re.compile(r"\bid[=:]\s*(\d+)"),
                   re.compile(r"[\[,]\s*(\d{4,8})(?=\s*[,\]])"))

BOILER = [re.compile(p) for p in (
    r"^(?:import|from)\s+\S+", r"^#", r"^sys\.path\.(?:insert|append)", r"^os\.chdir\(",
    r"^\w+\s*=\s*f?['\"][^'\"]*/(?:corpus|stark|trials|attempt_)[^'\"]*['\"]\s*$",
    r"^for\s+\w+\s+in\s+(?:sorted\()?\s*(?:glob\.glob|glob\(|os\.listdir|Path\(|range\(|\w+\.glob\()",
    r"^for\s+\w+\s+in\s+(?:open\(|f\b|fh\b|h\b|fp\b|src\b|source\b|handle\b|sys\.stdin|csv\.(?:reader|DictReader)\(|r\b|reader\b)",
    r"^with\s+open\(", r"^\w+\s*=\s*json\.loads\(\s*\w+\s*\)\s*$", r"^try\s*:\s*\w+\s*=\s*json\.loads\(",
    r"^(?:try|else|finally)\s*:\s*$", r"^except\b[^:]*:\s*(?:continue|pass)?\s*$", r"^(?:continue|pass)\s*$",
    r"^\w+\s*=\s*csv\.(?:reader|DictReader)\(", r"^kb\s*=\s*kbtools\.Client\(", r"^(?:PY|EOF|PYEOF|END)$",
    r"^if\s+not\s+\w+(?:\.strip\(\))?\s*:\s*continue\s*$",
    r"^if\s+\w+\s*(?:\[0\])?\s*==\s*['\"](?:source_id|src)['\"]\s*:\s*continue\s*$",
)]


def strip_boilerplate(src):
    """Drop boilerplate statements; one-line programs are split at ';' so `import json; ...` keeps its body."""
    keep = []
    for line in src.splitlines():
        for s in line.split(";"):
            s = s.strip()
            if not s or any(p.search(s) for p in BOILER):
                continue
            keep.append(s)
    return "\n".join(keep)


def similarity(a, b):
    """difflib ratio on the first SIM_CAP characters; returns 0.0 when the cheap upper bounds are below 0.5."""
    if not a or not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a[:SIM_CAP], b[:SIM_CAP], autojunk=False)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return 0.0
    return sm.ratio()


def ints_loose(text, exclude):
    out = set()
    for m in INT_RE.findall(text):
        v = int(m)
        if 1900 <= v <= 2035 or v % 1000 == 0 or m in exclude:
            continue
        out.add(v)
    return out


def ints_strict(text):
    out = set()
    for rx in STRICT_PATTERNS:
        for m in rx.finditer(text):
            for g in m.groups():
                if g and 4 <= len(g) <= 8:
                    v = int(g)
                    if not (1900 <= v <= 2035 or v % 1000 == 0):
                        out.add(v)
    return out


def parse_shell(cmd):
    """Executables, inline Python sources, saved-script runs and file writes of one shell command (never executes)."""
    info = {"py": False, "search": False, "rg_files": False, "sources": [], "script_runs": [], "writes": [],
            "parse_ok": True}
    try:
        body = base.shell_body(cmd) if WRAP.match(cmd) else cmd
        masked, heredocs = base.separate_heredocs(body)
        for seg in base.shell_segments(masked):
            prog, args = base.program_args(seg)
            words = list(seg)
            if not prog:
                continue
            if PY_EXE.fullmatch(prog):
                info["py"] = True
                if "-c" in args and args.index("-c") + 1 < len(args):
                    info["sources"].append(args[args.index("-c") + 1])
                else:
                    rest = [a for a in args if not a.startswith("-")]
                    if rest and SCRIPT_EXT.search(rest[0]):
                        info["script_runs"].append(rest[0])
            elif prog in ("rg", "grep", "egrep", "fgrep"):
                if "--files" in args:
                    info["rg_files"] = True  # a file listing, not a search: counted as "other shell"
                else:
                    info["search"] = True
            elif prog in ("bash", "sh", "source", ".") and args and SCRIPT_EXT.search(args[0]):
                info["script_runs"].append(args[0])
            elif words and SCRIPT_EXT.search(words[0]) and words[0].startswith(("./", "/")):
                info["script_runs"].append(words[0])
            for i, w in enumerate(words):
                if w in (">", ">>") and i + 1 < len(words) and not words[i + 1].startswith("/dev/"):
                    info["writes"].append(words[i + 1])
            if prog == "tee":
                info["writes"] += [a for a in args if not a.startswith("-")]
        for header, source in heredocs:
            segs = list(base.shell_segments(header))
            if any(PY_EXE.fullmatch(base.program_args(s)[0]) for s in segs):
                info["py"] = True
                info["sources"].append(source)
    except (ValueError, IndexError):
        info = {"py": bool(PY_RE.search(cmd)), "search": bool(SEARCH_RE.search(cmd)) and "--files" not in cmd,
                "rg_files": bool(SEARCH_RE.search(cmd)) and "--files" in cmd, "sources": [],
                "script_runs": RUN_SCRIPT_RE.findall(cmd), "writes": [], "parse_ok": False}
        if info["py"]:
            info["sources"] = [m.group(2) for m in HEREDOC_RE.finditer(cmd)] + [m.group(2) for m in PYC_RE.finditer(cmd)]
            if not info["sources"]:
                info["sources"] = [cmd]
    for src in info["sources"]:
        for m in PY_WRITE_RE.finditer(src):
            info["writes"].append(m.group(1) or m.group(2))
    return info


def graph_labels(src):
    try:
        return graph.classify_source(src)["labels"]
    except (SyntaxError, ValueError, RecursionError, TypeError, AttributeError, KeyError):
        return None


def program_types(src, cmd, glabels, carry_strict, kind, svc):
    """Multi-label program types (see report section 3). glabels: analyze_graph_behavior labels or None."""
    g = set(glabels or ())
    t = set()
    if kind == "tool_code":
        names = Counter(s["name"] for s in svc) if svc else Counter(KB_CALL_RE.findall(src))
        if names.get("search"):
            t.add("search_tool")
        if names.get("get_node"):
            t.add("id_lookup")
        if names.get("neighbors"):
            t.add("join_1hop")
        if svc and any(s.get("hop2") for s in svc):
            t.add("join_2hop")
    else:
        scan = bool(NODE_SCAN_RE.search(src)) and bool(NODE_RE.search(src))
        if scan:
            t.add("scan_filter")
        if STDIN_RE.search(src) and SEARCH_RE.search(cmd):
            t.add("pipe_filter")
        if SHARD_RE.search(src):
            t.add("id_lookup")
        if "edge_read" in g or (glabels is None and EDGE_RE.search(src)):
            if "chained_expansion" in g or (("adjacency_index" in g or "multiple_relations" in g) and "neighbor_collect" in g
                                            and "seed_loop" in g):
                t.add("join_2hop")
            else:
                t.add("join_1hop")
    if g & {"intersection", "difference"} or SETOP_RE.search(src):
        t.add("set_ops")
    if SORT_RE.search(src):
        t.add("rank")
    if carry_strict and (("id_lookup" in t) or ("join_1hop" in t) or ("join_2hop" in t) or VERIFY_LOOKUP_RE.search(src)):
        t.add("verify")
    if not t:
        t.add("other")
    return sorted(t)


# --------------------------------------------------------------------------- trace readers

def _text(output):
    return C._text(output)


def pi_trace(attempt, union):
    """Ordered calls with model-visible outputs, isError and call windows (ms) from one events.jsonl pass."""
    path = Path(attempt) / "artifacts/events.jsonl"
    if not path.exists():
        return None
    prc, is_err, t_start, t_end = [], {}, {}, {}
    with open(path, "rb") as f:
        for raw in f:
            head = raw[:80]
            if b'"provider_request_context"' in head:
                if union:
                    prc.append(raw)
                else:
                    prc = [raw]
            elif b'"tool_execution_end"' in head:
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                is_err[str(e.get("toolCallId", "")).split("|")[0]] = bool(e.get("isError"))
            elif b'"message_end"' in head:
                try:
                    m = json.loads(raw).get("message") or {}
                except ValueError:
                    continue
                if m.get("role") == "toolResult":
                    t_end[str(m.get("toolCallId", "")).split("|")[0]] = m.get("timestamp")
                elif m.get("role") == "assistant":
                    for b in m.get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "toolCall":
                            t_start[str(b.get("id", "")).split("|")[0]] = m.get("timestamp")
    if not prc:
        return None
    calls, outs, prompt = {}, {}, ""
    for raw in prc:
        items = json.loads(raw).get("payload", {}).get("input", [])
        for it in items:
            t = it.get("type")
            if t == "function_call":
                cid = str(it.get("call_id", "")).split("|")[0]
                if cid not in calls:
                    try:
                        args = json.loads(it.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    calls[cid] = {"name": it.get("name"), "args": args if isinstance(args, dict) else {}}
            elif t == "function_call_output":
                outs.setdefault(str(it.get("call_id", "")).split("|")[0], _text(it.get("output")))
            elif not prompt and it.get("role") == "user":
                prompt = _text(it.get("content"))
    out = []
    for cid, c in calls.items():
        a = c["args"]
        name = c["name"]
        cmd = a.get("command") if name == "bash" else (a.get("path") if name == "read" else json.dumps(a, sort_keys=True))
        out.append({"id": cid, "name": name, "cmd": cmd or "", "args": a, "out": outs.get(cid), "is_err": is_err.get(cid),
                    "t0": t_start.get(cid), "t1": t_end.get(cid), "turn": t_start.get(cid)})
    return {"calls": out, "prompt": prompt}


def codex_trace(attempt):
    """exec_command steps in invocation order, each with its cell's model-visible text (Codex harness)."""
    attempt = Path(attempt)
    traces = sorted(attempt.glob("codex_trace/*/trace.jsonl"))
    mto = attempt / "model_tool_outputs.jsonl"
    prompt = (attempt / "prompt.txt").read_text(errors="replace") if (attempt / "prompt.txt").exists() else ""
    if traces and mto.exists():
        cells, cell_of, req_order = {}, {}, {}
        with open(mto) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                req_order.setdefault(r.get("request_id"), len(req_order))
                if r["call_id"] in cells:
                    continue
                cells[r["call_id"]] = {"text": _text(r.get("output")), "notice": bool(r.get("truncation_notice")),
                                       "links": r.get("linked_runtime_call_ids") or [],
                                       "turn": req_order[r.get("request_id")]}
                for x in cells[r["call_id"]]["links"]:
                    cell_of[x] = r["call_id"]
        steps, ended, other = [], {}, Counter()
        tdir = traces[-1].parent
        with open(traces[-1]) as f:
            for line in f:
                try:
                    p = json.loads(line).get("payload") or {}
                except ValueError:
                    continue
                typ = p.get("type")
                if typ == "tool_call_started":
                    kind = (p.get("kind") or {}).get("type")
                    if kind != "exec_command":
                        other[kind or "unknown"] += 1
                        continue
                    cmd = ""
                    inv = (p.get("invocation_payload") or {}).get("path")
                    if inv and (tdir / inv).exists():
                        try:
                            pay = C.read_json(tdir / inv).get("payload") or {}
                            cmd = json.loads(pay.get("arguments") or "{}").get("cmd", "")
                        except ValueError:
                            cmd = ""
                    if not cmd:
                        try:
                            cmd = json.loads((p.get("summary") or {}).get("input_preview") or "{}").get("cmd", "")
                        except ValueError:
                            cmd = ""
                    steps.append({"id": p.get("tool_call_id"), "name": "bash", "cmd": cmd or "", "args": {}})
                elif typ == "tool_call_ended":
                    res = (p.get("result_payload") or {}).get("path")
                    ended[p.get("tool_call_id")] = res
        n_in_cell = Counter(cell_of.get(s["id"]) for s in steps)
        for s in steps:
            res = ended.get(s["id"])
            rt, code = None, None
            if res and (tdir / res).exists():
                try:
                    v = C.read_json(tdir / res).get("value") or {}
                    if isinstance(v, dict):
                        rt, code = v.get("output"), v.get("exit_code")
                except ValueError:
                    pass
            cell = cells.get(cell_of.get(s["id"]))
            if cell is not None and n_in_cell[cell_of.get(s["id"])] == 1:
                s["out"] = cell["text"]
            else:
                s["out"] = rt if isinstance(rt, str) else (cell["text"] if cell else None)
            s["is_err"] = (code not in (None, 0)) if code is not None else None
            s["cell"] = cell_of.get(s["id"])
            # Turn = the request in which the cell's output first reached the model; calls of one turn were
            # issued before any of their outputs was visible.
            s["turn"] = cell["turn"] if cell else None
        return {"calls": steps, "prompt": prompt, "visibility": "recorded", "n_cells": len(cells),
                "multi_exec_cells": sum(1 for c in cells.values() if len(c["links"]) > 1), "other_runtime": dict(other)}
    cmds = C.codex_commands(attempt)
    if cmds is None:
        return None
    steps = [{"id": str(i), "name": "bash", "cmd": c["command"], "args": {}, "out": c["output"],
              "is_err": (c["exit_code"] not in (None, 0)) if c["completed"] else None} for i, c in enumerate(cmds)]
    return {"calls": steps, "prompt": prompt, "visibility": "raw", "n_cells": None, "multi_exec_cells": None,
            "other_runtime": {}}


def service_requests(attempt):
    path = Path(attempt) / "tool_requests.jsonl"
    if not path.exists():
        return None
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("phase") != "agent":
                continue
            req = r.get("request") or {}
            text = (r.get("response") or {}).get("text") or ""
            args = req.get("arguments") or {}
            m = C.FOUND.search(text)
            k = args.get("k", 20 if req.get("name") == "neighbors" else 5)
            out.append({"t": (r.get("time") or 0) * 1000.0, "name": req.get("name"), "node_id": args.get("node_id"),
                        "capped": bool(req.get("name") == "neighbors" and m and k and int(m[1]) > int(k)),
                        # result ids other than the seed itself (a self-loop result would make a second
                        # neighbors call on the same node look like a two-hop chain)
                        "ids": sorted({int(x) for x in NB_RESULT_RE.findall(text)} - {args.get("node_id")})[:400]})
    return out


def scratch_files(attempt):
    """Files the agent left in the attempt's scratch directory (harness full_outputs/ excluded)."""
    tmp = Path(attempt) / "tmp"
    files = []
    if tmp.is_dir():
        for dirpath, dirnames, filenames in os.walk(tmp):
            if Path(dirpath) == tmp and "full_outputs" in dirnames:
                dirnames.remove("full_outputs")
            files += [os.path.relpath(os.path.join(dirpath, n), tmp) for n in filenames]
    return files


# --------------------------------------------------------------------------- per-query extraction

def score(result, split="test"):
    ds, qid = result.get("dataset"), str(result.get("query_id"))
    g = C.gold(ds, split).get(qid, [])
    try:
        pred = [int(x) for x in json.loads(result.get("prediction") or "{}").get("node_ids", [])]
    except (ValueError, TypeError, AttributeError):
        pred = []
    pred = list(dict.fromkeys(pred))
    gs = set(g)
    rank = next((i + 1 for i, p in enumerate(pred) if p in gs), None)
    scored = result.get("status") == "scored"
    return g, pred, {"hit1": float(scored and rank == 1), "r20": (len(gs & set(pred[:20])) / len(gs) if scored and gs else 0.0),
                     "mrr": (1.0 / rank if scored and rank and rank <= 100 else 0.0), "gold_rank": rank,
                     "top1": pred[0] if pred else None, "n_pred": len(pred)}


def step_kind(name, info):
    if name == "read":
        return "read"
    if name in ("search", "get_node", "neighbors", "get_schema"):
        return "tool"
    if name != "bash":
        return "other"
    if info["py"]:
        return "py"
    if info["search"]:
        return "rg"
    return "sh"


def extract(job):
    cfg_kind, harness, result, src = job
    ds, qid = result.get("dataset"), str(result.get("query_id"))
    g, pred, sc = score(result)
    rec = {"src": src, "dataset": ds, "query_id": qid, "status": result.get("status"),
           "finished_at": result.get("finished_at"), "version": VERSION, "extract_error": None, **sc,
           "visibility": "missing", "steps": [], "progs": [], "saved": {}, "svc": {}}
    try:
        _extract(rec, cfg_kind, harness, result, g, pred)
    except Exception as e:  # noqa: BLE001  (one bad trace never stops the analysis)
        rec["extract_error"] = repr(e)[:300]
    return rec


def _extract(rec, cfg_kind, harness, result, g, pred):
    attempt = result.get("attempt_path") or result.get("attempt")
    if not attempt or not Path(attempt).exists():
        return
    if harness == "pi":
        tr = pi_trace(attempt, union=bool(result.get("compaction_starts")))
        vis = "recorded"
    else:
        tr = codex_trace(attempt)
        vis = tr["visibility"] if tr else "missing"
    if tr is None:
        return
    rec["visibility"] = vis
    if harness == "codex":
        rec["codex"] = {"n_cells": tr.get("n_cells"), "multi_exec_cells": tr.get("multi_exec_cells"),
                        "other_runtime": tr.get("other_runtime")}
    prompt = tr["prompt"] or ""
    if not prompt and (Path(attempt) / "prompt.txt").exists():
        prompt = (Path(attempt) / "prompt.txt").read_text(errors="replace")
    exclude = set(re.findall(r"\d+", prompt))
    grx, rrx = C.id_regex(g), C.record_regex(g)
    top1 = pred[0] if pred else None
    t1rx = C.id_regex([top1]) if top1 is not None else None
    svc = service_requests(attempt) if cfg_kind in ("tool", "tool_code") else None
    calls = tr["calls"]
    turn_ids, tu = {}, []
    for i, c in enumerate(calls):
        key = c.get("turn") if c.get("turn") is not None else ("solo", i)
        turn_ids.setdefault(key, len(turn_ids))
        tu.append(turn_ids[key])
    # Map graph-service requests to the call whose window contains them (Graph tools + code).
    svc_of = defaultdict(list)
    unassigned = 0
    if svc and cfg_kind == "tool_code":
        wins = [(c.get("t0"), c.get("t1"), i) for i, c in enumerate(calls) if c.get("t0") and c.get("t1")]
        wins.sort(key=lambda w: w[1])
        for s in svc:
            hit = next((i for t0, t1, i in wins if t0 - 5 <= s["t"] <= t1 + 50), None)
            if hit is None:
                unassigned += 1
            else:
                svc_of[hit].append(s)
    # "Earlier" means an earlier turn: the calls of one model response were issued before any of their outputs was
    # visible, so ids, the top-1 answer and written files of the current turn are merged in only when the turn changes.
    seen_loose, seen_strict = set(), set()
    pend_loose, pend_strict = set(), set()
    progs, steps = [], []
    written, saved = [], {"writes": 0, "script_writes": 0, "script_runs": 0, "run_written_script": 0,
                          "read_written": 0, "defs": 0, "local_exec": 0}
    written_prev, written_pend = [], []
    top1_seen, top1_pend = False, False
    cur_turn = None
    for i, c in enumerate(calls):
        if tu[i] != cur_turn:
            seen_loose |= pend_loose
            seen_strict |= pend_strict
            pend_loose, pend_strict = set(), set()
            top1_seen = top1_seen or top1_pend
            written_prev += written_pend
            written_pend = []
            cur_turn = tu[i]
        name, cmd = c["name"], c["cmd"] or ""
        text = c.get("out")
        visible = text is not None
        text = text or ""
        info = parse_shell(cmd) if name == "bash" else {"py": False, "search": False, "rg_files": False, "sources": [],
                                                        "script_runs": [], "writes": [], "parse_ok": True}
        k = step_kind(name, info)
        m6 = H6_MARK.search(text)
        trunc = bool(m6) or bool(CODEX_MARK.search(text))
        exitm = EXIT_RE.search(text[-400:]) if text else None
        err = bool(c.get("is_err")) or TRACEBACK in text or bool(exitm and exitm.group(1) != "0") or bool(PY_ERR_RE.search(text[-3000:]))
        if k == "rg" and not TRACEBACK in text and exitm and exitm.group(1) == "1" and len(text.strip()) < 80:
            err = False  # rg/grep "no match" exit code 1 is a result, not an error
        # Interpreter not found: the program did not run; not a program failure.
        notrun = bool(k == "py" and TRACEBACK not in text and PY_NOTFOUND_RE.search(text[-3000:]))
        if notrun:
            err = False
        etype = None
        if err and k == "py":
            em = ERR_NAME_RE.findall(text[-4000:])
            etype = em[-1] if em else ("timeout" if TIMEOUT_RE.search(text[-600:]) else
                                       "command not found" if exitm and exitm.group(1) == "127" else "non-zero exit")
        lits_loose = ints_loose(cmd, exclude)
        carry_l = bool(lits_loose & seen_loose)
        carry_s = bool(lits_loose & seen_strict)
        t1_cmd = bool(t1rx and t1rx.search(cmd)) or (name == "neighbors" and top1 is not None and c["args"].get("node_id") == top1) \
            or (name == "get_node" and top1 is not None and c["args"].get("node_id") == top1)
        edges = bool(EDGE_RE.search(cmd)) or name == "neighbors" or (k == "read" and "/edges/" in cmd)
        mysvc = svc_of.get(i, [])
        if cfg_kind == "tool_code" and any(s["name"] == "neighbors" for s in mysvc):
            edges = True
        t1_out = bool(t1rx and visible and t1rx.search(text))
        shell = k in ("rg", "sh")
        st = {"k": k, "tu": tu[i], "n": name if k in ("tool", "other") else None, "tr": trunc, "e": err, "nr": notrun,
              "ob": len(text.encode()), "vis": visible, "rf": bool(k == "sh" and info.get("rg_files")),
              "proj": bool(shell and SH_PROJ_RE.search(cmd)),
              "lim": bool(shell and LIMIT_RE.search(cmd)), "fo": bool(FULLOUT_RE.search(cmd)), "ed": edges,
              "nd": bool(NODE_RE.search(cmd)), "t1c": t1_cmd, "t1o": t1_out,
              "gi": bool(grx and visible and grx.search(text)), "gr": bool(rrx and visible and rrx.search(text)),
              "cl": carry_l, "cs": carry_s, "sh": bool(SHARD_RE.search(cmd)), "pp": bool(k == "py" and SEARCH_RE.search(cmd)
                                                                                           and re.search(r"\|\s*python", cmd))}
        top1_pend = top1_pend or t1_out
        st["t1v"] = bool(t1_cmd and top1_seen)  # names the top-1 after an earlier turn's output showed it
        if cfg_kind == "tool_code":
            st["svc"] = len(mysvc)
        # saved files / tools
        for w in info["writes"]:
            written.append((i, w))
            written_pend.append(w)
        if info["writes"]:
            saved["writes"] += 1
            if any(SCRIPT_EXT.search(w) for w in info["writes"]):
                saved["script_writes"] += 1
        runs = info["script_runs"]
        if runs:
            saved["script_runs"] += 1
            # lenient on purpose: any earlier call, same turn included (a zero count is then the stronger statement)
            if any(os.path.basename(r) == os.path.basename(w) for r in runs for j, w in written if j < i):
                saved["run_written_script"] += 1
        earlier = [w for w in written_prev if len(os.path.basename(w)) >= 4]  # written in an earlier turn
        if earlier and any(os.path.basename(w) in cmd for w in earlier):
            saved["read_written"] += 1
        srcs = info["sources"]
        if k == "py":
            code = "\n".join(srcs) if srcs else cmd
            if DEF_RE.search(code):
                saved["defs"] += 1
            if PY_IMPORT_LOCAL_RE.search(code):
                saved["local_exec"] += 1
            stripped = strip_boilerplate(code)
            best = (0.0, None)
            best_s = (0.0, None)
            for j, p in enumerate(progs):
                s = similarity(code, p["_code"])
                if s > best[0]:
                    best = (s, j)
                s2 = similarity(stripped, p["_strip"])
                if s2 > best_s[0]:
                    best_s = (s2, j)
            gl = None
            for s in srcs[:3]:
                lab = graph_labels(s)
                if lab is not None:
                    gl = sorted(set(gl or []) | set(lab))
            # Graph tools + code: dynamic two-hop = a neighbors request on an id returned by an earlier neighbors
            # request of the same program.
            prev_ids = set()
            svc_rec = []
            for s in mysvc:
                hop2 = s["name"] == "neighbors" and s.get("node_id") in prev_ids
                svc_rec.append({"name": s["name"], "hop2": hop2})
                if s["name"] == "neighbors":
                    prev_ids |= set(s["ids"])
            types = program_types(code, cmd, gl, carry_s, cfg_kind, svc_rec)
            progs.append({"i": i, "_code": code, "_strip": stripped, "len": len(code), "lines": code.count("\n") + 1,
                          "sim": round(best[0], 3), "par": best[1], "sims": round(best_s[0], 3), "pars": best_s[1], "tu": tu[i],
                          "pseen": best[1] is not None and progs[best[1]]["tu"] < tu[i],
                          "pseens": best_s[1] is not None and progs[best_s[1]]["tu"] < tu[i],
                          "types": types, "glab": gl or [], "scan": "scan_filter" in types, "e": err, "nr": notrun, "tr": trunc,
                          "gi": st["gi"], "gr": st["gr"], "cl": carry_l, "cs": carry_s, "ed": edges,
                          "svc": len(mysvc), "svc_n": dict(Counter(s["name"] for s in mysvc)),
                          "svc_h2": sum(1 for s in svc_rec if s["hop2"]),
                          "svc_cap": sum(1 for s in mysvc if s["capped"]), "nsrc": len(srcs), "parse_ok": info["parse_ok"],
                          "et": etype})
        steps.append(st)
        if visible:
            pend_loose |= ints_loose(text[:300000], exclude)
            pend_strict |= ints_strict(text[:300000])
    # Every program: the next Python program issued in a later turn (an agent may first inspect, e.g. `ls edges`,
    # then repair). For failed programs this gives the repair rate; for programs that ran, the comparison point.
    for p in progs:
        q = next((q for q in progs if q["tu"] > p["tu"]), None)
        p["rp"] = None if q is None else {"sim": round(similarity(q["_code"], p["_code"]), 3), "e": bool(q["e"]),
                                          "nr": bool(q["nr"]), "gap": q["tu"] - p["tu"]}
    # Error -> repair within the next turn (the first calls issued after the error was visible). Repair = a Python
    # program of that turn similar (>= 0.5) to the failed one; its outcome is that of the most similar one.
    for p in progs:
        if not p["e"]:
            continue
        nt = next((steps[j]["tu"] for j in range(p["i"] + 1, len(steps)) if steps[j]["tu"] > p["tu"]), None)
        if nt is None:
            p["rx"] = None
            continue
        cands = [q for q in progs if q["tu"] == nt]
        best = max(((similarity(q["_code"], p["_code"]), q) for q in cands), key=lambda x: x[0], default=(0.0, None))
        p["rx"] = {"py": bool(cands), "sim": round(best[0], 3), "e": bool(best[1]["e"]) if best[1] else None,
                   "nr": bool(best[1]["nr"]) if best[1] else None,
                   "kinds": sorted({steps[j]["k"] for j in range(p["i"] + 1, len(steps)) if steps[j]["tu"] == nt})}
    for p in progs:
        p.pop("_code")
        p.pop("_strip")
    rec["steps"] = steps
    rec["progs"] = progs
    files = scratch_files(attempt)
    saved.update({"scratch_files": len(files), "scratch_scripts": sum(1 for f in files if SCRIPT_EXT.search(f)),
                  "scratch_names": files[:10]})
    rec["saved"] = saved
    if svc is not None:
        rec["svc"] = {"total": len(svc), "by_name": dict(Counter(s["name"] for s in svc)),
                      "neighbors": sum(1 for s in svc if s["name"] == "neighbors"),
                      "capped": sum(1 for s in svc if s["capped"]), "unassigned": unassigned}


def _work(job):
    return extract(job)


# Cache rows store only non-default step/program fields (keeps the cache small); expand() restores the defaults.
STEP_DEFAULTS = {"n": None, "tr": False, "e": False, "nr": False, "rf": False, "ob": 0, "vis": False, "proj": False,
                 "lim": False, "fo": False,
                 "ed": False, "nd": False, "t1c": False, "t1o": False, "gi": False, "gr": False, "cl": False, "cs": False,
                 "sh": False, "pp": False, "t1v": False}
PROG_DEFAULTS = {"sim": 0.0, "par": None, "sims": 0.0, "pars": None, "pseen": False, "pseens": False, "types": [], "glab": [],
                 "scan": False, "e": False, "nr": False, "tr": False, "gi": False, "gr": False, "cl": False, "cs": False,
                 "ed": False,
                 "svc": 0, "svc_n": {}, "svc_h2": 0, "svc_cap": 0, "nsrc": 1, "parse_ok": True, "et": None}


def compact(rec):
    out = dict(rec)
    out["steps"] = [{k: v for k, v in st.items() if k not in STEP_DEFAULTS or v != STEP_DEFAULTS[k]} for st in rec["steps"]]
    out["progs"] = [{k: v for k, v in p.items() if k not in PROG_DEFAULTS or v != PROG_DEFAULTS[k]} for p in rec["progs"]]
    return out


def expand(rec):
    rec["steps"] = [{**STEP_DEFAULTS, **st} for st in rec.get("steps") or []]
    rec["progs"] = [{**PROG_DEFAULTS, **p} for p in rec.get("progs") or []]
    return rec


def load(cfg, use_cache=True, processes=None):
    """Per-query records (terminal results only). Cache per source: reused when path, status, finish time and
    VERSION are unchanged (plain value comparison)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cfg['source']}.jsonl"
    cached = {}
    if path.exists():
        for r in C.read_jsonl(path):
            cached[r["src"]] = expand(r)
    # Configurations that read the same trajectories share one cache file (e.g. Full DKGI on the test subset and the
    # full-test run); with --no-cache their rows stay in the file, but none is reused.
    reusable = cached if use_cache else {}
    rows, todo = [], []
    for result, src in iter_results(cfg):
        c = reusable.get(src)
        if c is not None and c.get("status") == result.get("status") and c.get("finished_at") == result.get("finished_at") \
                and c.get("version") == VERSION:
            rows.append(c)
        else:
            todo.append((cfg["kind"], cfg["harness"], result, src))
    if todo:
        t0 = time.time()
        if len(todo) > 16:
            from multiprocessing import Pool
            with Pool(processes or min(48, os.cpu_count() or 4)) as pool:
                fresh = list(pool.imap_unordered(_work, todo, chunksize=4))
        else:
            fresh = [_work(t) for t in todo]
        print(f"  {cfg['name']}: extracted {len(fresh)} trajectories in {time.time() - t0:.0f}s", flush=True)
        rows += fresh
        cached.update({r["src"]: r for r in fresh})
        tmp = path.with_suffix(".jsonl.part")
        with open(tmp, "w") as f:
            for r in cached.values():
                f.write(json.dumps(compact(r), separators=(",", ":")) + "\n")
        os.replace(tmp, path)
    rows.sort(key=lambda r: (r["dataset"], int(r["query_id"]) if r["query_id"].isdigit() else 0))
    return rows


# --------------------------------------------------------------------------- aggregation helpers

KINDS = ("py", "rg", "read", "sh", "tool", "other")
KIND_LABEL = {"py": "Python", "rg": "rg/grep", "read": "read tool", "sh": "other shell", "tool": "graph-tool call",
              "other": "other"}
TYPES = ("scan_filter", "pipe_filter", "id_lookup", "join_1hop", "join_2hop", "set_ops", "rank", "verify", "search_tool",
         "other")
TYPE_LABEL = {"scan_filter": "Filter/projection over a full node scan", "pipe_filter": "Filter/projection of piped rg output",
              "id_lookup": "Lookup by id (shard arithmetic; get_node)", "join_1hop": "1-hop join (edge file; neighbors)",
              "join_2hop": "2-hop or longer join", "set_ops": "Set intersection / difference",
              "rank": "Scoring / ranking (sort, counts, scores)", "verify": "Targeted check of ids from earlier outputs",
              "search_tool": "search tool calls (Graph tools + code only)", "other": "None of the above"}


def frac(k, n):
    return k / n if n else None


def P(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def F(x, d=1):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    return C.median(list(xs))


def quantile(xs, q):
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(math.floor(q * (len(v) - 1) + 0.5)))]


def ok_rows(rows):
    return [r for r in rows if not r.get("extract_error") and r.get("visibility") != "missing"]


def families(progs, tau, strip=False):
    ks, kp = ("sims", "pars") if strip else ("sim", "par")
    fam = []
    for p in progs:
        if p.get(ks, 0) >= tau and p.get(kp) is not None:
            fam.append(fam[p[kp]])
        else:
            fam.append(max(fam) + 1 if fam else 0)
    return fam


def pos_bin(i, n):
    return min(4, int(5 * i / n)) if n else 0


# --------------------------------------------------------------------------- 0. coverage

def coverage(cfg, rows):
    out = {"name": cfg["name"], "paper": cfg["paper"], "expected": cfg["expected"], "n": len(rows),
           "n_ok": len(ok_rows(rows)), "extract_errors": sum(bool(r.get("extract_error")) for r in rows),
           "visibility": dict(Counter(r.get("visibility") for r in rows)), "note": cfg.get("note", ""), "by_ds": {}}
    for ds in C.DATASETS:
        sel = [r for r in rows if r["dataset"] == ds]
        out["by_ds"][ds] = {"n": len(sel), "h1": mean([r["hit1"] for r in sel]), "failed": sum(r["status"] != "scored" for r in sel)}
    out["h1"] = mean([r["hit1"] for r in rows])
    out["h1_ds_mean"] = None
    return out


# --------------------------------------------------------------------------- 1. lineages

def lineage_stats(rows, tau, strip=False):
    rows = ok_rows(rows)
    pyq = [r for r in rows if r["progs"]]
    progs = [p for r in rows for p in r["progs"]]
    ks = "sims" if strip else "sim"
    lin_per_q, sizes, maxlen = [], [], []
    fl = Counter()
    n_multi = 0
    for r in pyq:
        fam = families(r["progs"], tau, strip)
        cnt = Counter(fam)
        lin_per_q.append(len(cnt))
        sizes += list(cnt.values())
        maxlen.append(max(cnt.values()))
        for f, c in cnt.items():
            if c < 2:
                continue
            n_multi += 1
            idx = [j for j, x in enumerate(fam) if x == f]
            a, b = r["progs"][idx[0]], r["progs"][idx[-1]]
            for key in ("tr", "e", "gi", "gr"):
                fl[f"first_{key}"] += bool(a.get(key))
                fl[f"last_{key}"] += bool(b.get(key))
    rev = sum(p.get(ks, 0) >= tau for p in progs)
    not_raw = sum(p.get(ks, 0) >= tau and p.get("sim", 0) < tau for p in progs) if strip else 0
    rev_seen = sum(p.get(ks, 0) >= tau and p.get("pseens" if strip else "pseen", False) for p in progs)
    return {"links": rev, "links_not_raw": not_raw, "revisions_parent_seen": frac(rev_seen, rev), "tau": tau, "strip": strip, "n_queries": len(rows), "n_queries_py": len(pyq), "n_programs": len(progs),
            "programs_per_query": frac(len(progs), len(rows)), "programs_per_py_query_median": median(len(r["progs"]) for r in pyq),
            "revision_share": frac(rev, len(progs)), "queries_with_revision": frac(sum(any(p.get(ks, 0) >= tau for p in r["progs"]) for r in pyq), len(pyq)),
            "lineages_per_py_query_mean": mean(lin_per_q), "lineages_per_py_query_median": median(lin_per_q),
            "versions_per_lineage_mean": mean(sizes), "versions_per_lineage_median": median(sizes), "n_lineages": len(sizes),
            "multi_version_lineages": n_multi, "multi_version_share": frac(n_multi, len(sizes)),
            "longest_lineage_median": median(maxlen),
            "first_to_last": {k: frac(v, n_multi) for k, v in sorted(fl.items())}}


def reuse_repair(rows):
    rows = ok_rows(rows)
    pyq = [r for r in rows if r["progs"]]
    progs = [p for r in rows for p in r["progs"]]
    n_err = n_last = n_next = n_next_py = n_rep = n_rep_ok = 0
    n_np = n_np_rep = n_np_ok = 0
    n_ok_next = n_ok_next_sim = n_ok_next_sim8 = n_np_rep8 = 0
    n_nr = sum(bool(p.get("nr")) for p in progs)
    etypes, nkinds, gaps = Counter(), Counter(), Counter()
    for r in pyq:
        for p in r["progs"]:
            rp = p.get("rp")
            if not p.get("e"):
                # comparison point: a program that ran without failing, followed by a later program
                if not p.get("nr") and rp is not None:
                    n_ok_next += 1
                    n_ok_next_sim += rp["sim"] >= 0.5
                    n_ok_next_sim8 += rp["sim"] >= 0.8
                continue
            n_err += 1
            etypes[p.get("et") or "unknown"] += 1
            if rp is not None:
                n_np += 1
                n_np_rep8 += rp["sim"] >= 0.8
                if rp["sim"] >= 0.5:
                    n_np_rep += 1
                    n_np_ok += not rp["e"] and not rp.get("nr")
                    gaps[min(rp["gap"], 3)] += 1
            rx = p.get("rx")
            if rx is None:
                n_last += 1
                continue
            n_next += 1
            n_next_py += bool(rx["py"])
            nkinds["Python" if rx["py"] else "+".join(rx.get("kinds") or []) or "none"] += 1
            if rx["py"] and rx["sim"] >= 0.5:
                n_rep += 1
                n_rep_ok += not rx["e"] and not rx.get("nr")
    n_ran = len(progs) - n_nr
    return {"n_programs": len(progs), "n_queries_py": len(pyq), "not_run_programs": n_nr,
            "not_run_share": frac(n_nr, len(progs)), "n_programs_ran": n_ran,
            "ok_with_later_program": n_ok_next, "ok_nextprog_similar": frac(n_ok_next_sim, n_ok_next),
            "ok_nextprog_similar_0.8": frac(n_ok_next_sim8, n_ok_next),
            "err_nextprog_similar_0.8": frac(n_np_rep8, n_np),
            "carry_loose_programs": frac(sum(p.get("cl", False) for p in progs), len(progs)),
            "carry_strict_programs": frac(sum(p.get("cs", False) for p in progs), len(progs)),
            "carry_loose_queries": frac(sum(any(p.get("cl") for p in r["progs"]) for r in pyq), len(pyq)),
            "carry_strict_queries": frac(sum(any(p.get("cs") for p in r["progs"]) for r in pyq), len(pyq)),
            "error_programs": n_err, "error_share": frac(n_err, n_ran), "error_at_last_turn": n_last,
            "error_with_next_turn": n_next, "err_next_python": frac(n_next_py, n_next),
            "err_next_repair": frac(n_rep, n_next), "err_next_repair_ok": frac(n_rep_ok, n_next),
            "error_with_later_program": n_np, "err_nextprog_repair": frac(n_np_rep, n_np),
            "err_nextprog_repair_ok": frac(n_np_ok, n_np),
            "err_nextprog_repair_turn_gap": {("3+" if k == 3 else str(k)): v for k, v in sorted(gaps.items())},
            "error_types": dict(etypes.most_common(12)), "err_next_turn_kinds": dict(nkinds.most_common(8)),
            "scan_share": frac(sum(p.get("scan", False) for p in progs), len(progs)),
            "edge_programs": frac(sum(p.get("ed", False) for p in progs), len(progs)),
            "pipe_programs": frac(sum("pipe_filter" in p.get("types", []) for p in progs), len(progs)),
            "lines_median": median(p["lines"] for p in progs), "chars_median": median(p["len"] for p in progs)}


# --------------------------------------------------------------------------- 2. saved tools

def saved_stats(rows):
    rows = ok_rows(rows)
    n = len(rows)
    s = [r.get("saved") or {} for r in rows]
    q = lambda key: sum(bool(x.get(key)) for x in s)  # noqa: E731
    return {"n": n, "writes_any_file": q("writes"), "writes_script_file": q("script_writes"), "runs_script_file": q("script_runs"),
            "runs_script_written_earlier": q("run_written_script"), "reads_back_written_file": q("read_written"),
            "defines_function": q("defs"), "exec_or_runpy": q("local_exec"), "scratch_any_file": q("scratch_files"),
            "scratch_script_file": q("scratch_scripts"),
            "examples_scratch": [(r["dataset"], r["query_id"], (r.get("saved") or {}).get("scratch_names", [])[:3])
                                 for r in rows if (r.get("saved") or {}).get("scratch_scripts")][:10]}


# --------------------------------------------------------------------------- 3. program types

def type_stats(rows):
    rows = ok_rows(rows)
    pyq = [r for r in rows if r["progs"]]
    progs = [p for r in rows for p in r["progs"]]
    out = {"n_programs": len(progs), "n_queries_py": len(pyq), "programs": {}, "queries": {}}
    for t in TYPES:
        out["programs"][t] = frac(sum(t in p.get("types", []) for p in progs), len(progs))
        out["queries"][t] = frac(sum(any(t in p.get("types", []) for p in r["progs"]) for r in pyq), len(pyq))
    return out


# --------------------------------------------------------------------------- 4. phases

def phase_stats(rows):
    rows = ok_rows(rows)
    bins = [Counter() for _ in range(5)]
    for r in rows:
        n = len(r["steps"])
        for i, s in enumerate(r["steps"]):
            bins[pos_bin(i, n)][s["k"]] += 1
    return {"n_queries": len(rows), "n_steps": [sum(b.values()) for b in bins],
            "share": [{k: frac(b[k], sum(b.values())) for k in KINDS} for b in bins]}


# --------------------------------------------------------------------------- 5. harness feedback

NEXT_KEYS = ("py", "proj", "lim", "fo")
CTRL_BYTES = (4000, 9000)  # large-output controls: uncapped outputs of at least this many bytes


def feedback_stats(rows, prev_kind="rg", min_cell=20, min_unc_bytes=0, rg_files=False):
    """After a non-error step of prev_kind: next-step rates, capped vs not, within position quintiles of that step.
    min_unc_bytes > 0 keeps only uncapped outputs of at least that size (capped outputs are all above 18 KB);
    rg_files=True also counts `rg --files` listings as rg calls (sensitivity check)."""
    rows = ok_rows(rows)
    cells = defaultdict(Counter)
    cap_bytes = []
    for r in rows:
        s = r["steps"]
        n = len(s)
        for i in range(n - 1):
            a = s[i]
            if not (a["k"] == prev_kind or (rg_files and prev_kind == "rg" and a.get("rf"))):
                continue
            if a["e"] or a.get("nr") or not a.get("vis"):
                continue
            if not a["tr"] and a.get("ob", 0) < min_unc_bytes:
                continue
            # next turn = the calls issued in the following model response (after this output was visible)
            nt = next((s[j].get("tu", j) for j in range(i + 1, n) if s[j].get("tu", j) > a.get("tu", i)), None)
            if nt is None:
                continue
            nb = [b for b in s[i + 1:] if b.get("tu") == nt]
            c = cells[(pos_bin(i, n), bool(a["tr"]))]
            c["n"] += 1
            c["py"] += any(b["k"] == "py" for b in nb)
            c["proj"] += any(b.get("proj") for b in nb)
            c["lim"] += any(b.get("lim") for b in nb)
            c["fo"] += any(b.get("fo") for b in nb)
            if a["tr"]:
                cap_bytes.append(a.get("ob", 0))
    out = {"prev_kind": prev_kind, "min_uncapped_bytes": min_unc_bytes, "rg_files": rg_files, "bins": [], "raw": {},
           "weighted": {}, "n_capped": 0, "n_uncapped": 0,
           "capped_visible_bytes_min": min(cap_bytes) if cap_bytes else None,
           "capped_visible_bytes_median": median(cap_bytes) if cap_bytes else None}
    for cap in (True, False):
        tot = Counter()
        for b in range(5):
            tot.update(cells[(b, cap)])
        out["raw"]["capped" if cap else "uncapped"] = {"n": tot["n"], **{k: frac(tot[k], tot["n"]) for k in NEXT_KEYS}}
        out["n_capped" if cap else "n_uncapped"] = tot["n"]
    w = 0
    acc = Counter()
    for b in range(5):
        t, u = cells[(b, True)], cells[(b, False)]
        row = {"bin": b + 1, "n_capped": t["n"], "n_uncapped": u["n"]}
        for k in NEXT_KEYS:
            row[f"{k}_capped"] = frac(t[k], t["n"])
            row[f"{k}_uncapped"] = frac(u[k], u["n"])
        row["used"] = t["n"] >= min_cell and u["n"] >= min_cell
        out["bins"].append(row)
        if row["used"]:
            w += t["n"]
            for k in NEXT_KEYS:
                acc[f"{k}_c"] += t["n"] * row[f"{k}_capped"]
                acc[f"{k}_u"] += t["n"] * row[f"{k}_uncapped"]
    for k in NEXT_KEYS:
        out["weighted"][k] = {"capped": acc[f"{k}_c"] / w if w else None, "uncapped_matched": acc[f"{k}_u"] / w if w else None,
                              "diff": (acc[f"{k}_c"] - acc[f"{k}_u"]) / w if w else None}
    out["weight_n"] = w
    return out


# --------------------------------------------------------------------------- 6. tool-mediated variants

def tool_stats(rows, kind):
    rows = ok_rows(rows)
    n = len(rows)
    out = {"n": n}
    svc = [r.get("svc") or {} for r in rows]
    out["service_calls_mean"] = mean([s.get("total", 0) for s in svc])
    out["service_calls_median"] = median([s.get("total", 0) for s in svc])
    by = Counter()
    for s in svc:
        by.update(s.get("by_name") or {})
    out["service_by_name_per_query"] = {k: v / n for k, v in sorted(by.items())} if n else {}
    nb = sum(s.get("neighbors", 0) for s in svc)
    out["neighbors_calls"] = nb
    out["neighbors_cap_rate"] = frac(sum(s.get("capped", 0) for s in svc), nb)
    out["queries_with_cap_hit"] = frac(sum(s.get("capped", 0) > 0 for s in svc), n)
    out["calls_per_query_mean"] = mean([len(r["steps"]) for r in rows])
    out["calls_per_query_median"] = median([len(r["steps"]) for r in rows])
    if kind == "tool_code":
        progs = [p for r in rows for p in r["progs"]]
        per = [p.get("svc", 0) for p in progs]
        out.update({"programs_per_query_mean": frac(len(progs), n), "programs_per_query_median": median(len(r["progs"]) for r in rows),
                    "service_per_program_mean": mean(per), "service_per_program_median": median(per),
                    "service_per_program_p90": quantile(per, 0.9), "service_per_program_max": max(per) if per else None,
                    "programs_with_2plus_calls": frac(sum(x >= 2 for x in per), len(per)),
                    "programs_with_2plus_tools": frac(sum(len(p.get("svc_n") or {}) >= 2 for p in progs), len(progs)),
                    "programs_with_dynamic_2hop": frac(sum(p.get("svc_h2", 0) > 0 for p in progs), len(progs)),
                    "programs_with_cap_hit": frac(sum(p.get("svc_cap", 0) > 0 for p in progs), len(progs)),
                    "unassigned_service_calls": sum(s.get("unassigned", 0) for s in svc),
                    "assigned_share": frac(sum(per), sum(s.get("total", 0) for s in svc))})
    return out


# --------------------------------------------------------------------------- 7. Luna vs Astra, same questions

def query_feats(r):
    st, ps = r["steps"], r["progs"]
    n = len(st)
    return {"steps": n, "programs": len(ps), "py_share": frac(sum(s["k"] == "py" for s in st), n),
            "edges_any": any(s.get("ed") for s in st), "py_edges": any(p.get("ed") for p in ps),
            "top1_named_after_seen": any(s.get("t1v") for s in st),
            "top1_edge_check": any(s.get("t1c") and s.get("ed") for s in st),
            "fullout_reuse": any(s.get("fo") for s in st), "pipe_rg_python": any(s.get("pp") for s in st),
            "scan_share": frac(sum(p.get("scan", False) for p in ps), len(ps)) if ps else None,
            "shard_lookup": any(s.get("sh") for s in st), "carry": any(p.get("cs") for p in ps),
            "revision": any(p.get("sim", 0) >= 0.5 for p in ps), "py_error": any(p.get("e") for p in ps),
            "lineages3": len(set(families(ps, 0.5))) >= 3 if ps else False, "any_scan": any(p.get("scan") for p in ps),
            "capped_steps": sum(bool(s.get("tr")) for s in st)}


PAIR_FEATS = (("steps", "tool calls per query (mean)", "num"), ("programs", "Python programs per query (mean)", "num"),
              ("py_share", "Python share of calls", "pct"), ("edges_any", "any call touches edge files", "pct"),
              ("py_edges", "a Python program reads edges", "pct"),
              ("top1_named_after_seen", "final top-1 named after an earlier turn's output showed it", "pct"),
              ("top1_edge_check", "edge read naming the final top-1", "pct"),
              ("fullout_reuse", "re-reads a saved full output", "pct"), ("pipe_rg_python", "rg piped into Python", "pct"),
              ("shard_lookup", "shard-targeted id lookup", "pct"), ("scan_share", "full-scan share of programs", "pct"),
              ("carry", "a program embeds ids from earlier outputs", "pct"))


def paired_stats(L, A):
    Lm = {(r["dataset"], r["query_id"]): r for r in ok_rows(L)}
    Am = {(r["dataset"], r["query_id"]): r for r in ok_rows(A)}
    keys = sorted(set(Lm) & set(Am))
    groups = {"all": keys, "both_right": [k for k in keys if Lm[k]["hit1"] and Am[k]["hit1"]],
              "astra_only": [k for k in keys if Am[k]["hit1"] and not Lm[k]["hit1"]],
              "luna_only": [k for k in keys if Lm[k]["hit1"] and not Am[k]["hit1"]],
              "both_wrong": [k for k in keys if not Lm[k]["hit1"] and not Am[k]["hit1"]]}
    out = {"n_pairs": len(keys), "groups": {}}
    fL = {k: query_feats(Lm[k]) for k in keys}
    fA = {k: query_feats(Am[k]) for k in keys}
    for g, ks in groups.items():
        row = {"n": len(ks), "by_ds": {ds: sum(1 for k in ks if k[0] == ds) for ds in C.DATASETS}, "luna": {}, "astra": {}}
        for f, _, _ in PAIR_FEATS:
            row["luna"][f] = mean([float(fL[k][f]) if fL[k][f] is not None else None for k in ks])
            row["astra"][f] = mean([float(fA[k][f]) if fA[k][f] is not None else None for k in ks])
        out["groups"][g] = row
    # Astra minus Luna per feature, among questions both answer correctly vs questions only Astra answers correctly.
    # "consistent" = a gap of at least 5 points among both-right questions, the same sign among only-Astra-right
    # questions, and the two gaps within 10 points of each other (percent features only).
    out["gaps"] = {}
    for f, _, kind in PAIR_FEATS:
        if kind != "pct":
            continue
        br, ao = out["groups"]["both_right"], out["groups"]["astra_only"]
        if None in (br["luna"][f], br["astra"][f], ao["luna"][f], ao["astra"][f]):
            continue
        gb, ga = 100 * (br["astra"][f] - br["luna"][f]), 100 * (ao["astra"][f] - ao["luna"][f])
        out["gaps"][f] = {"both_right": gb, "astra_only": ga,
                          "consistent": abs(gb) >= 5 and gb * ga > 0 and abs(ga - gb) <= 10,
                          "sign_flip": gb * ga < 0}
    out["by_ds"] = {}
    for ds in C.DATASETS:
        ks = [k for k in keys if k[0] == ds]
        out["by_ds"][ds] = {"n": len(ks), "luna": {f: mean([float(fL[k][f]) if fL[k][f] is not None else None for k in ks]) for f, _, _ in PAIR_FEATS},
                            "astra": {f: mean([float(fA[k][f]) if fA[k][f] is not None else None for k in ks]) for f, _, _ in PAIR_FEATS},
                            "h1_luna": mean([Lm[k]["hit1"] for k in ks]), "h1_astra": mean([Am[k]["hit1"] for k in ks])}
    return out


# --------------------------------------------------------------------------- 8. relation to correctness

CORR_FEATS = (("py_edges", "a Python program reads edges"), ("carry", "a program embeds ids from earlier outputs"),
              ("revision", "any program revision (sim >= 0.5)"), ("py_error", "any Python error"),
              ("fullout_reuse", "re-reads a saved full output"),
              ("top1_named_after_seen", "final top-1 named after an earlier turn showed it"),
              ("top1_edge_check", "edge read naming the final top-1"), ("lineages3", ">= 3 program lineages"),
              ("any_scan", "any full-scan program"))


def correctness_stats(rows, min_n=20):
    rows = ok_rows(rows)
    feats = [(r, query_feats(r)) for r in rows]
    out = {}
    for f, _ in CORR_FEATS:
        per = {}
        num = wsum = 0.0
        for ds in C.DATASETS:
            sel = [(r, q) for r, q in feats if r["dataset"] == ds]
            yes = [r["hit1"] for r, q in sel if q[f]]
            no = [r["hit1"] for r, q in sel if not q[f]]
            d = {"n_yes": len(yes), "n_no": len(no), "h1_yes": mean(yes), "h1_no": mean(no)}
            if len(yes) >= min_n and len(no) >= min_n:
                d["diff"] = d["h1_yes"] - d["h1_no"]
                num += len(sel) * d["diff"]
                wsum += len(sel)
            else:
                d["diff"] = None
            per[ds] = d
        out[f] = {"by_ds": per, "ds_weighted_diff": num / wsum if wsum else None, "weight_n": int(wsum)}
    return out


# --------------------------------------------------------------------------- manual-check sample of lineage links

def programs_text(result, harness):
    """Inline Python sources in call order (same parse as extract), for the manual-check sample only."""
    attempt = result.get("attempt_path")
    tr = pi_trace(attempt, union=bool(result.get("compaction_starts"))) if harness == "pi" else codex_trace(attempt)
    out = []
    for c in (tr or {}).get("calls", []):
        if c["name"] != "bash":
            continue
        info = parse_shell(c["cmd"] or "")
        if info["py"]:
            out.append("\n".join(info["sources"]) if info["sources"] else c["cmd"])
    return out


def lineage_sample(data, cfgs, n_per=25, seed=20261002):
    """50 parent/child pairs (25 Luna, 25 Astra; 20 with 0.5 <= sim < 0.8 and 5 with sim >= 0.8 each) for a manual check."""
    import random
    rng = random.Random(seed)
    picks = []
    for name in ("luna_full", "astra_full"):
        rows = ok_rows(data.get(name, []))
        mid, high = [], []
        for r in rows:
            for j, p in enumerate(r["progs"]):
                if p.get("par") is None:
                    continue
                (high if p["sim"] >= 0.8 else mid if p["sim"] >= 0.5 else []).append((r, j))
        rng.shuffle(mid)
        rng.shuffle(high)
        k_high = n_per // 5
        picks += [(name, r, j) for r, j in mid[:n_per - k_high] + high[:k_high]]
    lines = ["# Manual check of program-lineage links (sample of 50)", "",
             "Each pair is a later program and the earlier program of the same query that is most similar to it "
             "(difflib ratio on the first 4,000 characters). Mark each pair as `revision` (the later program is an edited "
             "copy of the earlier one) or `not` (similar only through boilerplate, or a different operation).",
             "Sampling: fixed seed 20261002; per model 20 pairs with 0.5 <= sim < 0.8 and 5 with sim >= 0.8.", ""]
    for k, (name, r, j) in enumerate(picks, 1):
        cfg = cfgs[name]
        try:
            res = C.read_json(r["src"])
            texts = programs_text(res, cfg["harness"])
            child, parent = texts[j], texts[r["progs"][j]["par"]]
        except Exception as e:  # noqa: BLE001
            child = parent = f"(could not re-read: {e!r})"
        p = r["progs"][j]
        cut = lambda t: "\n".join(t.splitlines()[:30])[:2500]  # noqa: E731
        lines += [f"## {k}. {cfg['short']} {r['dataset']} {r['query_id']}: program {p['par'] + 1} -> program {j + 1} "
                  f"(sim {p['sim']:.2f}; boilerplate removed {p['sims']:.2f})", "", "Judgement: ______", "",
                  "Earlier program:", "", "```python", cut(parent), "```", "", "Later program:", "", "```python", cut(child), "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures

COLORS = {"luna_full": C.PALETTE[0], "astra_full": C.PALETTE[1], "A3": C.PALETTE[2], "H": C.PALETTE[6], "A4": C.PALETTE[0]}
MARKERS = {"luna_full": "o", "astra_full": "s", "A3": "^", "H": "D", "A4": "o"}
LABELS = {"luna_full": "GPT-6-Luna", "astra_full": "GPT-6-Astra", "A3": "Shell only (test subset)", "H": "Full DKGI (Codex)"}
COL_W = 3.15  # inches, ACL single column
FS = 7.0      # smallest font size in the figures (pt)
FB_CTRL = 4000  # large-output control shown in the figure and the main table (bytes)


def fig_phases(ax_py, ax_rg, phases):
    x = [1, 2, 3, 4, 5]
    for ax, kind, names, title in ((ax_py, "py", ("luna_full", "astra_full"), "Python calls"),
                                   (ax_rg, "rg", ("luna_full", "astra_full", "A3"), "rg / grep calls")):
        for name in names:
            ph = phases.get(name)
            if not ph:
                continue
            y = [100 * (b.get(kind) or 0) for b in ph["share"]]
            ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], markersize=3.8, linewidth=1.3,
                    markerfacecolor=COLORS[name] if name != "A3" else "white", markeredgewidth=1.0,
                    label=LABELS[name] if name != "A3" else "_nolegend_", clip_on=False, zorder=3)
            if name == "A3":  # labelled directly: it is on the test subset, the other lines on the full test set
                ax.text(5.0, y[2] - 13, "Shell only\n(test subset)", ha="right", va="top", fontsize=FS,
                        color=COLORS[name], linespacing=1.0)
        ax.set_title(title, fontsize=FS + 0.5, pad=2, color=C.INK)
        ax.set_xticks(x)
        ax.set_xlim(0.7, 5.3)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=FS, length=2, pad=1.5)
    ax_py.set_ylabel("share of calls (%)", fontsize=FS, labelpad=2)
    ax_rg.tick_params(labelleft=False)


def fig_feedback(ax_all, ax_ctrl, fb, ctrl):
    """Next turn runs Python after a capped rg output (filled) vs after an uncapped one (open), position-matched;
    left: all uncapped outputs; right: uncapped outputs of at least FB_CTRL bytes (large-output control)."""
    names = [n for n in ("luna_full", "astra_full") if fb.get(n)]
    ys = list(range(len(names)))[::-1]
    for ax, src, title in ((ax_all, {n: fb[n] for n in names}, "all uncapped"),
                           (ax_ctrl, {n: (ctrl.get(n) or {}).get(str(FB_CTRL)) for n in names},
                            f"uncapped ≥ {FB_CTRL // 1000} KB")):
        for y, name in zip(ys, names):
            if not src.get(name):
                continue
            w = src[name]["weighted"]["py"]
            if w["capped"] is None:
                continue
            u, c = 100 * w["uncapped_matched"], 100 * w["capped"]
            ax.plot([u, c], [y, y], color=C.GRAY, linewidth=1.1, zorder=1)
            ax.scatter([u], [y], s=22, facecolor="white", edgecolor=COLORS[name], linewidth=1.1, zorder=3, marker=MARKERS[name])
            ax.scatter([c], [y], s=22, facecolor=COLORS[name], edgecolor=COLORS[name], linewidth=1.1, zorder=3, marker=MARKERS[name])
            ax.text(max(u, c) + 4, y, f"{c - u:+.0f}", ha="left", va="center", fontsize=FS, color=C.INK2)
        ax.set_title(title, fontsize=FS + 0.5, pad=2, color=C.INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABELS[n] for n in names], fontsize=FS)
        ax.set_xlim(20, 100)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.tick_params(labelsize=FS, length=2, pad=1.5)
        ax.grid(axis="y", visible=False)
    ax_ctrl.tick_params(labelleft=False)


def feedback_legend(fig, x, y):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=C.INK2, markersize=4,
                      label="uncapped rg output"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor=C.INK2, markeredgecolor=C.INK2, markersize=4,
                      label="rg output hit the cap")]
    fig.legend(handles=handles, fontsize=FS, loc="lower center", bbox_to_anchor=(x, y), ncol=2, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0.0)


def save(fig, name):
    d = OUTDIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=300, facecolor="white")  # fixed size: exactly the ACL column width


FB_XLABEL = "next turn runs Python (% of non-error rg calls)"


def make_figures(phases, fb, ctrl):
    plt = C.mpl()
    L, R, GAP = 0.135, 0.985, 0.04
    mid = (L + R) / 2
    wd = (R - L - GAP) / 2
    LB = 0.255  # panel (b) needs room for the model names
    GB = 0.07
    wb = (R - 0.03 - LB - GB) / 2  # leaves room for the last "100" tick label
    midb = (LB + R - 0.03) / 2
    # (a) alone
    fig = plt.figure(figsize=(COL_W, 1.8))
    a = fig.add_axes([L, 0.22, wd, 0.54])
    b = fig.add_axes([L + wd + GAP, 0.22, wd, 0.54], sharey=a)
    fig_phases(a, b, phases)
    fig.text(mid, 0.015, "position in the trajectory (fifths)", ha="center", va="bottom", fontsize=FS)
    h, lab = b.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(mid, 1.0), ncol=2, fontsize=FS, handlelength=1.8,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
    save(fig, "coevolution_phases")
    plt.close(fig)
    # (b) alone
    fig = plt.figure(figsize=(COL_W, 1.3))
    a = fig.add_axes([LB, 0.28, wb, 0.44])
    b = fig.add_axes([LB + wb + GB, 0.28, wb, 0.44], sharey=a)
    fig_feedback(a, b, fb, ctrl)
    fig.text(midb, 0.015, FB_XLABEL, ha="center", va="bottom", fontsize=FS)
    feedback_legend(fig, mid, 0.86)
    save(fig, "coevolution_feedback")
    plt.close(fig)
    # combined two-panel figure: (a) phases on top, (b) feedback below
    H = 3.15
    fig = plt.figure(figsize=(COL_W, H))
    y_a, h_a = 0.565, 0.29
    y_b, h_b = 0.11, 0.165
    a = fig.add_axes([L, y_a, wd, h_a])
    b = fig.add_axes([L + wd + GAP, y_a, wd, h_a], sharey=a)
    fig_phases(a, b, phases)
    fig.text(mid, y_a - 0.105, "position in the trajectory (fifths)", ha="center", va="bottom", fontsize=FS)
    h, lab = b.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(mid, 1.0), ncol=2, fontsize=FS, handlelength=1.8,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
    c = fig.add_axes([LB, y_b, wb, h_b])
    d = fig.add_axes([LB + wb + GB, y_b, wb, h_b], sharey=c)
    fig_feedback(c, d, fb, ctrl)
    fig.text(midb, 0.005, FB_XLABEL, ha="center", va="bottom", fontsize=FS)
    feedback_legend(fig, mid, y_b + h_b + 0.075)
    fig.text(0.005, 0.985, "(a)", fontsize=8, weight="bold", va="top")
    fig.text(0.005, y_b + h_b + 0.135, "(b)", fontsize=8, weight="bold", va="top")
    save(fig, "coevolution_figure")
    plt.close(fig)


# --------------------------------------------------------------------------- report

LIN_CFGS = ("luna_full", "astra_full", "A4", "H", "codex_full", "A6", "A5")
PHASE_CFGS = ("luna_full", "astra_full", "A3", "A4", "H", "A6", "A5", "A2")
FB_CFGS = ("luna_full", "astra_full", "A4", "H", "codex_full", "A6", "A3")


def compute(data, cfgs):
    res = {"coverage": {}, "lineage": {}, "reuse": {}, "saved": {}, "types": {}, "phases": {}, "feedback": {},
           "feedback_py_capped": {}, "feedback_ctrl": {}, "feedback_rg_files": {}, "tools": {}, "paired": None,
           "correctness": {}, "codex_harness": {}}
    for name, rows in data.items():
        cfg = cfgs[name]
        res["coverage"][name] = coverage(cfg, rows)
        res["saved"][name] = saved_stats(rows)
        res["phases"][name] = phase_stats(rows)
        if name in LIN_CFGS:
            res["lineage"][name] = {f"{'strip' if st else 'raw'}_{tau}": lineage_stats(rows, tau, st)
                                    for tau in THRESHOLDS for st in (False, True)}
            res["reuse"][name] = reuse_repair(rows)
            res["types"][name] = type_stats(rows)
        if name in FB_CFGS:
            res["feedback"][name] = feedback_stats(rows, "rg")
            res["feedback_py_capped"][name] = feedback_stats(rows, "py")
            res["feedback_ctrl"][name] = {str(b): feedback_stats(rows, "rg", min_unc_bytes=b) for b in CTRL_BYTES}
            res["feedback_rg_files"][name] = feedback_stats(rows, "rg", rg_files=True)
        if cfg["kind"] in ("tool", "tool_code"):
            res["tools"][name] = tool_stats(rows, cfg["kind"])
        if name in ("luna_full", "astra_full", "A4", "H"):
            res["correctness"][name] = correctness_stats(rows)
        if cfg["harness"] == "codex":
            cx = [r.get("codex") or {} for r in ok_rows(rows)]
            res["codex_harness"][name] = {"n": len(cx), "cells": sum(c.get("n_cells") or 0 for c in cx),
                                          "multi_exec_cells": sum(c.get("multi_exec_cells") or 0 for c in cx),
                                          "other_runtime_calls": dict(sum((Counter(c.get("other_runtime") or {}) for c in cx), Counter()))}
    if "luna_full" in data and "astra_full" in data:
        res["paired"] = paired_stats(data["luna_full"], data["astra_full"])
    if "codex_full" in data and "luna_full" in data:
        keys = {(r["dataset"], r["query_id"]) for r in data["codex_full"]}
        same = [r for r in data["luna_full"] if (r["dataset"], r["query_id"]) in keys]
        res["luna_on_codex_full_questions"] = {"n": len(same), "lineage_raw_0.5": lineage_stats(same, 0.5),
                                               "reuse": reuse_repair(same), "h1": mean([r["hit1"] for r in same]),
                                               "h1_codex": mean([r["hit1"] for r in data["codex_full"]])}
    return res


def nfmt(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def D(x, d=1):
    """A difference of two shares, in points."""
    return "n/a" if x is None else f"{100 * x:+.{d}f}"


def wpy(f, key="py"):
    """(capped, matched uncapped, diff) of a feedback_stats result, or Nones."""
    if not f:
        return None, None, None
    w = f["weighted"][key]
    return w["capped"], w["uncapped_matched"], w["diff"]


def phase_gap(ph, a="A4", b="luna_full"):
    """Largest absolute difference (points) between two configurations' Python and rg shares over the fifths."""
    if a not in ph or b not in ph:
        return None
    return max(abs((x.get(k) or 0) - (y.get(k) or 0)) for k in ("py", "rg")
               for x, y in zip(ph[a]["share"], ph[b]["share"]))


def cache_mb():
    return sum(p.stat().st_size for p in CACHE.glob("*.jsonl")) / 1e6 if CACHE.is_dir() else 0.0


DISPLAY = {"luna_full": "GPT-6-Luna", "astra_full": "GPT-6-Astra", "A4": "Full DKGI (test subset)",
           "H": "Full DKGI (Codex)", "codex_full": "Full DKGI (Codex, full test)", "A6": "No edges",
           "A5": "Graph tools + code", "A2": "Graph tools", "A3": "Shell only"}


def report(res, cfgs, names, elapsed, no_cache=False):
    cov, lin, reu, sav, typ, ph, fb, tools = (res[k] for k in ("coverage", "lineage", "reuse", "saved", "types", "phases",
                                                               "feedback", "tools"))
    ctrl, rgf, fpy = res.get("feedback_ctrl", {}), res.get("feedback_rg_files", {}), res.get("feedback_py_capped", {})
    present = [n for n in names if n in cov]
    S = lambda n: cfgs[n]["short"]  # noqa: E731
    L = []
    g = lambda n, key="raw_0.5": lin.get(n, {}).get(key, {})  # noqa: E731
    kb = lambda b: f"{b // 1000} KB"  # noqa: E731
    L += ["# Tool coding and tool calling within one query: behaviour analysis", "",
          f"Generated by `analysis_final/coevolution.py` ({elapsed:.0f} s, "
          f"{'all trajectories re-extracted' if no_cache else 'cached features'}). Reads existing trajectories only; "
          "no model call. All numbers are point estimates; n is given in parentheses.", ""]
    pr = res.get("paired")
    crL, crA = res["correctness"].get("luna_full"), res["correctness"].get("astra_full")
    # ---------------- English part
    L += ["## Definitions", "",
          "- **Program (tool).** One inline Python source issued inside one shell call (`python3 -c`, or a heredoc fed to "
          "`python3`). Programs are not saved or reused across queries; a program that writes a script file and later runs it "
          "would count as a saved tool (section 2).",
          "- **Call / turn.** A call is one tool call the model issued (Pi: `bash`, `read`, or a graph tool; Codex: one "
          "`exec_command` inside a code cell). A turn groups the calls issued in one model response; the *next turn* after a "
          "call is the following model response, i.e. the first calls issued after that output was visible (one response "
          "can issue several calls). *Earlier* (ids, the top-1 answer, written files) means an earlier turn: calls of the same "
          "turn cannot see each other's outputs.",
          "- **rg call.** A shell call whose commands include `rg`/`grep` as a search, and no Python. File listings "
          "(`rg --files`) are counted as other shell calls (section 5 also reports the variant that counts them as rg calls).",
          f"- **Revision / lineage.** A program revises an earlier program of the same query if their difflib ratio (first {SIM_CAP:,} "
          "characters) is at least the threshold (0.5 or 0.8); its parent is the most similar earlier program. A lineage is a "
          "chain of such links. Boilerplate-removed variant (used in the main table): the same after removing boilerplate "
          "lines (imports, path constants, file/shard loops, `json.loads(line)`, try/except-continue, the Graph tools client setup).",
          "- **Embeds earlier ids.** The program text contains a 4-8 digit integer (not a year 1900-2035, not a multiple of 1000, "
          "not in the prompt) that appeared in a model-visible output of an earlier turn. Strict: it appeared there as a record id "
          "(`\"node_id\": N`, `[N]`), an edge row, a list element or at the start of an output line; loose: anywhere.",
          "- **Failed program.** Non-zero exit (Pi `isError` / `Command exited with code N`; Codex exit code), a Python traceback "
          "or syntax error, or a timeout. **Did not run**: the shell reports the interpreter itself as not found (`python: "
          "command not found`, where only `python3` exists); such programs are counted as programs but not as failures, and "
          "are left out of failure shares and repair rates. Failure shares use the programs that ran as denominator.",
          "- **Repair.** The next Python program issued in a later turn has similarity >= 0.5 to the failed one; `runs` = that "
          "program neither fails nor did not run. Comparison point: the same similarity test after a program that ran without "
          "failing.",
          "- **Capped output.** Pi: the single-stage output cap marker (`[... harness output limit ...]`); the cap shows at most "
          "9,000 bytes and 900 lines of the head and the same of the tail, and saves the complete output to a file whose path "
          "the marker gives. Codex: its own `…N tokens truncated…` marker (a different limit; reported separately).",
          "- **Field projection / output limit (section 5).** Counted for shell commands of the next turn only (not Python): "
          "projection = prints parts of lines (`rg -o`, `cut -f`, `jq`, `awk '{print $..}'`, `sed -n 's...'`); output limit = "
          "`| head`, `| tail`, `-m N`, `--max-count`, `wc`, `-c`. A Python key-access pattern matched almost every program, so "
          "it would only re-count \"runs Python\".",
          "- **Touches edge files.** The call names an edge file or directory (`edges/`, `.tsv`; this includes `ls edges`), "
          "reads one with the read tool, or (Graph tools) calls `neighbors`.",
          "- **Gold in output.** Integer criterion: a gold id appears as a standalone integer in that call's visible output; "
          "record criterion: as a node record (`\"node_id\": ID` or `[ID]`).", ""]
    L += ["## 0. Coverage", ""]
    rows = []
    for n in present:
        c = cov[n]
        rows.append([cfgs[n]["paper"], *(f"{c['by_ds'][d]['n']}" for d in C.DATASETS), f"{c['n']}/{c['expected']}",
                     c["extract_errors"], ", ".join(f"{k}:{v}" for k, v in c["visibility"].items()), P(c["h1"]),
                     cfgs[n].get("note", "")])
    L += [C.md_table(["Configuration", "Amazon", "MAG", "Prime", "Terminal / expected", "Extract errors", "Visibility",
                      "H@1 (pooled)", "Note"], rows), "",
          "Full DKGI and the ablation variants are on the test subset (710 queries); Full DKGI there is the test-subset part "
          "of the Pi + GPT-6-Luna full-test run. H@1 is recomputed from the prediction and the test gold file.", ""]
    # ---------------- 1. lineages
    L += ["## 1. Program lineages", "", "### 1a. Revisions and lineages", ""]
    rows = []
    for n in [x for x in LIN_CFGS if x in lin]:
        a, b, sa, sb = g(n), g(n, "raw_0.8"), g(n, "strip_0.5"), g(n, "strip_0.8")
        rows.append([S(n), a["n_queries"], a["n_programs"], F(a["programs_per_query"], 2), P(sa["revision_share"]),
                     P(sb["revision_share"]), P(a["revision_share"]), P(b["revision_share"]), P(a["revisions_parent_seen"]),
                     P(a["queries_with_revision"]), F(a["lineages_per_py_query_median"]), F(sa["lineages_per_py_query_median"]),
                     F(a["versions_per_lineage_mean"], 2), F(a["longest_lineage_median"])])
    L += [C.md_table(["Config", "n queries", "n programs", "Programs / query", "Boilerplate removed >=0.5 %",
                      "removed >=0.8 %", "Raw >=0.5 %", "raw >=0.8 %", "Parent output visible % (raw >=0.5)",
                      "Py queries with a revision % (raw)", "Lineages / Py query (median, raw)", "same, removed",
                      "Versions / lineage (mean, raw)", "Longest lineage (median, raw)"], rows), "",
          "The main table quotes the boilerplate-removed shares, since shared boilerplate lines (imports, file "
          "loops) make unrelated programs look similar. Almost all boilerplate-removed links are also raw links (see the counts "
          "below). Graph tools + code programs share a fixed client-setup block, so its raw similarities are inflated.", ""]
    rows = []
    for n in [x for x in LIN_CFGS if x in lin]:
        sa, a = g(n, "strip_0.5"), g(n)
        rows.append([S(n), sa.get("links", "n/a"), sa.get("links_not_raw", "n/a"), a.get("links", "n/a")])
    L += [C.md_table(["Config", "Removed links >=0.5 (n)", "of which not raw links (n)", "Raw links >=0.5 (n)"], rows), ""]
    L += ["### 1b. Ids from earlier outputs, failures and repair", ""]
    rows = []
    for n in [x for x in LIN_CFGS if x in reu]:
        r = reu[n]
        rows.append([S(n), r["n_programs"], P(r["carry_strict_programs"]), P(r["carry_loose_programs"]), P(r["carry_strict_queries"]),
                     r["not_run_programs"], r["error_programs"], P(r["error_share"]), r["error_with_later_program"],
                     P(r["err_nextprog_repair"]), P(r["err_nextprog_repair_ok"]), P(r["ok_nextprog_similar"]),
                     r["error_with_next_turn"], P(r["err_next_python"]), P(r["err_next_repair"]), P(r["err_next_repair_ok"])])
    L += [C.md_table(["Config", "n programs", "Embed earlier ids, strict %", "loose %", "Py queries with strict reuse %",
                      "Did not run (n)", "Failed programs (n)", "Failed % of programs that ran",
                      "Failed with a later program (n)", "Next program similar (repair) %", "and runs %",
                      "After a success: next program similar %", "Failed with a next turn (n)",
                      "Next turn runs Python %", "Next turn repairs it %", "and runs %"], rows),
          "", "Repair rates use as denominator the failed programs followed by a later program (next-program columns) or by a "
          "next turn (next-turn columns); `and runs` is a share of the same denominator. The success column is the comparison "
          "point: programs that ran without failing and were followed by a later program."
          + "".join(f" {S(n)}: {P(reu[n]['err_nextprog_repair'])}% after a failure vs {P(reu[n]['ok_nextprog_similar'])}% after a success."
                    for n in ("luna_full", "astra_full") if n in reu)
          + " For Graph tools + code the shared client-setup block inflates raw similarity, so its repair and comparison "
          "columns say little.", ""]
    rows = []
    for n in [x for x in LIN_CFGS if x in reu]:
        r = reu[n]
        if not r["error_programs"]:
            continue
        rows.append([S(n), r["error_programs"], ", ".join(f"{k} {v}" for k, v in r["error_types"].items()),
                     ", ".join(f"{k} {v}" for k, v in r["err_next_turn_kinds"].items()),
                     ", ".join(f"{k} {v}" for k, v in r["err_nextprog_repair_turn_gap"].items())])
    L += ["Failure types (last exception name in the output, else timeout / command not found / non-zero exit; the missing "
          "interpreter is not among them), call types of the next turn, and turns between a failure and its repair:", "",
          C.md_table(["Config", "Failed programs", "Failure types (n)", "Next turn call types (n)", "Turns to repair (n)"], rows), ""]
    L += ["### 1c. First to last version within a lineage (lineages with >= 2 versions)", ""]
    def moves(key, up):
        out = []
        for n in LIN_CFGS:
            f = g(n, key).get("first_to_last") or {} if n in lin else {}
            a, b = f.get("first_tr"), f.get("last_tr")
            if a is not None and b is not None and ((b > a) if up else (b < a)):
                out.append(DISPLAY.get(n, cfgs[n]["paper"]))
        return out
    moves_txt = "; ".join(f"at {key.replace('_', ' ')} it falls for {', '.join(moves(key, False)) or 'none'} and rises for "
                          f"{', '.join(moves(key, True)) or 'none'}" for key in ("raw_0.5", "raw_0.8"))
    rows = []
    for n in [x for x in LIN_CFGS if x in lin]:
        for key in ("raw_0.5", "raw_0.8", "strip_0.5", "strip_0.8"):
            a = g(n, key)
            f = a.get("first_to_last") or {}
            if not a.get("multi_version_lineages"):
                continue
            rows.append([S(n), key.replace("raw_", "raw ").replace("strip_", "removed "), a["multi_version_lineages"],
                         f"{P(f.get('first_tr'))} -> {P(f.get('last_tr'))}", f"{P(f.get('first_e'))} -> {P(f.get('last_e'))}",
                         f"{P(f.get('first_gi'))} -> {P(f.get('last_gi'))}", f"{P(f.get('first_gr'))} -> {P(f.get('last_gr'))}"])
    L += [C.md_table(["Config", "Similarity", "n lineages", "Output capped %", "Failed %", "Gold id in output %",
                      "Gold record in output %"], rows), "",
          "**Not for the main table.** The direction of the capped share from first to last version depends on the "
          f"threshold: {moves_txt}. The failed share falls almost by "
          "construction: a failure is usually followed by a similar program (section 1b), that program joins the same lineage, and the "
          "last version is the one the agent stopped editing. More generally the last version comes later in the "
          "trajectory, when more is known, and the agent tends to stop once an output looks usable, so these rows describe "
          "that lineages converge; they are not evidence that editing causes the improvement.", ""]
    # ---------------- 2. saved tools
    L += ["## 2. Saved tools", "",
          "A saved tool would be a script file written during the query and run later in the same query. Counts are queries.", ""]
    rows = []
    for n in present:
        s = sav[n]
        rows.append([S(n), s["n"], s["writes_any_file"], s["writes_script_file"], s["runs_script_file"], s["runs_script_written_earlier"],
                     s["reads_back_written_file"], s["defines_function"], s["exec_or_runpy"], s["scratch_any_file"], s["scratch_script_file"]])
    L += [C.md_table(["Config", "n queries", "Writes any file", "Writes a script file", "Runs a script file",
                      "Runs a script written earlier", "Reads back a written file (later turn)", "Defines a function",
                      "exec/runpy", "Leaves any file in scratch", "Leaves a script in scratch"], rows), "",
          "Writes are shell redirections, `tee`, Python `open(..., 'w')` and `Path.write_text`; files left in scratch are listed "
          "from the attempt's scratch directory (harness-saved full outputs excluded). `Runs a script written earlier` counts "
          "any earlier call, the same turn included, so its zero is the stronger statement; `Reads back` counts only files "
          "written in an earlier turn. Pi exposes only `read` and `bash`; it has no write or edit tool.", ""]
    # ---------------- 3. program types
    L += ["## 3. Program types", "", "Share of programs with each label (labels overlap); in brackets, share of queries with "
          "Python that have at least one such program. File-based configurations use AST labels from "
          "`analyze_graph_behavior.py` (edge reads, chained expansion, set operations) plus pattern rules; Graph tools + code "
          "uses the graph-service requests each program actually made.", ""]
    tn = [x for x in LIN_CFGS if x in typ]
    rows = []
    for t in TYPES:
        rows.append([TYPE_LABEL[t], *(f"{P(typ[n]['programs'][t])} ({P(typ[n]['queries'][t])})" for n in tn)])
    rows.append(["n programs (n queries with Python)", *(f"{typ[n]['n_programs']} ({typ[n]['n_queries_py']})" for n in tn)])
    L += [C.md_table(["Program type", *(S(n) for n in tn)], rows), "",
          "Rules: full node scan = iterates over all node shards (glob/listdir/range over shards); lookup by id = shard "
          "arithmetic (`// 10000`, `:04d`) or `get_node`; 2-hop = an edge filter seeded by an earlier edge result in the same "
          "program (`chained_expansion`) or an adjacency index expanded from a seed loop, or (tools) a `neighbors` request on "
          "an id listed as a result of an earlier `neighbors` request of the same program (the seed that the response "
          "header echoes, and the seed itself as a result, are excluded); targeted check = embeds a strict earlier id "
          "and looks it up (shard, edge filter, membership or candidate check).", ""]
    # ---------------- 4. phases
    gap = phase_gap(ph)
    L += ["## 4. Phase structure", "", "Share of calls by call type in each fifth of the trajectory (calls pooled over queries)."
          + (f" Shell only is on the test subset (710 queries); on the same subset, Full DKGI's Python and rg shares differ "
             f"from the full-test GPT-6-Luna values by at most {100 * gap:.1f} points, so the figure can show Shell only next to "
             "the full-test lines if the caption says so." if gap is not None else ""), ""]
    rows = []
    for n in [x for x in PHASE_CFGS if x in ph]:
        p = ph[n]
        for k in KINDS:
            vals = [b.get(k) or 0 for b in p["share"]]
            if not any(vals):
                continue
            rows.append([S(n), KIND_LABEL[k], *(P(v) for v in vals)])
        rows.append([S(n), "n calls", *(str(x) for x in p["n_steps"])])
    L += [C.md_table(["Config", "Call type", "1st fifth", "2nd", "3rd", "4th", "5th"], rows), ""]
    # ---------------- 5. feedback
    pi_cap_min = min((fb[n]["capped_visible_bytes_min"] for n in FB_CFGS if n in fb and cfgs[n]["harness"] == "pi"
                      and fb[n]["capped_visible_bytes_min"]), default=0)
    L += ["## 5. Harness feedback and the next turn (association)", "",
          "Unit: a non-error rg/grep call with visible output and a later turn. Outcome = the next turn contains such a call. "
          "Capped vs uncapped are compared within position fifths of the rg call (fifths with >= 20 calls on both sides) and "
          "averaged with the capped counts as weights (`matched` = uncapped rate reweighted to the capped position mix). "
          f"Under the single-stage cap, capped outputs are all large (the model sees at least {pi_cap_min / 1000:.1f} KB of "
          "each; column 5), so the comparison partly contrasts large with small results; the control rows keep "
          "only uncapped outputs of at least 4 KB or 9 KB. Read every row as \"after an output that hit the single-stage cap, "
          "the next turn runs Python more often\", not as an effect of the cap.", ""]
    rows = []
    for n in [x for x in FB_CFGS if x in fb]:
        f = fb[n]
        w = f["weighted"]
        rows.append([S(n), f["n_capped"], f["n_uncapped"], f["weight_n"],
                     "n/a" if f["capped_visible_bytes_min"] is None else f"{f['capped_visible_bytes_min']:,} / {int(f['capped_visible_bytes_median']):,}",
                     *(f"{P(w[k]['capped'])} / {P(w[k]['uncapped_matched'])} ({D(w[k]['diff'])})" for k in ("py", "proj", "lim", "fo"))])
    L += [C.md_table(["Config", "n capped", "n uncapped", "n capped in matched fifths", "Capped visible bytes min / median",
                      "Next turn runs Python: capped / matched (diff)", "Shell field projection", "Shell output limit",
                      "Re-reads saved full output"], rows), "",
          "Codex rows use Codex's own truncation, not the single-stage cap. Shell only cannot run Python.", ""]
    rows = []
    for n in [x for x in FB_CFGS if x in fb]:
        cells = []
        for f in (fb[n], ctrl.get(n, {}).get(str(CTRL_BYTES[0])), ctrl.get(n, {}).get(str(CTRL_BYTES[1])), rgf.get(n)):
            c, u, d = wpy(f)
            cells.append("n/a" if d is None else f"{P(c)} / {P(u)} ({D(d)}; n = {f['n_uncapped']:,} unc., {f['weight_n']:,} capped used)")
        rows.append([S(n), *cells])
    L += ["Next turn runs Python, capped / matched uncapped (diff), with the large-output controls and the `rg --files` "
          "variant:", "",
          C.md_table(["Config", "All uncapped outputs (main)", f"Uncapped >= {kb(CTRL_BYTES[0])} only",
                      f"Uncapped >= {kb(CTRL_BYTES[1])} only", "Counting `rg --files` as rg calls"], rows), "",
          "".join(f"{S(n)}: {D(wpy(fb[n])[2])} against all uncapped outputs, {D(wpy(ctrl[n][str(CTRL_BYTES[0])])[2])} against "
                  f"uncapped outputs of at least {kb(CTRL_BYTES[0])}, {D(wpy(ctrl[n][str(CTRL_BYTES[1])])[2])} against at least "
                  f"{kb(CTRL_BYTES[1])}. " for n in ("luna_full", "astra_full") if n in fb and n in ctrl)
          + "On the test subset the 9 KB control keeps few uncapped outputs, so fewer fifths are usable (last number in each cell).", ""]
    for n in ("luna_full", "astra_full"):
        if n not in fb:
            continue
        rows = []
        c4 = ctrl.get(n, {}).get(str(CTRL_BYTES[0])) or {"bins": [{}] * 5}
        for b, b4 in zip(fb[n]["bins"], c4["bins"]):
            rows.append([b["bin"], b["n_capped"], b["n_uncapped"], P(b["py_capped"]), P(b["py_uncapped"]),
                         b4.get("n_uncapped", "n/a"), P(b4.get("py_uncapped")), "yes" if b["used"] else "no"])
        L += [f"Per fifth, {S(n)}:", "", C.md_table(["Fifth", "n capped", "n uncapped", "Python after capped %",
                                                       "after uncapped %", f"n uncapped >= {kb(CTRL_BYTES[0])}",
                                                       f"after uncapped >= {kb(CTRL_BYTES[0])} %", "Used"], rows), ""]
    rows = []
    for n in [x for x in FB_CFGS if x in fpy]:
        f = fpy[n]
        w = f["weighted"]["fo"]
        if w["capped"] is None:
            continue
        rows.append([S(n), f["n_capped"], f["n_uncapped"], P(w["capped"]), P(w["uncapped_matched"])])
    L += ["After a non-error **Python** call, the next turn re-reads the saved full output (the saved file's path appears "
          "only in the cap marker, so the uncapped column is a floor, not a comparison):", "",
          C.md_table(["Config", "n capped", "n uncapped", "capped %", "matched uncapped %"], rows), ""]
    # ---------------- 6. tools
    L += ["## 6. Tool-mediated variants (test subset)", ""]
    rows = []
    if "A2" in tools:
        t = tools["A2"]
        rows.append(["Graph tools", t["n"], F(t["calls_per_query_mean"]), "-", F(t["service_calls_mean"]), F(t["service_calls_median"]),
                     ", ".join(f"{k} {v:.1f}" for k, v in t["service_by_name_per_query"].items()), "-", "-", "-", "-",
                     f"{P(t['neighbors_cap_rate'])} (n={t['neighbors_calls']})", P(t["queries_with_cap_hit"])])
    if "A5" in tools:
        t = tools["A5"]
        rows.append(["Graph tools + code", t["n"], F(t["calls_per_query_mean"]), F(t["programs_per_query_mean"]),
                     F(t["service_calls_mean"]), F(t["service_calls_median"]),
                     ", ".join(f"{k} {v:.1f}" for k, v in t["service_by_name_per_query"].items()),
                     f"{F(t['service_per_program_mean'])} / {F(t['service_per_program_median'])} / {t['service_per_program_p90']} / {t['service_per_program_max']}",
                     P(t["programs_with_2plus_calls"]), P(t["programs_with_2plus_tools"]), P(t["programs_with_dynamic_2hop"]),
                     f"{P(t['neighbors_cap_rate'])} (n={t['neighbors_calls']})", P(t["queries_with_cap_hit"])])
    L += [C.md_table(["Config", "n", "Model calls / query", "Programs / query", "Graph-tool calls / query (mean)", "(median)",
                      "By tool, per query", "Calls per program mean / median / p90 / max", "Programs with >= 2 calls %",
                      "with >= 2 tools %", "with a 2-hop neighbors chain %", "neighbors at k cap %", "Queries with a cap hit %"], rows), ""]
    if "A5" in tools:
        L += [f"Graph-service requests mapped to the issuing program by time window: {P(tools['A5']['assigned_share'])}% "
              f"({tools['A5']['unassigned_service_calls']} unassigned). For comparison, Full DKGI writes "
              f"{F(reu.get('A4', {}).get('n_programs', 0) / max(1, cov.get('A4', {}).get('n', 1)), 2)} programs per query on the same questions. "
              "The service returns at most k = 20 `neighbors` results by default, so a response reporting more found results "
              "than k hit the cap.", ""]
    # ---------------- 7. paired
    if pr:
        L += ["## 7. GPT-6-Luna vs GPT-6-Astra on the same questions (full test)", "",
              "Per-query behaviour (share of queries unless marked mean), Luna / Astra. Groups by H@1 of the two runs.", ""]
        gs = ("all", "both_right", "astra_only", "luna_only", "both_wrong")
        rows = []
        for f, lab, kind in PAIR_FEATS:
            cells = []
            for gname in gs:
                gr = pr["groups"][gname]
                a, b = gr["luna"][f], gr["astra"][f]
                cells.append(f"{F(a)} / {F(b)}" if kind == "num" else f"{P(a)} / {P(b)}")
            rows.append([lab, *cells])
        rows.append(["n questions", *(str(pr["groups"][gname]["n"]) for gname in gs)])
        L += [C.md_table(["Behaviour", "All", "Both right", "Only Astra right", "Only Luna right", "Both wrong"], rows), ""]
        rows = []
        for f, lab, kind in PAIR_FEATS:
            if f not in pr.get("gaps", {}):
                continue
            x = pr["gaps"][f]
            rows.append([lab, f"{x['both_right']:+.1f}", f"{x['astra_only']:+.1f}",
                         "consistent" if x["consistent"] else ("sign flips" if x["sign_flip"] else "not consistent")])
        edge_lab = {f: lab for f, lab, _ in PAIR_FEATS}
        edge_incons = [f for f in ("edges_any", "py_edges") if f in pr.get("gaps", {}) and not pr["gaps"][f]["consistent"]]
        L += ["Astra minus Luna (points) among questions both answer correctly and among questions only Astra answers "
              "correctly. A feature is called consistent when the gap is at least 5 points among both-right questions, has the "
              "same sign among only-Astra-right questions, and the two gaps are within 10 points:", "",
              C.md_table(["Behaviour", "Gap, both right", "Gap, only Astra right", "Verdict"], rows), "",
              "Only the consistent features can be described as differences of style between the two models (the gap is "
              "about as large where both models are right as where only Astra is)."
              + ((" Edge use is not consistent (Luna / Astra): " + "; ".join(
                  f"{edge_lab[f]} - both right {P(pr['groups']['both_right']['luna'][f])} / {P(pr['groups']['both_right']['astra'][f])}, "
                  f"only Astra right {P(pr['groups']['astra_only']['luna'][f])} / {P(pr['groups']['astra_only']['astra'][f])}"
                  + (" (the gap changes sign)" if pr["gaps"][f]["sign_flip"] else "") for f in edge_incons) + ".") if edge_incons else "")
              + " Features that are not consistent are not style differences in this sense; no row explains the questions "
              "Astra wins.", ""]
        rows = []
        for f, lab, kind in PAIR_FEATS:
            rows.append([lab, *((f"{F(pr['by_ds'][d]['luna'][f])} / {F(pr['by_ds'][d]['astra'][f])}" if kind == "num" else
                                 f"{P(pr['by_ds'][d]['luna'][f])} / {P(pr['by_ds'][d]['astra'][f])}") for d in C.DATASETS)])
        rows.append(["H@1", *(f"{P(pr['by_ds'][d]['h1_luna'])} / {P(pr['by_ds'][d]['h1_astra'])}" for d in C.DATASETS)])
        rows.append(["n questions", *(str(pr["by_ds"][d]["n"]) for d in C.DATASETS)])
        L += ["By dataset (all questions):", "", C.md_table(["Behaviour", "Amazon", "MAG", "Prime"], rows), ""]
    # ---------------- 8. correctness
    corr_flips = []
    if crL and crA:
        corr_flips = [f"{lab} ({D(crL[f]['ds_weighted_diff'])} / {D(crA[f]['ds_weighted_diff'])})" for f, lab in CORR_FEATS
                      if crL[f]["ds_weighted_diff"] is not None and crA[f]["ds_weighted_diff"] is not None
                      and crL[f]["ds_weighted_diff"] * crA[f]["ds_weighted_diff"] < 0]
    L += ["## 8. Relation to correctness (descriptive, not causal)", "",
          "H@1 of queries with vs without a behaviour, per dataset; pooled = dataset-weighted difference over the datasets "
          "with >= 20 queries on both sides. Harder questions elicit more revisions and failures, so none of these is an "
          "effect." + (" The pooled difference has opposite signs for GPT-6-Luna and GPT-6-Astra for: " + "; ".join(corr_flips)
                       + ". Quote both models or none." if corr_flips else ""), ""]
    for n in ("luna_full", "astra_full", "A4", "H"):
        if n not in res["correctness"]:
            continue
        cr = res["correctness"][n]
        rows = []
        for f, lab in CORR_FEATS:
            c = cr[f]
            cells = [f"{P(c['by_ds'][d]['h1_yes'])} (n={c['by_ds'][d]['n_yes']}) vs {P(c['by_ds'][d]['h1_no'])} (n={c['by_ds'][d]['n_no']})"
                     for d in C.DATASETS]
            rows.append([lab, *cells, D(c["ds_weighted_diff"])])
        L += [f"**{cfgs[n]['paper']}**", "", C.md_table(["Behaviour", "Amazon", "MAG", "Prime", "Pooled diff (pts)"], rows), ""]
    # ---------------- Codex
    if res.get("codex_harness"):
        L += ["## Codex adapter notes", ""]
        for n, c in res["codex_harness"].items():
            L += [f"- {S(n)}: {c['n']} queries, {c['cells']} code cells, {c['multi_exec_cells']} cells with more than one "
                  f"`exec_command` (their per-command visible text is the command's runtime output), other runtime tool calls "
                  f"{c['other_runtime_calls'] or 'none'} (not counted as calls)."]
        lc = res.get("luna_on_codex_full_questions")
        if lc:
            a = lc["lineage_raw_0.5"]
            L += [f"- On the {lc['n']} full-test questions of the Codex run, Pi + GPT-6-Luna writes "
                  f"{F(a['programs_per_query'], 2)} programs per query with {P(a['revision_share'])}% raw revisions (>= 0.5); "
                  f"H@1 Pi {P(lc['h1'])}, Codex {P(lc['h1_codex'])}."]
        L += [""]
    L += ["## Caveats", "",
          "- All behaviour labels are regex/AST/similarity proxies with the thresholds stated above. A 50-pair "
          "sample of lineage links is in `lineage_pairs_sample.md`.",
          "- Relation to correctness is association only; difficulty drives both behaviour and errors.",
          "- The capped-vs-uncapped comparison is an association, and capped outputs are all large; see the controls in section 5.",
          "- First-to-last lineage changes are partly circular (later versions are later in the trajectory).",
          "- The *tool* is an inline program; Pi has no write/edit tool, which may contribute to programs never being saved.",
          "- Codex counts one `exec_command` as a call; its JavaScript cell wrapper is not counted as a program, and its "
          "truncation differs from the single-stage cap.", "",
          "## How to run", "",
          "```", "cd experiments/stark",
          "python3 analysis_final/coevolution.py            # cached; re-extracts only changed results",
          "python3 analysis_final/coevolution.py --no-cache # re-extract every trajectory (48 processes)", "```", "",
          "A cached run takes seconds; `--no-cache` re-reads every trajectory.", "",
          "Outputs: `REPORT.md`, `coevolution.json`, `tex/coevolution_table.tex`, `tex/appendix/coevolution_table_codex.tex`, "
          "`tex/coevolution_figure.tex` (figure environment with caption), `figures/coevolution_{phases,feedback}.{pdf,png}` "
          "(single panels) and `figures/coevolution_figure.{pdf,png}` (both panels, ACL column width), "
          f"`lineage_pairs_sample.md`, `cache/` (per-query features, {cache_mb():.0f} MB at this run; safe to "
          "delete). Also run by `make_all.py`, which passes `--no-cache` on to this script."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- LaTeX table and figure

def tex_table(res, cfgs, cols=("luna_full", "astra_full"), wide=False):
    """Main-text table (single column: GPT-6-Luna and GPT-6-Astra on the full test set); wide=True adds Full DKGI and
    Full DKGI (Codex) on the test subset as table*. Needs only booktabs (labels use \\raggedright + \\tabularnewline)."""
    lin, reu, sav, fb, ctrl = res["lineage"], res["reuse"], res["saved"], res["feedback"], res.get("feedback_ctrl", {})
    cols = [n for n in cols if n in lin]
    pct = lambda x: "--" if x is None else f"{100 * x:.1f}"  # noqa: E731
    g = lambda n, key="strip_0.5": lin[n][key]  # noqa: E731
    sl = "\\,/\\,"
    c4 = lambda n: (ctrl.get(n) or {}).get(str(CTRL_BYTES[0]))  # noqa: E731

    def fbval(f, key):
        if not f or f["weighted"]["py"]["capped"] is None:
            return "--"
        w = f["weighted"]["py"]
        return pct(w["capped"]) if key == "capped" else f"{100 * w['diff']:+.1f}"
    rows = [
        ("Programs per query", lambda n: f"{g(n)['programs_per_query']:.1f}"),
        ("Revisions, sim $\\geq$0.5", lambda n: pct(g(n)["revision_share"])),
        ("Revisions, sim $\\geq$0.8", lambda n: pct(g(n, "strip_0.8")["revision_share"])),
        ("Uses ids seen earlier", lambda n: pct(reu[n]["carry_strict_programs"])),
        ("Next program similar:", None),
        ("\\quad after a failure", lambda n: pct(reu[n]["err_nextprog_repair"])),
        ("\\quad after a success", lambda n: pct(reu[n]["ok_nextprog_similar"])),
        ("Saved scripts run later", lambda n: f"{sav[n]['runs_script_written_earlier']}"),
        ("Next turn runs Python:", None),
        ("\\quad after a capped \\texttt{rg}", lambda n: fbval(fb.get(n), "capped")),
        ("\\quad minus uncapped", lambda n: fbval(fb.get(n), "diff")),
        (f"\\quad minus uncapped $\\geq${CTRL_BYTES[0] // 1000}\\,KB", lambda n: fbval(c4(n), "diff")),
    ]
    head1 = {"luna_full": "GPT-6-Luna", "astra_full": "GPT-6-Astra", "A4": "Full DKGI", "H": "Full DKGI (Codex)$^\\dagger$"}
    env = "table*" if wide else "table"
    lab_w = "5.2cm" if wide else "3.75cm"
    out = ["% Generated by analysis_final/coevolution.py (the paper's table adds the capped / uncapped >= 4 KB rates from",
           "% coevolution.json feedback_ctrl[...][\"4000\"].weighted.py).",
           f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           f"\\begin{{tabular}}{{@{{}}p{{{lab_w}}}{'r' * len(cols)}@{{}}}}", "\\toprule"]
    full = [n for n in cols if n in ("luna_full", "astra_full")]
    sub = [n for n in cols if n not in full]
    if sub:
        out.append(" & \\multicolumn{%d}{c}{Full test set} & \\multicolumn{%d}{c}{Test subset (710 queries)} \\tabularnewline"
                   % (len(full), len(sub)))
        out.append("\\cmidrule(lr){2-%d} \\cmidrule(l){%d-%d}" % (1 + len(full), 2 + len(full), 1 + len(cols)))
        out.append(" & " + " & ".join("Full DKGI (Codex)$^\\dagger$" if n == "H" else "Full DKGI" for n in cols)
                   + " \\tabularnewline")
        out.append(" & " + " & ".join("GPT-6-Astra" if n == "astra_full" else "GPT-6-Luna" for n in cols) + " \\tabularnewline")
    else:
        out.append(" & \\multicolumn{%d}{c}{Full DKGI, full test set} \\tabularnewline" % len(full))
        out.append("\\cmidrule(l){2-%d}" % (1 + len(full)))
        out.append(" & " + " & ".join(head1[n] for n in cols) + " \\tabularnewline")
    out.append("\\midrule")
    for lab, fn in rows:
        out.append("\\raggedright " + lab + " & " + " & ".join(fn(n) if fn else "" for n in cols) + " \\tabularnewline")
    out += ["\\bottomrule", "\\end{tabular}"]
    nq = " / ".join(f"{lin[n]['raw_0.5']['n_queries']:,}" for n in cols)
    npg = " / ".join(f"{lin[n]['raw_0.5']['n_programs']:,}" for n in cols)
    ncap = " / ".join(f"{fb[n]['n_capped']:,}" if n in fb else "--" for n in cols)
    plain = {"luna_full": "GPT-6-Luna", "astra_full": "GPT-6-Astra", "A4": "Full DKGI on the test subset",
             "H": "Full DKGI (Codex)"}
    raw = "; ".join(f"{plain[n]} {pct(lin[n]['raw_0.5']['revision_share'])}{sl}"
                    f"{pct(lin[n]['raw_0.8']['revision_share'])}" for n in cols)
    cap = ("Tool coding and tool calling within one query. A program is one inline Python source in a shell call; "
           "programs are not saved. Values are percentages, except programs per query, saved scripts (queries) and the "
           "two differences (points). Revision: difflib similarity to the most similar earlier program of the same query, "
           f"after removing boilerplate lines such as imports and file loops (without removal, $\\geq$0.5\\,/\\,0.8: {raw}). "
           "Ids seen earlier: ids that outputs of earlier model turns showed. Next program similar: the next program in a "
           "later turn has similarity $\\geq$0.5; failures exclude programs whose interpreter was not found. Next-turn rows "
           "follow non-error \\texttt{rg} calls; uncapped rates are matched to the trajectory-position mix of the capped "
           "calls, and since capped outputs are all large, the last row keeps only uncapped outputs of at least "
           f"{CTRL_BYTES[0] // 1000}\\,KB. All rows are associations."
           + (" Full DKGI on the test subset is the test-subset part of the GPT-6-Luna full-test run; "
              "$^\\dagger$Codex truncates by its own token limit, not the single-stage cap." if sub else "")
           + f" n: queries {nq}; programs {npg}; capped \\texttt{{rg}} outputs {ncap}.")
    out += [f"\\caption{{{cap}}}", f"\\label{{tab:coevolution{'_codex' if wide else ''}}}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


def tex_figure(res):
    """Figure environment with a caption for figures/coevolution_figure.pdf (copy the PDF into the paper's figures/)."""
    ph, fb, ctrl = res["phases"], res["feedback"], res.get("feedback_ctrl", {})
    gap = phase_gap(ph)
    nL, nA = (fb.get(n, {}).get("n_capped") for n in ("luna_full", "astra_full"))
    cap = ("Tool calls over a trajectory, Full DKGI on the full test set. (a)~Share of Python and \\texttt{rg}/\\texttt{grep} "
           "calls in each fifth of the trajectory. Shell only, which cannot run Python, is on the test subset (710 queries)"
           + (f"; on that subset Full DKGI with GPT-6-Luna stays within {100 * gap:.1f} points of the full-test GPT-6-Luna "
              "line" if gap is not None else "")
           + ". (b)~Share of non-error \\texttt{rg} calls whose next turn runs Python, after an output that hit the "
           "single-stage cap (filled) and after an uncapped output (open), matched by trajectory fifth; right: only "
           f"uncapped outputs of at least {CTRL_BYTES[0] // 1000}\\,KB, since capped outputs are all large. Numbers give "
           "the difference in points; an association, not an effect of the cap"
           + (f" (capped \\texttt{{rg}} outputs: {nL:,} GPT-6-Luna, {nA:,} GPT-6-Astra)." if nL and nA else "."))
    return "\n".join(["% Needs figures/coevolution_figure.pdf (copy from analysis_final/out/coevolution/figures/).",
                      "\\begin{figure}[t]", "\\centering",
                      "\\includegraphics[width=\\columnwidth]{figures/coevolution_figure.pdf}",
                      f"\\caption{{{cap}}}", "\\label{fig:coevolution}", "\\end{figure}", ""])


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cache", action="store_true", help="re-extract every trajectory")
    ap.add_argument("--configs", nargs="*", help="restrict to these configuration names")
    ap.add_argument("--no-sample", action="store_true", help="skip the 50-pair lineage sample")
    args = ap.parse_args()
    t0 = time.time()
    cfgs = {c["name"]: c for c in configurations()}
    names = [n for n in cfgs if not args.configs or n in args.configs]
    data = {}
    for n in names:
        data[n] = load(cfgs[n], use_cache=not args.no_cache)
        print(f"  {n}: {len(data[n])} terminal results", flush=True)
    res = compute(data, cfgs)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gi = OUTDIR / ".gitignore"
    if not gi.exists():
        gi.write_text("# per-query feature cache of coevolution.py (regenerated; tens of MB)\ncache/\n")
    meta = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "extraction_version": VERSION,
            "thresholds": THRESHOLDS, "sim_chars": SIM_CAP, "large_output_controls_bytes": CTRL_BYTES,
            "no_cache": bool(args.no_cache),
            "configs": {n: {k: (str(v) if isinstance(v, Path) else v) for k, v in cfgs[n].items()} for n in names}}
    with open(OUTDIR / "coevolution.json", "w") as f:
        json.dump({"meta": meta, **res}, f, indent=1, default=str)
    if all(n in res["lineage"] for n in ("luna_full", "astra_full")):
        C.write_text(OUTDIR / "tex/coevolution_table.tex", tex_table(res, cfgs))
        if "H" in res["lineage"] and "A4" in res["lineage"]:
            C.write_text(OUTDIR / "tex/appendix/coevolution_table_codex.tex",
                         tex_table(res, cfgs, cols=("luna_full", "astra_full", "A4", "H"), wide=True))
    if all(n in res["phases"] for n in ("luna_full", "astra_full")):
        make_figures(res["phases"], res["feedback"], res["feedback_ctrl"])
        C.write_text(OUTDIR / "tex/coevolution_figure.tex", tex_figure(res))
    if not args.no_sample and "luna_full" in data and "astra_full" in data:
        C.write_text(OUTDIR / "lineage_pairs_sample.md", lineage_sample(data, cfgs))
    C.write_text(OUTDIR / "REPORT.md", report(res, cfgs, names, time.time() - t0, no_cache=args.no_cache))
    print(f"wrote {OUTDIR.relative_to(C.STARK)}/REPORT.md, coevolution.json, tex/, figures/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
