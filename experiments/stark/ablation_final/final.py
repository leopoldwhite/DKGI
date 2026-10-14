"""Tool variants on Pi (A1 Retrieval only, A2 Graph tools, A3 Shell only, A5 Graph tools + code, A6 No edges):
prompt, H6 install, isolation, one item. Uses ../ablation/ for the shared preparation. Runs are identified
only by arm / dataset / query_id; an item is finished when its result.json is terminal.
"""

import importlib.util
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STARK = HERE.parent
ABLATION = STARK / "ablation"
sys.path.insert(0, str(ABLATION))
import common as c  # noqa: E402  (shared helpers of the tool variants; loads the harness adapter)
sys.path.insert(0, str(ABLATION / "tools"))


class _ToolService:
    """The graph-tool service (A1, A2, A5) needs the ARK checkout and its packages; it is imported on first use."""

    def __getattr__(self, name):
        import service as module
        return getattr(module, name)


service = _ToolService()

_spec = importlib.util.spec_from_file_location("ablation_tool_usage", ABLATION / "tool_usage.py")
ablation_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation_runner)

base, stark, variants, launcher = c.base, c.stark, c.val.variants, c.adapter.launcher
pi_transport = c.adapter.pi_transport

ARMS = ("A1", "A2", "A3", "A5", "A6")
DATASETS = ("amazon", "mag", "prime")
SPLITS = {"val": "val", "test-0.1": "test"}          # split name -> runs/ subdirectory
# Attempt directories are visible to the model (scratch dir, A5 client path, H6 notice), so they live under
# a neutral path.
RUNS = STARK / "trials"
VISIBLE_FORBIDDEN = "ablation"
MAX_TECHNICAL_ATTEMPTS = 3                          # attempts that ended by a technical failure or crash
MAX_ATTEMPT_DIRS = 5                                # hard cap including attempts stopped by a stop signal
AGENT_TIMEOUT_SECONDS = 1800
STOP = threading.Event()                           # no new admissions / attempts
HARD_STOP = threading.Event()                      # active agent processes were asked to terminate

# Final prompt P1-R1: the base prompt P0 with its output sentence replaced exactly once.
R1_OUTPUT = (
    "The query may have several correct answers. Return 20 unique integer IDs observed in the corpus, "
    "ranked by how well the evidence supports them. Before ranking a candidate first, check it against "
    "every requirement in the query; entities that the query mentions only as a reference point are not "
    "answers unless the query asks for them. After the first candidate, add the other plausible candidates "
    "you found in order of support; they do not each need the same level of verification. Return fewer "
    "than 20 only if you cannot find that many plausible candidates. Do not invent IDs."
)
TOOL_BOUNDARY = "Corpus directory:"

# H6: every Pi arm's shellPath is a capped wrapper; the wrapped base is what ../ablation/common.prepare set.
H6_DIR = HERE / "h6"
CAP_FILTER = STARK / "harness_fixes_v2/cap_filter.py"
BASE_SHELL = {"A1": launcher.SHELL, "A2": launcher.SHELL, "A6": launcher.SHELL,
              "A3": ABLATION / "sandbox/sandbox-shell.sh", "A5": ABLATION / "a5_variant/shell.py"}
WRAPPER = {"A1": H6_DIR / "capped-default.sh", "A2": H6_DIR / "capped-default.sh",
           "A6": H6_DIR / "capped-default.sh", "A3": H6_DIR / "capped-a3.sh", "A5": H6_DIR / "capped-a5.sh"}
H6_PARAMETERS = {"h6": "single-stage head+tail bash output cap (harness_fixes_v2/cap_filter.py, unchanged)",
                 "head_bytes": 9000, "head_lines": 900, "tail_bytes": 9000, "tail_lines": 900,
                 "full_output_dir": "attempt/tmp/full_outputs"}
L3_MARKER = re.compile(r"\n\[\.\.\.truncated, ([0-9]+) chars omitted\]\Z")
L3_LIMIT_UNITS = 20000

# Provider refusal pattern (as in dci_run/core.py). A refusal ends only that item and is never resubmitted.
REFUSAL = re.compile(r"(?:^|\n)(bio_policy|content_policy_violation|safety_policy_violation):", re.I)


def now():
    return base.now()


def save(path, value):
    c.save(path, value)


def read(path):
    return c.read(path)


def redact(text):
    return base.redact(text)


def key(row):
    return f"{row['dataset']}:{row['query_id']}"


def jobkey(job):
    return f"{job['arm']}:{job['dataset']}:{job['query_id']}"


def item_dir(arm, dataset, split, query_id, runs=None):
    return Path(runs or RUNS) / arm / dataset / SPLITS[split] / str(query_id)


def result_path(job, runs=None):
    return item_dir(job["arm"], job["dataset"], job["split"], job["query_id"], runs) / "result.json"


def read_optional(path):
    try:
        return read(path)
    except (OSError, ValueError):
        return None


def is_terminal(job, runs=None):
    value = read_optional(result_path(job, runs))
    return bool(value and value.get("terminal") is True)


# ---------------------------------------------------------------- data and question sets

def load_split(split):
    """Rows, labels and candidate sets. Labels stay in this (scoring) process only."""
    if split == "val":
        rows, labels, candidates, _ = c.val.load_data()
        return rows, labels, candidates
    if split != "test-0.1":
        raise ValueError("Unsupported split: " + split)
    rows, labels, candidates = {}, {}, {}
    for dataset in DATASETS:
        for row in base.read_jsonl(STARK / "public" / dataset / "test-0.1.jsonl"):
            if row["dataset"] != dataset or row["split"] != "test-0.1" or key(row) in rows:
                raise ValueError("Invalid or duplicate test-0.1 row")
            rows[key(row)] = row
        for row in base.read_jsonl(STARK / "private" / dataset / "test-0.1.jsonl"):
            labels[f"{dataset}:{row['query_id']}"] = row["answer_ids"]
        candidates[dataset] = {int(v) for v in (STARK / "corpus" / dataset / "candidate_ids.txt").read_text().split()}
    if rows.keys() != labels.keys():
        raise ValueError("test-0.1 questions and labels differ")
    return rows, labels, candidates


def test_questions():
    out = []
    for dataset in DATASETS:
        for row in base.read_jsonl(STARK / "public" / dataset / "test-0.1.jsonl"):
            out.append({"dataset": dataset, "query_id": str(row["query_id"])})
    return out


# Val questions used by the prompt-validation rounds, when their selection files are present.
USED_VAL_SOURCES = [STARK / "prompt_val/selection.json", STARK / "prompt_val_r1/selection.json"]


def used_val_keys():
    paths = list(USED_VAL_SOURCES)
    for folder in (STARK / "prompt_val", STARK / "prompt_val_r1"):
        paths += sorted(p for p in folder.glob("supplement_*.json") if not p.name.startswith("supplement_recommendation"))
    used = set()
    for path in paths:
        if path.exists():
            used.update(key(r) for r in read(path) if isinstance(r, dict) and "query_id" in r)
    reservation = ABLATION / "reservation.json"
    if reservation.exists():
        used.update(read(reservation).get("reservation_keys", []))
    return used, [str(p) for p in paths if p.exists()]


def val_selection(rows):
    """20 unused val questions per dataset (first one per dataset is the smoke question). Written once."""
    path = HERE / "selection_val.json"
    if path.exists():
        return read(path)["questions"]
    used, sources = used_val_keys()
    chosen = []
    for dataset in DATASETS:
        available = sorted((r for r in rows.values() if r["dataset"] == dataset and key(r) not in used),
                           key=lambda r: int(r["query_id"]))
        picked = random.Random(20260928).sample(available, 20)
        chosen += [{"dataset": dataset, "query_id": r["query_id"], "smoke": i == 0} for i, r in enumerate(picked)]
    save(path, {"questions": chosen, "seed": 20260928, "excluded_sources": sources,
                "excluded_count": len(used), "created_at": now()})
    return chosen


def schedule(questions, arms, split):
    """Interleave: each question contributes one job per arm (arm order rotates per question)."""
    rng = random.Random(20260929)
    questions = list(questions)
    rng.shuffle(questions)
    jobs = []
    for q in questions:
        order = list(arms)
        rng.shuffle(order)
        jobs += [{"arm": arm, "dataset": q["dataset"], "query_id": str(q["query_id"]), "split": split} for arm in order]
    return jobs


# ---------------------------------------------------------------- prompt

def p0_prompt(row, attempt):
    return variants.prompt_for(row, attempt, "p0")


# Corpus views, directories with neutral names; files are hardlinks except
# DATA_FORMAT.txt. v1 (A3): full corpus (nodes, edges, schema, candidates); its DATA_FORMAT.txt omits the
# line suggesting Python. v2 (A6): nodes, candidate_ids.txt, edge-free schema.json and DATA_FORMAT.txt
# (from ../kb/<ds>/, whose manifest.json is deliberately not included).
VIEWS = STARK / "corpus_views"
VIEW = {"A3": VIEWS / "v1", "A6": VIEWS / "v2"}
# Working directory of the tool arms A1, A2, A5 (Pi shows it in its system prompt). Same README text as
# ../ablation/workspace/README.txt, at a neutral path.
WORKSPACE = VIEWS / "workspace"
WORKSPACE_README = "Knowledge-base data is available through the supplied tools.\n"
PYTHON_LINE = "Use shell tools or Python's json/csv modules to read these files.\n"
A3_OLD_TOOLS = "Use shell tools, rg, or Python 3 standard-library modules to search and read them."
A3_NEW_TOOLS = "Use shell tools and rg to search and read them."


def view_plan(arm, dataset):
    """{relative path: (source path, text or None)}; None = hardlink of source."""
    source = STARK / "corpus" / dataset
    plan = {"candidate_ids.txt": (source / "candidate_ids.txt", None)}
    plan.update({f"nodes/{p.name}": (p, None) for p in sorted((source / "nodes").glob("*.jsonl"))})
    if arm == "A3":
        plan.update({f"edges/{p.name}": (p, None) for p in sorted((source / "edges").glob("*.tsv"))})
        plan["schema.json"] = (source / "schema.json", None)
        text = (source / "DATA_FORMAT.txt").read_text()
        if text.count(PYTHON_LINE) != 1:
            raise ValueError("Expected one Python line in DATA_FORMAT.txt")
        plan["DATA_FORMAT.txt"] = (source / "DATA_FORMAT.txt", text.replace(PYTHON_LINE, "", 1))
    elif arm == "A6":
        kb = STARK / "kb" / dataset
        plan["schema.json"] = (kb / "schema.json", None)
        plan["DATA_FORMAT.txt"] = (kb / "DATA_FORMAT.txt", None)
    else:
        raise ValueError("No corpus view for " + arm)
    return plan


def build_view(arm, dataset, create=True):
    """Create (idempotent) and verify one view: exact file set, inode identity, DATA_FORMAT content."""
    target = VIEW[arm] / dataset
    plan = view_plan(arm, dataset)
    for rel, (source, text) in plan.items():
        path = target / rel
        if create and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                os.link(source, path)
            else:
                path.write_text(text)
                path.chmod(0o444)
        if text is None and not os.path.samefile(source, path):
            raise ValueError(f"View file is not a hardlink of its source: {path}")
        if text is not None and path.read_text() != text:
            raise ValueError(f"View DATA_FORMAT.txt differs from the expected text: {path}")
    present = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    if present != set(plan):
        raise ValueError(f"View {target} has unexpected files: {sorted(present ^ set(plan))[:5]}")
    return {"view": str(target), "files": len(plan), "hardlinks": sum(t is None for _, t in plan.values()),
            "edges": sum(r.startswith("edges/") for r in plan)}


def build_workspace(create=True):
    path = WORKSPACE / "README.txt"
    if create and not path.exists():
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        path.write_text(WORKSPACE_README)
    if path.read_text() != WORKSPACE_README or (ABLATION / "workspace/README.txt").read_text() != WORKSPACE_README:
        raise ValueError("Workspace README differs")
    if {p.name for p in WORKSPACE.iterdir()} != {"README.txt"}:
        raise ValueError("Workspace must contain only README.txt")
    return str(WORKSPACE)


def corpus_for(arm, dataset):
    if arm in VIEW:
        return VIEW[arm] / dataset
    if arm in ("A1", "A2", "A5"):
        return WORKSPACE
    return STARK / "corpus" / dataset


def arm_paragraph(row, attempt, arm):
    corpus = corpus_for(arm, row["dataset"])
    text = (ABLATION / "prompts" / f"{arm}.txt").read_text()
    if arm == "A3":
        text = variants.replace_once(text, A3_OLD_TOOLS, A3_NEW_TOOLS)
    return text.format(corpus=corpus, scratch=Path(attempt) / "tmp", client_repr=repr(str(attempt)),
                       socket_repr=repr(str(Path(attempt) / "kb.sock")))


def prompt(row, attempt, arm):
    """P1-R1 task text with the arm's tool paragraph (from ../ablation/prompts/<arm>.txt)."""
    original = dict(row, corpus_dir=str(STARK / "corpus" / row["dataset"]))  # P0 is built on the original row
    text = variants.replace_once(p0_prompt(original, attempt), variants.OLD_OUTPUT, R1_OUTPUT)
    if text.count(TOOL_BOUNDARY) != 1:
        raise ValueError("Expected one tool paragraph boundary")
    prefix = text.partition(TOOL_BOUNDARY)[0]
    return prefix + arm_paragraph(row, attempt, arm)


def prepare_base(attempt, row, arm, protocol):
    """The steps of ../ablation/common.prepare, with each arm's working directory from corpus_for()
    (common.prepare hardcodes ../ablation/workspace and ../kb/<ds>, whose paths the model would see)."""
    import shutil
    chosen = dict(row, corpus_dir=str(corpus_for(arm, row["dataset"])))
    settings = dict(protocol, pi_max_turns=300)
    command, env = c.adapter.prepare_attempt(attempt, chosen, settings, "pi", prompt(row, attempt, arm))
    config = read(attempt / "agent/settings.json")
    config["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 15000, "maxDelayMs": 120000}
    if arm == "A3":
        local = attempt / "sandbox"
        (local / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ABLATION / "sandbox/exec_only.py", local / "exec_only.py")
        for name in "rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash".split():
            destination = local / "bin" / name
            source = shutil.which(name, path=f"{base.CODEX_ROOT}/codex-path:/usr/bin:/bin")
            if destination.is_symlink() and destination.readlink() != Path(source):
                destination.unlink()
            if not destination.exists():
                destination.symlink_to(source)
        config["shellPath"] = str(ABLATION / "sandbox/sandbox-shell.sh")
        boundary = command.index("--")
        command[boundary:boundary] = ["--read", config["shellPath"]]
    save(attempt / "agent/settings.json", config)
    if arm in ("A1", "A2"):
        command[command.index("--tools") + 1] = ""
        command += ["--extra-arg=--no-tools", f"--extra-arg=-e {ABLATION / 'tools/extension.mjs'}"]
        boundary = command.index("--")
        command[boundary:boundary] = ["--read", str(ABLATION / "tools/extension.mjs")]
        env.update(DCI_KB_SOCKET=str(attempt / "kb.sock"), DCI_KB_ARM=arm)
    if arm == "A5":
        shutil.copyfile(ABLATION / "tools/kbtools.py", attempt / "_kbtools_base.py")
        shutil.copyfile(ABLATION / "a5_variant/kbtools.py", attempt / "kbtools.py")
        config = read(attempt / "agent/settings.json")
        config["shellPath"] = str(ABLATION / "a5_variant/shell.py")
        save(attempt / "agent/settings.json", config)
        boundary = command.index("--")
        command[boundary:boundary] = ["--read", config["shellPath"]]
    save(attempt / "command.json", command)
    save(attempt / "arm.json", {"arm": arm, "harness": "pi", "cwd": chosen["corpus_dir"],
                               "expected_tools": sorted(c.EXPECTED_TOOLS[arm])})
    return command, env, Path(chosen["corpus_dir"])


def prepare(attempt, row, arm, protocol):
    """Shared preparation + H6 install + the outer isolation preflight run again on the final command."""
    attempt = Path(attempt)
    # The STaRK harness adapter (prompt_val/harness.py) accepts only rows labelled "val"; this runner gates
    # test-0.1 itself (run.py). The label is changed only on the copy passed to the adapter; the real split is
    # recorded in arm_final.json.
    harness_row = dict(row, split="val")
    if arm in VIEW:
        build_view(arm, row["dataset"], create=False)
    if arm in ("A1", "A2", "A5"):
        build_workspace(create=False)
    command, env, cwd = prepare_base(attempt, harness_row, arm, protocol)
    if cwd != corpus_for(arm, row["dataset"]):
        raise ValueError(f"{arm} working directory is not its configured corpus: {cwd}")
    visible = [(attempt / "prompt.txt").read_text(), str(cwd), str(attempt)]
    if any(VISIBLE_FORBIDDEN in v.lower() for v in visible):
        raise ValueError("A model-visible path or the prompt contains a forbidden substring")
    settings_path = attempt / "agent/settings.json"
    settings = read(settings_path)
    if settings.get("shellPath") != str(BASE_SHELL[arm]):
        raise ValueError(f"Unexpected Pi shellPath before H6 for {arm}: {settings.get('shellPath')}")
    settings["shellPath"] = str(WRAPPER[arm])
    save(settings_path, settings)
    command = list(command)
    boundary = command.index("--")
    command[boundary:boundary] = ["--read", str(WRAPPER[arm]), "--read", str(H6_DIR / "cap_select.py"),
                                  "--read", str(CAP_FILTER)]
    launcher.preflight(command, env, attempt, dict(harness_row, corpus_dir=str(cwd)))
    save(attempt / "command.json", command)
    record = read(attempt / "harness_fixes.json")
    record["h6"] = {**H6_PARAMETERS, "wrapper": str(WRAPPER[arm]), "base_shell": str(BASE_SHELL[arm])}
    save(attempt / "harness_fixes.json", record)
    save(attempt / "arm_final.json", {"arm": arm, "dataset": row["dataset"], "query_id": row["query_id"],
                                      "split": row["split"], "prompt_base": "P1-R1", "cwd": str(cwd),
                                      "shellPath": str(WRAPPER[arm]), "h6_base_shell": str(BASE_SHELL[arm])})
    return command, env, Path(cwd)


def isolation(attempt, row, arm, command, env, cwd):
    """Shared isolation probes + final-run paths + H6 wrapper liveness."""
    checks = c.isolation_check(attempt, row, arm, command, env, cwd)
    outer = command[:command.index("--") + 1]
    denied = [RUNS, STARK / "trials", HERE, ABLATION, HERE / "runs_h", STARK / "pi_full_test", HERE / "selection_val.json",
              *([STARK / "corpus", STARK / "corpus2", STARK / "kb", *VIEW.values()] if arm == "A3" else []),
              *([STARK / "corpus", STARK / "corpus2", STARK / "kb", VIEW["A3"], VIEW["A6"] / "manifest.json"]
                if arm == "A6" else []),
              Path(attempt).parent / "result.json", STARK / "private", STARK / "public"]
    code = "import os\nfor path in " + repr([str(p) for p in denied]) + ":\n"
    code += (" try:\n  fd=os.open(path,os.O_RDONLY)\n except (PermissionError, FileNotFoundError):\n  continue\n"
             " else:\n  os.close(fd)\n  raise AssertionError('Readable denied path: '+path)\n")
    probe = subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", code], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise RuntimeError("Final-run isolation probe failed: " + probe.stderr)
    shell = str(WRAPPER[arm])
    live = subprocess.run(outer + [shell, "-c", "echo h6-ok; exit 5"], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)
    if live.returncode != 5 or live.stdout.strip() != "h6-ok":
        raise RuntimeError("H6 wrapper probe failed: " + live.stdout + live.stderr)
    if arm == "A3":
        denied_run = subprocess.run(outer + [shell, "-c", "python3 -c 1"], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=60)
        if denied_run.returncode == 0:
            raise RuntimeError("A3 capped shell allowed python3")
        checks["h6_a3_interpreter"] = "denied"
    checks.update(final_run_paths="denied", h6_wrapper="passed")
    save(Path(attempt) / "ablation_preflight.json", checks)
    return checks


# ---------------------------------------------------------------- post-run audits

def js_units(text):
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def h6_audit(attempt):
    """Every provider request: L3 truncation markers and model-visible bash output sizes (UTF-16 units)."""
    path = Path(attempt) / "artifacts/events.jsonl"
    names, outputs, requests = {}, {}, 0
    with path.open("rb") as source:
        for raw in source:
            if b'"provider_request_context"' not in raw[:200]:
                continue
            event = json.loads(raw)
            if event.get("type") != "provider_request_context":
                continue
            requests += 1
            for item in c.adapter.pi_context.provider_items(event["payload"]):
                if item.get("type") == "function_call":
                    names[item["call_id"]] = item.get("name")
                elif item.get("type") == "function_call_output":
                    text = item.get("output") or ""
                    if not isinstance(text, str):
                        text = json.dumps(text)
                    outputs[item["call_id"]] = text
    bash = [t for k, t in outputs.items() if names.get(k) == "bash"]
    return {"requests": requests, "tool_outputs": len(outputs), "bash_outputs": len(bash),
            "l3_truncation_markers": sum(bool(L3_MARKER.search(t)) for t in outputs.values()),
            "l3_markers_bash": sum(bool(L3_MARKER.search(t)) for t in bash),
            "l3_markers_other_tools": {n: sum(bool(L3_MARKER.search(t)) for k, t in outputs.items() if names.get(k) == n)
                                       for n in sorted({names.get(k) for k in outputs} - {"bash"}, key=str)
                                       if any(L3_MARKER.search(t) for k, t in outputs.items() if names.get(k) == n)},
            "bash_max_utf16_units": max((js_units(t) for t in bash), default=0),
            "bash_over_limit": sum(js_units(t) > L3_LIMIT_UNITS for t in bash),
            "h6_capped_outputs": sum("[... harness output limit" in t for t in bash),
            "full_output_files": len(list((Path(attempt) / "tmp/full_outputs").glob("out-*.log")))}


def visible_path_leaks(attempt):
    """Occurrences of an ablation directory path in any archived provider request (prompt, tools, outputs)."""
    count = 0
    with (Path(attempt) / "artifacts/events.jsonl").open("rb") as source:
        for raw in source:
            if b'"provider_request_context"' in raw[:200]:
                count += len(re.findall(rb"stark/ablation", raw))
    return count


def refusal(result):
    return result.get("status") == "failed" and result.get("error_phase") == "agent" and bool(
        REFUSAL.search(result.get("error", "")))


def has_final(attempt):
    attempt = Path(attempt)
    path = attempt / "artifacts/final.txt"
    return path.exists() and bool(path.read_text(errors="replace").strip())


# ---------------------------------------------------------------- one attempt / one item

def run_attempt(job, row, gold, candidates, protocol, attempt):
    arm = job["arm"]
    result = {**job, "harness": "pi", "prompt_base": "P1-R1", "status": "running", "terminal": False,
              "attempt_path": str(attempt), "started_at": now(), "metrics": {}}
    save(attempt / "result.json", result)
    server, phase, started = None, "preflight", time.monotonic()
    try:
        if arm in ("A1", "A2", "A5"):
            server = service.start(attempt, row["dataset"], arm)
        command, env, cwd = prepare(attempt, row, arm, protocol)
        result["preflight"] = isolation(attempt, row, arm, command, env, cwd)
        if STOP.is_set():
            raise InterruptedError("Queue stopped before model call")
        phase = "agent"
        if server:
            server.phase = "agent"
        result["returncode"] = base.run_process(command, env, attempt, cwd, AGENT_TIMEOUT_SECONDS)
        if result["returncode"]:
            raise RuntimeError("Agent exit " + str(result["returncode"]))
        result.update(pi_transport.collect(attempt, "pi"))
        phase = "provider_tools"
        result["provider_tools"] = c.validate_tools(attempt, arm)
        phase = "scoring"
        result["metrics"] = stark.score(result["prediction"], gold, candidates)
        result["returned_id_count"] = result["metrics"]["returned_count"]
        result["status"] = "scored"
    except Exception as error:
        result.update(status="parse_failed" if phase == "scoring" else "failed", error_phase=phase,
                      error_type=type(error).__name__, error=redact(error))
        state = attempt / "artifacts/state.json"
        if state.exists():
            try:
                result["error"] += " " + redact(read(state).get("error", ""))
            except Exception:
                pass
        if phase in ("agent", "provider_tools", "scoring"):
            try:
                result.update({k: v for k, v in pi_transport.collect(attempt, "pi", require_completed=False).items()
                               if k not in result or phase == "agent"})
            except Exception:
                pass
        if HARD_STOP.is_set() and phase == "agent" or isinstance(error, InterruptedError):
            result["status"] = "interrupted"
        result["retry_reason"] = None if result["status"] == "interrupted" else stark.retry_reason(result, attempt)
        # Load-induced preflight failures (the sandbox probes did not finish in time) made no model call:
        # retry them and do not count the attempt.
        if result.get("error_phase") == "preflight" and result.get("error_type") in ("TimeoutExpired", "InterruptedError"):
            result["retry_reason"] = "preflight_timeout"
            result["uncounted_attempt"] = True
        result["provider_refusal"] = refusal(result)
        result["rate_limit_exhausted"] = any(w in result["error"].lower() for w in ("429", "rate limit", "concurrency limit"))
    finally:
        if server:
            server.shutdown()
            server.server_close()
    for name, method in (("tool_usage", lambda: ablation_runner.issued(attempt, arm)), ("h6_audit", lambda: h6_audit(attempt))):
        try:
            if (attempt / "artifacts/events.jsonl").exists():
                result[name] = method()
        except Exception as error:
            result[name + "_error"] = redact(error)
    result["attempt_cost_usd"] = result.get("observed_cost_usd_lower_bound")
    result.update(finished_at=now(), wall_time_seconds=time.monotonic() - started)
    save(attempt / "result.json", result)
    return result


def attempts(directory):
    return sorted(p for p in Path(directory).glob("attempt_*") if p.is_dir())


def counted_attempts(directory):
    """Technical attempts consumed; an attempt interrupted by a stop signal does not count."""
    n = 0
    for attempt in attempts(directory):
        value = read_optional(attempt / "result.json")
        if not (value and (value.get("status") == "interrupted" or value.get("uncounted_attempt"))):
            n += 1
    return n


def finalize(job, directory, last, reason=None):
    history = [r for r in (read_optional(a / "result.json") for a in attempts(directory)) if r]
    out = dict(last)
    out.update(arm=job["arm"], dataset=job["dataset"], query_id=job["query_id"], split=job["split"],
               terminal=True, attempt_count=len(history), finished_at=now(),
               cost_usd=base.total_cost(history) if len(history) == len(list(directory.glob("attempt_*"))) else None,
               total_input_tokens=sum(r.get("input_tokens") or 0 for r in history),
               total_output_tokens=sum(r.get("output_tokens") or 0 for r in history),
               total_wall_seconds=sum(r.get("wall_time_seconds") or 0 for r in history),
               tool_calls=last.get("tool_calls") or 0,
               tool_calls_all_attempts=sum(r.get("tool_calls") or 0 for r in history),
               attempts=[{"attempt": Path(r["attempt_path"]).name, "status": r.get("status"),
                          "error_type": r.get("error_type"), "retry_reason": r.get("retry_reason")} for r in history])
    if out.get("status") != "scored":
        out["metrics"] = out.get("metrics") or {}
        out["failure_kind"] = reason or ("provider_refusal" if out.get("provider_refusal") else "technical")
    save(directory / "result.json", out)
    return out


def try_recover(job, directory, previous, gold, candidates):
    """A crashed attempt whose agent already finished is scored without a new model call."""
    attempt = Path(previous.get("attempt_path", ""))
    if not attempt.is_dir() or not has_final(attempt):
        return None
    try:
        recovered = pi_transport.collect(attempt, "pi")
        tools = c.validate_tools(attempt, job["arm"])
        metrics = stark.score(recovered["prediction"], gold, candidates)
    except Exception:
        return None
    result = {**previous, **recovered, "provider_tools": tools, "metrics": metrics,
              "returned_id_count": metrics["returned_count"], "status": "scored",
              "recovered_without_new_model_call": True}
    for name, method in (("tool_usage", lambda: ablation_runner.issued(attempt, job["arm"])), ("h6_audit", lambda: h6_audit(attempt))):
        try:
            result[name] = method()
        except Exception as error:
            result[name + "_error"] = redact(error)
    result["attempt_cost_usd"] = result.get("observed_cost_usd_lower_bound")
    save(attempt / "result.json", result)
    return finalize(job, directory, result)


def execute(job, row, gold, candidates, protocol, on_rate_limit=None):
    """Run one item to a terminal state (or leave it non-terminal only if the queue is stopping)."""
    directory = result_path(job).parent
    with base.job_lock(directory):
        previous = read_optional(directory / "result.json")
        if previous and previous.get("terminal") is True:
            return previous
        if previous and previous.get("status") == "running":
            recovered = try_recover(job, directory, previous, gold, candidates)
            if recovered:
                return recovered
        while True:
            dirs = attempts(directory)
            last = read_optional(dirs[-1] / "result.json") if dirs else None
            if dirs and (counted_attempts(directory) >= MAX_TECHNICAL_ATTEMPTS or len(dirs) >= MAX_ATTEMPT_DIRS):
                return finalize(job, directory, last or {**job, "status": "failed", "error": "attempt crashed"},
                                "attempts_exhausted")
            if STOP.is_set():
                if previous or dirs:
                    save(directory / "result.json", {**(last or previous or job), "terminal": False, "status": "interrupted"})
                return None
            attempt = directory / f"attempt_{len(dirs) + 1:03d}"
            attempt.mkdir(parents=True)
            save(directory / "result.json", {**job, "status": "running", "terminal": False,
                                            "attempt_path": str(attempt), "started_at": now()})
            result = run_attempt(job, row, gold, candidates, protocol, attempt)
            if result["status"] == "scored":
                return finalize(job, directory, result)
            if result["status"] == "interrupted":
                save(directory / "result.json", {**result, "terminal": False})
                return None
            if result.get("provider_refusal"):
                return finalize(job, directory, result, "provider_refusal")
            if result.get("retry_reason") is None:
                return finalize(job, directory, result, "non_retryable_" + str(result.get("error_phase")))
            if counted_attempts(directory) >= MAX_TECHNICAL_ATTEMPTS:
                return finalize(job, directory, result, "attempts_exhausted")
            if len(attempts(directory)) >= 10:  # bound uncounted preflight-timeout retries
                return finalize(job, directory, result, "preflight_timeouts_exhausted")
            if result.get("rate_limit_exhausted") and on_rate_limit:
                on_rate_limit(job)
            delay = min(300, 60 * 2 ** (counted_attempts(directory) - 1))
            print(f"RETRY {jobkey(job)} reason={result['retry_reason']} in {delay}s", flush=True)
            if STOP.wait(delay):
                save(directory / "result.json", {**result, "terminal": False})
                return None


def kill_active_agents():
    """Second stop signal: ask each agent supervisor (own session) to terminate its process group."""
    pids = set()
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            pids.update(int(p) for p in (task / "children").read_text().split())
        except OSError:
            pass
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return sorted(pids)
