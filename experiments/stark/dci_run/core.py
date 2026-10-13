"""Full DKGI on STaRK: Pi (context level L3, single-stage output cap H6) or Codex, final prompt (P1-R1), any split.

Uses the harness in harness_fixes/adapter.py (Pi transport, Landlock launcher, preflight, observability)
and the STaRK scoring code in ../task.py. Runs are identified only by group/dataset/split/query_id;
a question is done when its result.json is terminal.

Result layout: <root>/<runs_dir>/<group>/<dataset>/<split_dir>/<query_id>/{attempt_NNN/, result.json}
(split_dir is "test" for both test and test-0.1, since test-0.1 is a subset of test).
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
import traceback

STARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARK / "harness_fixes"))
import adapter  # noqa: E402  Pi transport, Landlock launcher, observability

runner = adapter.runner
variants, harness, stark, base, launcher = runner.variants, runner.harness, runner.stark, runner.base, adapter.launcher
H6_DIR = STARK / "harness_fixes_v2"
H6_SHELL = H6_DIR / "capped-shell.sh"
H6_FILES = ("capped-shell.sh", "cap_filter.py")
DATASETS = ("amazon", "mag", "prime")
SPLITS = ("train", "val", "test", "test-0.1", "human_generated_eval")
MAX_ATTEMPTS = 3
P1R1_OUTPUT = (
    "The query may have several correct answers. Return 20 unique integer IDs observed in the corpus, "
    "ranked by how well the evidence supports them. Before ranking a candidate first, check it against "
    "every requirement in the query; entities that the query mentions only as a reference point are not "
    "answers unless the query asks for them. After the first candidate, add the other plausible candidates "
    "you found in order of support; they do not each need the same level of verification. Return fewer "
    "than 20 only if you cannot find that many plausible candidates. Do not invent IDs."
)
REFUSAL = re.compile(r"(?:^|\n)(bio_policy|content_policy_violation|safety_policy_violation):", re.I)
L3_MARK = re.compile(r"\n\[\.\.\.truncated, ([0-9]+) chars omitted\]\Z")
STOP = threading.Event()
LOCK = threading.Lock()


def harness_for(group):
    return group.split("_", 1)[0]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def redact(text):
    key = os.environ.get("MODEL_API_KEY")
    return text.replace(key, "[redacted]") if key else text


# ---------------------------------------------------------------- data

def load_questions(split, datasets=DATASETS):
    rows, labels, candidates = [], {}, {}
    for ds in datasets:
        for row in read_jsonl(STARK / "public" / ds / f"{split}.jsonl"):
            rows.append(row)
        for row in read_jsonl(STARK / "private" / ds / f"{split}.jsonl"):
            labels[(ds, split, str(row["query_id"]))] = row["answer_ids"]  # split in the key: ids repeat across splits
        candidates[ds] = {int(v) for v in (STARK / "corpus" / ds / "candidate_ids.txt").read_text().split()}
    return rows, labels, candidates


def sample(rows, per_dataset, seed):
    """Deterministic per-dataset sample: sort by integer query_id, then random.Random(seed).sample."""
    out = []
    for ds in DATASETS:
        pool = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: int(r["query_id"]))
        if pool:
            out += random.Random(seed).sample(pool, min(per_dataset, len(pool)))
    return out


def split_dir(split):
    return "test" if split in ("test", "test-0.1") else split


def result_path(root, runs_dir, group, row):
    return Path(root) / runs_dir / group / row["dataset"] / split_dir(row["split"]) / str(row["query_id"]) / "result.json"


# ---------------------------------------------------------------- prompt and harness

def prompt_for(row, attempt):
    prompt = variants.prompt_for(row, attempt, "p0")
    return variants.replace_once(prompt, variants.OLD_OUTPUT, P1R1_OUTPUT)


ASTRA_MODEL = {"id": "gpt-6-astra", "name": "GPT-6 Astra", "api": "openai-responses", "reasoning": True,
               "input": ["text"], "contextWindow": 272000, "maxTokens": 128000,
               "cost": {"input": 10.0, "output": 50.0, "cacheRead": 1.0, "cacheWrite": 12.5}}
OTHER_MODELS = {"gpt-6-astra": ASTRA_MODEL}


def model_name():
    """Backbone for OpenAI-compatible harnesses; set per preset by run.py via DCI_MODEL (default gpt-6-luna)."""
    return os.environ.get("DCI_MODEL", "gpt-6-luna")


def use_model(common):
    name = model_name()
    if name != common["model"]:
        definition = OTHER_MODELS[name]
        harness.LUNA_MODEL = definition        # the val harness checks and writes this definition for Pi
        common["model"] = name
    return common


def model_cost(result, harness_name):
    name = model_name()
    if name == "gpt-6-luna":
        return runner.cost(result, harness_name)
    if not result.get("usage_complete"):
        return None
    rate = {k: v / 1e6 for k, v in OTHER_MODELS[name]["cost"].items()}
    raw = result["usage_raw"]
    if harness_name == "pi":
        return raw["input"] * rate["input"] + raw["cacheRead"] * rate["cacheRead"] + \
            raw["cacheWrite"] * rate["cacheWrite"] + raw["output"] * rate["output"]
    return (raw["input_tokens"] - raw["cached_input_tokens"]) * rate["input"] + \
        raw["cached_input_tokens"] * rate["cacheRead"] + raw["output_tokens"] * rate["output"]


def protocol_for(harness_name):
    common = use_model(runner.protocol())
    if harness_name == "pi":
        common["pi_max_turns"] = 300
    return common


def install_h6(attempt, command, env, row):
    """Point Pi's bash at the single-stage head+tail cap (H6) and grant read access to its two files."""
    attempt = Path(attempt)
    settings_path = attempt / "agent/settings.json"
    settings = json.loads(settings_path.read_text())
    if settings.get("shellPath") != str(launcher.SHELL):
        raise ValueError("Unexpected Pi shellPath before H6")
    settings["shellPath"] = str(H6_SHELL)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    command = list(command)
    boundary = command.index("--")
    command[boundary:boundary] = [v for name in H6_FILES for v in ("--read", str(H6_DIR / name))]
    launcher.preflight(command, env, attempt, row)
    save(attempt / "command.json", command)
    record = json.loads((attempt / "harness_fixes.json").read_text())
    record["h6"] = {"shell": str(H6_SHELL), "head_bytes": 9000, "head_lines": 900,
                    "tail_bytes": 9000, "tail_lines": 900}
    save(attempt / "harness_fixes.json", record)
    return command, env


def prepare(attempt, row, protocol, harness_name, prompt):
    # The val harness only admits split == "val"; it uses the row only for corpus_dir and the query.
    admitted = dict(row, split="val")
    command, env = adapter.prepare_attempt(attempt, admitted, protocol, harness_name, prompt)
    if harness_name == "pi":
        command, env = install_h6(attempt, command, env, admitted)
    return command, env


def h6_audit(attempt):
    """Check every Pi request: no L3 truncation marker and every bash output <= 20,000 UTF-16 units."""
    source = Path(attempt) / "artifacts/events.jsonl"
    audit = {"requests": 0, "bash_outputs": 0, "l3_truncated": 0, "oversize_bash_outputs": 0, "max_bash_units": 0}
    if not source.exists():
        return {**audit, "passed": False, "missing_events": True}
    for raw in open(source, "rb"):
        if b'"provider_request_context"' not in raw[:200]:
            continue
        items = json.loads(raw)["payload"]["input"]
        audit["requests"] += 1
        names = {i["call_id"]: i.get("name") for i in items if i.get("type") == "function_call"}
        for i in items:
            if i.get("type") != "function_call_output" or names.get(i["call_id"]) != "bash":
                continue
            text = i.get("output") or ""
            units = len(text.encode("utf-16-le", "surrogatepass")) // 2
            audit["bash_outputs"] += 1
            audit["l3_truncated"] += bool(L3_MARK.search(text))
            audit["oversize_bash_outputs"] += units > 20000
            audit["max_bash_units"] = max(audit["max_bash_units"], units)
    # Requests repeat earlier outputs; counts are per request occurrence, which is enough for pass/fail.
    audit["passed"] = audit["l3_truncated"] == 0 and audit["oversize_bash_outputs"] == 0
    return audit


def is_refusal(result):
    return result.get("status") == "failed" and result.get("error_phase") == "agent" and bool(
        REFUSAL.search(result.get("error", "")))


# ---------------------------------------------------------------- one question

def run_attempt(job, attempt, gold, candidates, protocol):
    group, harness_name, row = job["group"], job["harness"], job["row"]
    if "ablation" in str(attempt).lower():
        raise ValueError("Attempt path must not reveal the experiment: " + str(attempt))
    result = {"group": group, "harness": harness_name, "dataset": row["dataset"], "split": row["split"],
              "query_id": str(row["query_id"]), "attempt_path": str(attempt), "status": "running",
              "started_at": now(), "terminal": False, "metrics": {}}
    save(attempt / "result.json", result)
    phase, started = "preflight", time.monotonic()
    env = {}
    try:
        prompt = prompt_for(row, attempt)
        # The question itself may contain the word (e.g. "laser ablation" in MAG); check everything else.
        if "ablation" in prompt.replace(str(row.get("query", "")), "").lower():
            raise ValueError("Prompt must not reveal the experiment")
        command, env = prepare(attempt, row, protocol, harness_name, prompt)
        phase = "agent"
        code = base.run_process(command, env, attempt, Path(row["corpus_dir"]), protocol["timeout_seconds"])
        result["returncode"] = code
        if code:
            raise RuntimeError(f"Agent exit {code}; see attempt logs")
        result.update(adapter.collect(attempt, harness_name))
        phase = "scoring"
        result["metrics"] = stark.score(result["prediction"], gold, candidates)
        result["returned_id_count"] = result["metrics"]["returned_count"]
        phase = "behavior"
        result.update(harness.behavior(attempt, harness_name))
        result["status"] = "scored"
    except Exception as error:  # the item fails; the queue continues
        result.update(status="parse_failed" if phase == "scoring" else "failed", error_phase=phase,
                      error_type=type(error).__name__, error=redact(str(error)))
        if phase == "agent":
            try:
                result.update(adapter.collect(attempt, harness_name, require_completed=False))
            except Exception:
                pass
            state_path = attempt / "artifacts/state.json"
            if harness_name == "pi" and state_path.exists():
                result["error"] = redact(result["error"] + "\n" + str(json.loads(state_path.read_text()).get("error", "")))
    if harness_name == "pi":
        try:
            result["h6_audit"] = h6_audit(attempt)
        except Exception as error:
            result["h6_audit"] = {"passed": False, "error": str(error)}
    result.update(finished_at=now(), wall_time_seconds=time.monotonic() - started)
    try:
        result["attempt_cost_usd"] = model_cost(result, harness_name)
    except Exception:
        result["attempt_cost_usd"] = None
    result["retry_reason"] = None if is_refusal(result) or result["status"] == "scored" else stark.retry_reason(result, attempt)
    save(attempt / "result.json", result)
    return result


def run_question(job, labels, candidates, protocols, root, runs_dir, extra_attempts=0):
    row = job["row"]
    final = result_path(root, runs_dir, job["group"], row)
    directory = final.parent
    if final.exists() and json.loads(final.read_text()).get("terminal"):
        return json.loads(final.read_text())
    gold = labels[(row["dataset"], row["split"], str(row["query_id"]))]
    history = []
    number = max((int(p.name.split("_")[-1]) for p in directory.glob("attempt_*")), default=0)
    while not STOP.is_set():
        if number >= MAX_ATTEMPTS + extra_attempts:
            break
        number += 1
        attempt = directory / f"attempt_{number:03d}"
        attempt.mkdir(parents=True, exist_ok=True)
        result = run_attempt(job, attempt, gold, candidates[row["dataset"]], protocols[job["harness"]])
        history.append(result)
        if STOP.is_set() and result["status"] != "scored":
            return result  # interrupted: not terminal, resumes later
        if result["status"] == "scored" or not result.get("retry_reason"):
            break
        delay = random.uniform(.5, 1) * min(120, 10 * 2 ** (number - 1))
        print(f"RETRY {job['group']}:{row['dataset']}:{row['query_id']} attempt={number} "
              f"delay={delay:.0f}s reason={result['retry_reason']}", flush=True)
        STOP.wait(delay)
    if STOP.is_set() and not history:
        return None
    attempts = sorted(directory.glob("attempt_*/result.json"))
    past = [json.loads(p.read_text()) for p in attempts]
    last = past[-1] if past else {"status": "failed", "error": "no attempt result", "metrics": {}}
    summary = {**last, "terminal": True, "attempt_count": len(list(directory.glob("attempt_*"))),
               "cost_usd": sum(r.get("attempt_cost_usd") or 0 for r in past),
               "total_input_tokens": sum(r.get("input_tokens") or 0 for r in past),
               "total_output_tokens": sum(r.get("output_tokens") or 0 for r in past),
               "total_wall_seconds": sum(r.get("wall_time_seconds") or 0 for r in past),
               "provider_refusal": is_refusal(last)}
    save(final, summary)
    return summary


# ---------------------------------------------------------------- queue

def workers_from(control, default):
    try:
        return max(0, int(json.loads(Path(control).read_text())["workers"]))
    except Exception:
        return default


def run_queue(jobs, labels, candidates, root, runs_dir, workers, log, control_name="control.json", extra_attempts=0):
    """Run jobs with per-item isolation. <root>/control.json {"workers": N} adjusts admission live."""
    root = Path(root)
    control = root / control_name
    save(control, {"workers": workers})  # the launch argument wins; edit the file later to change it live
    protocols = {h: protocol_for(h) for h in {j["harness"] for j in jobs}}
    pending = [j for j in jobs if not (result_path(root, runs_dir, j["group"], j["row"]).exists() and
                                        json.loads(result_path(root, runs_dir, j["group"], j["row"]).read_text()).get("terminal"))]
    done = len(jobs) - len(pending)
    log(f"{now()} start: {len(jobs)} jobs, {done} already terminal, {len(pending)} to run")
    active = {}
    with ThreadPoolExecutor(max_workers=64) as pool:
        while (pending or active) and not (STOP.is_set() and not active):
            limit = workers_from(control, workers)
            while pending and len(active) < limit and not STOP.is_set():
                job = pending.pop(0)
                active[pool.submit(run_question, job, labels, candidates, protocols, root, runs_dir, extra_attempts)] = job
            if not active:
                STOP.wait(10)
                continue
            finished, _ = wait(list(active), timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                name = f"{job['group']}:{job['row']['dataset']}:{job['row']['query_id']}"
                try:
                    result = future.result()
                    if result is not None:
                        done += bool(result.get("terminal"))
                        log(f"{now()} {result.get('status')}: {name} hit@1={result.get('metrics', {}).get('hit_at_1')} ({done}/{len(jobs)})")
                except BaseException:
                    detail = traceback.format_exc(limit=5)
                    log(f"{now()} worker error (item skipped, queue continues): {name}\n{redact(detail)}")
                    save(result_path(root, runs_dir, job["group"], job["row"]),
                         {"group": job["group"], "harness": job["harness"], "dataset": job["row"]["dataset"],
                          "split": job["row"]["split"], "query_id": str(job["row"]["query_id"]),
                          "status": "failed", "terminal": True, "error_phase": "worker",
                          "error": redact(detail), "metrics": {}})
                    done += 1
    log(f"{now()} queue ended: {done}/{len(jobs)} terminal; stop_requested={STOP.is_set()}")
    return done


def interleave(rows, groups, seed=20260928):
    """Jobs ordered question by question with the group order shuffled per question."""
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    jobs = []
    for row in order:
        gs = list(groups)
        rng.shuffle(gs)
        jobs += [{"group": g, "harness": harness_for(g), "row": row} for g in gs]
    return jobs


def technical_failures(jobs, root, runs_dir):
    """Terminal non-scored results that are infrastructure failures (not refusals, not scoring/parse errors)."""
    out = []
    for job in jobs:
        path = result_path(root, runs_dir, job["group"], job["row"])
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        if r.get("terminal") and r.get("status") == "failed" and not r.get("provider_refusal") \
                and not r.get("tail_retry_done"):
            out.append((job, path, r))
    return out


def tail_retry(jobs, labels, candidates, root, runs_dir, workers, log, control_name="control.json", extra_attempts=3):
    """One tail pass after the main queue: technically failed items get up to `extra_attempts` more attempts.
    The first-pass terminal result is kept as result_first_pass.json. Runs once per item."""
    failed = technical_failures(jobs, root, runs_dir)
    if not failed or STOP.is_set():
        return 0
    log(f"{now()} tail retry: {len(failed)} technically failed items get up to {extra_attempts} more attempts")
    for job, path, r in failed:
        save(path.with_name("result_first_pass.json"), r)
        path.unlink()
    run_queue([job for job, _, _ in failed], labels, candidates, root, runs_dir, workers, log,
              control_name=control_name, extra_attempts=extra_attempts)
    for job, path, _ in failed:
        if path.exists():
            r = json.loads(path.read_text())
            r["tail_retry_done"] = True
            save(path, r)
    return len(failed)
