"""Run Full DKGI on STaRK (Pi or Codex; GPT-6-Luna or GPT-6-Astra). Resumable; one failed question never stops the queue.

Presets (paper configuration; results under <root>/<runs_dir>/<group>/<dataset>/<split dir>/<query_id>/)
  pi_full_test    : Full DKGI, Pi + GPT-6-Luna, final prompt, test (7,108) and human_generated_eval (263)
  codex_h         : Full DKGI (Codex), Codex + GPT-6-Luna, final prompt, test subset (test-0.1, 710)
  codex_luna_full : Full DKGI (Codex) on the full test; reuses the test-subset results of codex_h
  pi_astra        : Full DKGI, Pi + GPT-6-Astra, test subset
  pi_astra_full   : Full DKGI, Pi + GPT-6-Astra, full test; reuses the test-subset results of pi_astra
  codex_astra     : Full DKGI (Codex), Codex + GPT-6-Astra, test subset

Examples
  python3 run.py pi_full_test --check                 # offline: prompts, H6 wiring, isolation preflight (no model call)
  python3 run.py pi_full_test --smoke --workers 1     # first question of each split and dataset (live; counts as results)
  python3 run.py pi_full_test --workers 8             # full run; resumable
Live worker changes: edit <root>/control.json {"workers": N}.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

STARK = core.STARK
PRESETS = {
    "pi_full_test": {"root": STARK / "pi_full_test", "runs_dir": "runs", "groups": ["pi_final"],
                     "splits": ["test", "human_generated_eval"], "per_dataset": None},
    # Neutral root: the model sees its attempt path, so it must not contain "ablation".
    "codex_h": {"root": STARK / "trials", "runs_dir": "codex_h", "groups": ["codex_final"],
                "splits": ["test-0.1"], "per_dataset": None},
    # Codex + GPT-6-Luna on the full test. Same root and group as codex_h, so the 710 test-0.1 results are reused
    # (a question is done when its result.json is terminal). The tag keeps the state files of the two presets apart.
    "codex_luna_full": {"root": STARK / "trials", "runs_dir": "codex_h", "groups": ["codex_final"], "splits": ["test"],
                        "per_dataset": None, "tag": "_codex_luna_full", "model": "gpt-6-luna"},
    # GPT-6-Astra backbone (same prompt, harness and budgets as Full DKGI with GPT-6-Luna).
    "codex_astra": {"root": STARK / "astra_runs", "runs_dir": "runs", "groups": ["codex_astra"], "splits": ["test-0.1"],
                    "per_dataset": None, "tag": "_codex_astra", "model": "gpt-6-astra"},
    "pi_astra": {"root": STARK / "astra_runs", "runs_dir": "runs", "groups": ["pi_astra"], "splits": ["test-0.1"],
                 "per_dataset": None, "tag": "_pi_astra", "model": "gpt-6-astra"},
    # Full test with the same root and group: the test-0.1 results live in the same <ds>/test/<qid> directories,
    # so the full run skips them (a question is done when its result.json is terminal).
    "pi_astra_full": {"root": STARK / "astra_runs", "runs_dir": "runs", "groups": ["pi_astra"], "splits": ["test"],
                      "per_dataset": None, "tag": "_pi_astra_full", "model": "gpt-6-astra"},
}


def selection(preset, datasets):
    rows, labels, candidates = [], {}, {}
    for split in preset["splits"]:
        r, l, c = core.load_questions(split, datasets)
        if preset["per_dataset"]:
            r = core.sample(r, preset["per_dataset"], seed=20260928)
        rows += r
        labels.update(l)
        candidates.update(c)
    return rows, labels, candidates


def check(name, preset, rows, keep_scratch=False):
    """Offline checks without any model call."""
    out = {"preset": name, "questions": len(rows), "by_split_dataset": {}}
    for r in rows:
        k = f"{r['split']}/{r['dataset']}"
        out["by_split_dataset"][k] = out["by_split_dataset"].get(k, 0) + 1
    firsts = {}
    for r in rows:
        firsts.setdefault((r["split"], r["dataset"]), r)
    scratch = Path(tempfile.mkdtemp(prefix="dci_run_check_", dir=str(preset["root"])))
    try:
        for (split, ds), row in sorted(firsts.items()):
            for group in preset["groups"]:
                h = core.harness_for(group)
                attempt = scratch / group / ds / split / "attempt_001"
                attempt.mkdir(parents=True)
                p0 = core.variants.prompt_for(row, attempt, "p0")
                final = core.prompt_for(row, attempt)
                assert final == p0.replace(core.variants.OLD_OUTPUT, core.P1R1_OUTPUT, 1) and final.count(core.P1R1_OUTPUT) == 1
                command, env = core.prepare(attempt, row, core.protocol_for(h), h, final)
                item = {"prompt_differs_from_p0_only_in_output_sentence": True, "preflight": "passed"}
                if h == "pi":
                    settings = json.loads((attempt / "agent/settings.json").read_text())
                    assert settings["shellPath"] == str(core.H6_SHELL)
                    assert all(str(core.H6_DIR / f) in command for f in core.H6_FILES)
                    item["h6_shell"] = settings["shellPath"]
                out[f"{group}:{split}:{ds}"] = item
    finally:
        if keep_scratch:
            out["scratch_kept"] = str(scratch)
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    out["passed"] = all(v.get("passed", True) for v in out.values() if isinstance(v, dict) and "passed" in v)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--datasets", nargs="+", default=list(core.DATASETS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="only the first question of each split/dataset")
    parser.add_argument("--check", action="store_true", help="offline checks only")
    parser.add_argument("--keep-scratch", action="store_true", help="--check: keep the scratch attempts for inspection")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    if preset.get("model"):
        os.environ["DCI_MODEL"] = preset["model"]
    root = Path(preset["root"])
    root.mkdir(parents=True, exist_ok=True)
    tag = preset.get("tag", "")
    rows, labels, candidates = selection(preset, args.datasets)
    core.save(root / f"selection{tag}.json", [{k: r[k] for k in ("dataset", "split", "query_id")} for r in rows])
    if args.check:
        result = check(args.preset, preset, rows, args.keep_scratch)
        core.save(root / f"check{tag}.json", result)
        print(json.dumps(result, indent=1))
        return
    if args.smoke:
        firsts = {}
        for r in sorted(rows, key=lambda r: (r["split"], r["dataset"], int(r["query_id"]))):
            firsts.setdefault((r["split"], r["dataset"]), r)
        rows = list(firsts.values())
    for harness_name in {core.harness_for(group) for group in preset["groups"]}:
        core.base.require_api_key(core.protocol_for(harness_name))
    log_path = root / f"workflow{tag}.log"

    def log(message):
        with core.LOCK, open(log_path, "a") as f:
            f.write(message + "\n")
        print(message, flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: core.STOP.set())
    jobs = core.interleave(rows, preset["groups"])
    control = f"control{preset.get('tag', '')}.json"
    core.run_queue(jobs, labels, candidates, root, preset["runs_dir"], args.workers, log, control_name=control)
    # Tail pass: technically failed items (not refusals, not scored) get up to 3 more attempts, once.
    core.tail_retry(jobs, labels, candidates, root, preset["runs_dir"], args.workers, log, control_name=control)


if __name__ == "__main__":
    main()
