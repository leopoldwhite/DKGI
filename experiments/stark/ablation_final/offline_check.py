"""Offline checks for the final ablation; never sends a model request."""

import ast
import json
from pathlib import Path
import py_compile
import re
import shlex
import subprocess
import sys
import tempfile

import final as f

CHECKS = f.HERE / "checks"


def log(text):
    print(text, flush=True)


def check_prompts(rows):
    """Diff vs P0 must be exactly the output sentence and the tool paragraph."""
    core_source = ast.parse((f.STARK / "dci_run/core.py").read_text())
    r1 = next(ast.literal_eval(node.value) for node in core_source.body if isinstance(node, ast.Assign)
              and any(getattr(t, "id", None) == "P1R1_OUTPUT" for t in node.targets))
    assert r1 == f.R1_OUTPUT, "Final-prompt output sentence differs from dci_run/core.py P1R1_OUTPUT"
    out = {"r1_matches_full_dkgi_runner": True, "arms": {}}
    for dataset in f.DATASETS:
        row = next(r for r in rows.values() if r["dataset"] == dataset)
        attempt = Path("/ATTEMPT")
        p0 = f.p0_prompt(row, attempt)
        assert p0.count(f.variants.OLD_OUTPUT) == 1 and p0.count(f.TOOL_BOUNDARY) == 1
        p0_prefix, _, rest = p0.partition(f.TOOL_BOUNDARY)
        p0_tail = f.TOOL_BOUNDARY + rest
        # The file-interface paragraph used by the H/A4 prompts must be exactly P0's paragraph.
        assert f.arm_paragraph(row, attempt, "H") == p0_tail, "Tool paragraph boundary is not P0's paragraph"
        expected_prefix = p0_prefix.replace(f.variants.OLD_OUTPUT, f.R1_OUTPUT, 1)
        assert expected_prefix.replace(f.R1_OUTPUT, f.variants.OLD_OUTPUT, 1) + p0_tail == p0
        for arm in f.ARMS:
            text = f.prompt(row, attempt, arm)
            assert text.startswith(expected_prefix), arm
            paragraph = text[len(expected_prefix):]
            assert paragraph == f.arm_paragraph(row, attempt, arm), arm
            if arm in ("A1", "A2", "A5"):
                assert f.TOOL_BOUNDARY not in paragraph and "Available tools" in paragraph, arm
            if arm == "A6":
                assert paragraph == p0_tail.replace("Nodes and edges are complete local text files",
                                                    "Nodes are complete local text files").replace(
                    str(f.STARK / "corpus" / dataset), str(f.VIEW["A6"] / dataset))
            if arm == "A3":
                assert paragraph == p0_tail.replace(f.A3_OLD_TOOLS, f.A3_NEW_TOOLS).replace(
                    str(f.STARK / "corpus" / dataset), str(f.VIEW["A3"] / dataset)) + (
                    "General-purpose programming languages are unavailable; use only shell text tools and bash syntax.\n")
                assert "Python" not in paragraph
            assert f.VISIBLE_FORBIDDEN not in text.lower() and f.VISIBLE_FORBIDDEN not in str(f.corpus_for(arm, dataset)).lower()
            out["arms"].setdefault(arm, {})[dataset] = {"prefix_equals_p0_with_r1": True, "tool_paragraph": paragraph}
    return out


def check_arm(arm, dataset, row, protocol):
    attempt = f.RUNS / "_offline_check" / arm / dataset
    if attempt.exists():
        subprocess.run(["rm", "-rf", str(attempt)], check=True)
    attempt.mkdir(parents=True)
    server = f.service.start(attempt, dataset, arm) if arm in ("A1", "A2", "A5") else None
    try:
        command, env, cwd = f.prepare(attempt, row, arm, protocol)
        settings = f.read(attempt / "agent/settings.json")
        assert settings["shellPath"] == str(f.WRAPPER[arm])
        outer = command[:command.index("--") + 1]
        for path in (f.WRAPPER[arm], f.H6_DIR / "cap_select.py", f.CAP_FILTER, f.BASE_SHELL[arm]):
            assert str(path) in outer, f"{arm}: outer Landlock lacks read for {path}"
        base_in_wrapper = f.WRAPPER[arm].read_text().split("cap_select.py\" ")[1].split()[0]
        assert {"default": f.launcher.SHELL, "a3": f.BASE_SHELL["A3"], "a5": f.BASE_SHELL["A5"]}[base_in_wrapper] == f.BASE_SHELL[arm]
        prompt_text = (attempt / "prompt.txt").read_text()
        assert prompt_text == f.prompt(row, attempt, arm), "prompt.txt is not the P1-R1 arm prompt"
        assert f.R1_OUTPUT in prompt_text and f.variants.OLD_OUTPUT not in prompt_text
        checks = f.isolation(attempt, row, arm, command, env, cwd)
        bad = f.VISIBLE_FORBIDDEN
        visible = {"prompt": prompt_text, "cwd": str(cwd), "DCI_CORPUS": env.get("DCI_CORPUS", ""), "attempt": str(attempt)}
        assert not [k for k, v in visible.items() if bad in v.lower()], {k: v[:200] for k, v in visible.items() if bad in v.lower()}
        wrapper = str(f.WRAPPER[arm])
        # H6 on a large output: single head+tail cap, full output kept, exit status preserved.
        big = subprocess.run(outer + [wrapper, "-c", 'for i in {1..40000}; do echo "line $i of the H6 probe"; done; exit 7'],
                             cwd=cwd, env=env, capture_output=True, timeout=300)
        text = big.stdout.decode("utf-8", "replace")
        saved = sorted((attempt / "tmp/full_outputs").glob("out-*.log"), key=lambda p: p.stat().st_mtime)
        h6 = {"returncode": big.returncode, "printed_utf16_units": f.js_units(text),
              "notice": "[... harness output limit" in text, "first_line": text.splitlines()[0],
              "last_line": text.splitlines()[-1], "full_output_lines": saved[-1].read_bytes().count(b"\n") if saved else 0}
        assert h6["returncode"] == 7 and h6["notice"] and h6["printed_utf16_units"] <= f.L3_LIMIT_UNITS, h6
        assert h6["first_line"] == "line 1 of the H6 probe" and h6["last_line"] == "line 40000 of the H6 probe", h6
        assert h6["full_output_lines"] == 40000, h6
        small = subprocess.run(outer + [wrapper, "-c", "echo small"], cwd=cwd, env=env, capture_output=True, timeout=60)
        assert small.stdout == b"small\n", small
        assert bad not in text.lower(), "H6 notice exposes a forbidden substring"
        for failing in ("nosuchcommand_xyz", "echo $0; pwd", "if then fi", "cat /nonexistent", "ls ..", "exec 9</nonexistent"):
            out = subprocess.run(outer + [wrapper, "-c", failing], cwd=cwd, env=env, capture_output=True, timeout=60)
            assert bad not in (out.stdout + out.stderr).decode("utf-8", "replace").lower(), (failing, out)
        if arm == "A3":
            for command_text in ("python3 -c 1", "awk 1", "perl -e 1", "echo x | xargs python3 -c 1",
                                 "/lib64/ld-linux-x86-64.so.2 /usr/bin/python3 -c 1"):
                denied = subprocess.run(outer + [wrapper, "-c", command_text], cwd=cwd, env=env, capture_output=True, timeout=60)
                assert denied.returncode != 0, command_text
            allowed = subprocess.run(outer + [wrapper, "-c", "rg -c dataset schema.json; head -c 1 schema.json | wc -c; "
                                     "cat " + shlex.quote(str(saved[-1])) + " | tail -n 1"],
                                     cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
            assert allowed.returncode == 0 and "line 40000" in allowed.stdout, allowed
            h6["a3_capped_allowlist"] = "interpreters denied, allowlisted tools work, full output readable"
        if arm == "A5":
            code = (f"import sys; sys.path.insert(0,{str(attempt)!r}); import kbtools; k=kbtools.Client({str(attempt / 'kb.sock')!r}); "
                    "s=k.get_schema(); r=k.search('cell', k=30); n=k.neighbors(0); print(len(s)>0, len(r)>0, len(n)>0); print(r*40)")
            fd3 = subprocess.run(outer + [wrapper, "-c", "python3 -S -c " + shlex.quote(code)], cwd=cwd, env=env,
                                 capture_output=True, timeout=300)
            out = fd3.stdout.decode("utf-8", "replace")
            assert fd3.returncode == 0 and out.startswith("True True True"), out[-2000:] + fd3.stderr.decode()[-2000:]
            assert f.js_units(out) <= f.L3_LIMIT_UNITS and "[... harness output limit" in out
            h6["a5_fd3_through_h6"] = "get_schema/search/neighbors over fd3; large output capped"
            server_log = [json.loads(line) for line in (attempt / "tool_requests.jsonl").read_text().splitlines()]
            h6["a5_service_calls"] = len(server_log)
        if server:
            cases = [("get_schema", {}), ("get_node", {"node_id": 0}), ("search", {"query": "cell", "k": 30})]
            if arm != "A1":
                cases += [("neighbors", {"node_id": 0}), ("neighbors", {"node_id": 1, "query": "a", "k": 20})]
            for name, args in cases:
                assert bad not in server.tools.call(name, args).lower(), (arm, name, args)
            for name, args in (("get_node", {"node_id": 10 ** 12}), ("neighbors", {"node_id": -1}), ("nope", {})):
                try:
                    server.tools.call(name, args)
                except Exception as error:
                    assert bad not in str(error).lower()
            h6["tool_service_outputs"] = f"{len(cases)} calls + 3 errors, no forbidden substring"
        return {"arm": arm, "dataset": dataset, "query_id": row["query_id"], "isolation": checks, "h6": h6,
                "shellPath": settings["shellPath"], "base_shell": str(f.BASE_SHELL[arm])}
    finally:
        if server:
            server.shutdown()
            server.server_close()


def check_views():
    ext = (f.ABLATION / "tools/extension.mjs").read_text().lower()
    assert f.VISIBLE_FORBIDDEN not in ext.replace("process.env", "")
    out = {"workspace": f.build_workspace(create=False)}
    for arm in f.VIEW:
        for dataset in f.DATASETS:
            info = f.build_view(arm, dataset, create=False)
            root = f.VIEW[arm] / dataset
            names = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
            assert "manifest.json" not in names
            fmt = (root / "DATA_FORMAT.txt").read_text()
            if arm == "A3":
                assert "Python" not in fmt and (root / "edges").is_dir()
            else:
                assert not (root / "edges").exists() and not any(n.endswith(".tsv") for n in names)
                assert "Graph edges" not in fmt and "relation_types" not in (root / "schema.json").read_text()
            for p in root.rglob("*"):
                if p.is_file() and p.suffix != ".jsonl":
                    text = p.read_text(errors="replace")
                    assert str(Path.home()) not in text and "inode" not in text, p
            out[f"{arm}/{dataset}"] = info
    return out


def check_resume():
    """Result paths and resume decisions on a scratch runs tree."""
    with tempfile.TemporaryDirectory() as scratch:
        runs = Path(scratch)
        job = {"arm": "A2", "dataset": "mag", "query_id": "123", "split": "test-0.1"}
        path = f.result_path(job, runs)
        assert path == runs / "A2/mag/test/123/result.json"
        assert f.result_path({**job, "split": "val"}, runs) == runs / "A2/mag/val/123/result.json"
        assert not f.is_terminal(job, runs)
        path.parent.mkdir(parents=True)
        f.save(path, {**job, "status": "running", "terminal": False})
        assert not f.is_terminal(job, runs)
        for n, status in ((1, "failed"), (2, "interrupted"), (3, "running")):
            (path.parent / f"attempt_{n:03d}").mkdir()
            f.save(path.parent / f"attempt_{n:03d}/result.json", {"status": status})
        assert f.counted_attempts(path.parent) == 2, "interrupted attempts must not count, crashed ones must"
        (path.parent / "attempt_004").mkdir()
        assert f.counted_attempts(path.parent) == 3
        f.save(path, {**job, "status": "scored", "terminal": True})
        assert f.is_terminal(job, runs)
        f.save(path, {**job, "status": "failed", "terminal": True, "failure_kind": "provider_refusal"})
        assert f.is_terminal(job, runs), "refusals are terminal"
        # Real execute() on a terminal item returns without any attempt or model call.
        original = f.RUNS
        f.RUNS = runs
        try:
            before = [p.name for p in f.attempts(path.parent)]
            value = f.execute(job, None, None, None, None)
            assert value["status"] == "failed" and [p.name for p in f.attempts(path.parent)] == before
        finally:
            f.RUNS = original
        jobs = f.schedule([{"dataset": "mag", "query_id": "1"}, {"dataset": "prime", "query_id": "2"}], f.ARMS, "test-0.1")
        assert len(jobs) == 10 and [j["query_id"] for j in jobs[:5]] in (["1"] * 5, ["2"] * 5)
        assert f.refusal({"status": "failed", "error_phase": "agent", "error": "x\ncontent_policy_violation: no"})
    questions = f.test_questions()
    counts = {d: sum(q["dataset"] == d for q in questions) for d in f.DATASETS}
    assert counts == {"amazon": 164, "mag": 266, "prime": 280}, counts
    return {"status": "passed", "test_0_1_counts": counts,
            "layout": "runs/<arm>/<dataset>/<val|test>/<query_id>/{attempt_NNN/,result.json}"}


def check_sources():
    files = sorted(p for p in f.HERE.rglob("*") if p.is_file() and "runs" not in p.relative_to(f.HERE).parts[:1]
                   and "checks" not in p.relative_to(f.HERE).parts[:1]
                   and p.relative_to(f.HERE).parts[0] not in ("data", "logs") and "__pycache__" not in p.parts
                   and p.suffix in (".py", ".sh", ".md", ".txt", ""))
    compiled = []
    with tempfile.TemporaryDirectory() as scratch:
        for path in files:
            if path.suffix == ".py":
                py_compile.compile(str(path), cfile=str(Path(scratch) / (path.name + "c")), doraise=True)
                compiled.append(path.name)
    shells = [p for p in files if p.suffix == ".sh"]
    for path in shells:
        subprocess.run(["bash", "-n", str(path)], check=True)
    return {"py_compiled": compiled, "shell_syntax_checked": [p.name for p in shells],
            "files_scanned": [str(p.relative_to(f.HERE)) for p in files]}


def run():
    report = {"no_model_requests": True, "started_at": f.now()}
    rows, _, _ = f.load_split("val")
    selection = f.val_selection(rows)
    report["sources"] = check_sources(); log("PASS sources (compile, shell syntax)")
    report["prompts"] = check_prompts(rows); log("PASS prompts (diff vs P0 = output sentence + tool paragraph)")
    report["views"] = check_views(); log("PASS corpus views (A3 v1, A6 v2)")
    report["resume"] = check_resume(); log("PASS result paths / resume")
    protocol = f.c.val.protocol()
    report["arms"] = []
    for arm in f.ARMS:
        for dataset in f.DATASETS:
            q = next(q for q in selection if q["dataset"] == dataset and q["smoke"])
            row = rows[f.key(q)]
            report["arms"].append(check_arm(arm, dataset, row, protocol))
            log(f"PASS {arm}/{dataset}: isolation, H6 wrapper ({f.WRAPPER[arm].name} -> {f.BASE_SHELL[arm].name})")
    report.update(status="passed", finished_at=f.now())
    f.save(CHECKS / "offline.json", report)
    return report


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        f.save(CHECKS / "offline.json", {"status": "failed", "error": f.redact(repr(error)), "at": f.now()})
        raise
    print("offline checks passed:", CHECKS / "offline.json")
