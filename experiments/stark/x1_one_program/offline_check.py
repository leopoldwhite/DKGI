"""Offline checks for X1 "One program" (no model calls), run through `run.py --check` -> checks/offline.json.

Every shell command below goes through exactly what Pi's bash tool uses in a run: the attempt's outer
Landlock command prefix, the X1 H6 wrapper, x1_shell.py, the fixed harness sandbox and the gate, with a
live per-attempt counter in this process. Scratch attempts live under ../trials/O1_offline/ (neutral path).
"""
import ast
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import x1 as x

CHECKS = x.HERE / "checks"
OFFLINE = x.RUNS / "O1_offline"
# Short scratch directory names: the counter socket <dataset>/<name>/attempt_001.gate.sock must fit the 107-byte limit
# of Unix socket paths (README, Requirements).
SCRATCH = {"preflight": "pre", "single_start": "one", "interpreters": "int", "concurrent_in_command": "cic",
           "parallel_tool_calls": "ptc", "tamper": "tam", "no_counter": "noc", "full_dkgi_reference": "ref",
           "background": "bg", "h6_network": "net", "timeout_kill": "kil"}
MARK = "PYRAN"          # printed only if an interpreter actually ran the probe program
INTERP = "INTERP-RAN"   # printed only if another interpreter ran
RESULTS = {}


def check(name, condition, detail=None):
    RESULTS.setdefault("items", []).append({"check": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


class Attempt:
    """A scratch attempt prepared exactly like a run attempt (prepare + per-attempt preflight), no model."""

    def __init__(self, name, dataset="mag", row=None):
        directory = OFFLINE / dataset / SCRATCH.get(name, name)
        if directory.exists():
            shutil.rmtree(directory)
        self.path = directory / "attempt_001"
        self.path.mkdir(parents=True)
        self.row = row or ROWS_VAL[SMOKE[dataset]]
        self.gate = x.gate_service.Gate(self.path, directory / "attempt_001.python_gate.jsonl")
        self.command, self.env, self.cwd = x.prepare(self.path, self.row, PROTOCOL)
        self.preflight = x.preflight(self.path, self.row, self.command, self.env, self.cwd, self.gate)

    def sh(self, script, timeout=120):
        """stdout is what the model sees: H6 (cap_filter) merges the shell's stderr into stdout."""
        r = x.shell(self.command, self.env, self.cwd, script, timeout)
        return {"script": script, "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

    def summary(self):
        return self.gate.summary("agent")

    def close(self):
        self.gate.close()


def no_leak(text):
    return "ablation" not in text.lower() and "/private/" not in text and "/public/" not in text


# ---------------------------------------------------------------- 1. prompt

def check_prompt():
    sys.path.insert(0, str(x.STARK / "dci_run"))
    import core  # Full DKGI runner (dci_run, preset pi_full_test); imported read-only
    check("R1 output sentence equals Full DKGI's", x.f.R1_OUTPUT == core.P1R1_OUTPUT)
    prompt_file = x.STARK.parents[1] / "prompts/one_program_prompt.txt"
    check("sentence is the last line of prompts/one_program_prompt.txt",
          prompt_file.read_text().endswith(x.SENTENCE + "\n"), x.SENTENCE)
    out = []
    for ds in x.DATASETS:
        test_qid = next(q["query_id"] for q in x.f.test_questions() if q["dataset"] == ds)
        for split, row in (("val", ROWS_VAL[SMOKE[ds]]), ("test-0.1", ROWS_TEST[f"{ds}:{test_qid}"])):
            attempt = x.item_dir(ds, split, row["query_id"]) / "attempt_001"     # path only; nothing is created
            full = core.prompt_for(dict(row, corpus_dir=str(x.STARK / "corpus" / ds)), attempt)
            mine = x.prompt(row, attempt)
            check(f"{ds} {split}: X1 prompt = Full DKGI prompt + sentence + newline", mine == full + x.SENTENCE + "\n")
            check(f"{ds} {split}: sentence appears once, no 'ablation'", mine.count(x.SENTENCE) == 1 and
                  "ablation" not in mine.lower())
            item = {"dataset": ds, "split": split, "query_id": row["query_id"], "chars_full": len(full),
                    "chars_x1": len(mine), "added": mine[len(full):]}
            recorded_path = x.STARK / "pi_full_test/runs/pi_final" / ds / "test" / str(row["query_id"]) / "result.json"
            if split == "test-0.1" and recorded_path.exists():   # compared once the Full DKGI run exists
                recorded = x.read(x.STARK / "pi_full_test/runs/pi_final" / ds / "test" / str(row["query_id"]) / "result.json")
                actual = (Path(recorded["attempt_path"]) / "prompt.txt").read_text()
                same = actual.replace(recorded["attempt_path"], str(attempt)) == full
                check(f"{ds}: Full DKGI prompt equals the recorded pi_final prompt.txt (attempt path substituted)", same)
                item["recorded_full_dkgi_prompt"] = str(Path(recorded["attempt_path"]) / "prompt.txt")
            out.append(item)
    tail = x.prompt(ROWS_VAL[SMOKE["mag"]], OFFLINE / "x").split("Corpus directory:")[1]
    RESULTS["prompt"] = {"per_question": out, "tool_paragraph_example": "Corpus directory:" + tail}


# ---------------------------------------------------------------- 2. per-attempt preparation and preflight

def check_preflight():
    out = {}
    for ds in x.DATASETS:
        a = Attempt("preflight", ds)
        try:
            check(f"{ds}: prompt.txt is the X1 prompt", (a.path / "prompt.txt").read_text() == x.prompt(a.row, a.path))
            check(f"{ds}: working directory is the Full DKGI corpus", a.cwd == x.STARK / "corpus" / ds)
            settings = x.read(a.path / "agent/settings.json")
            check(f"{ds}: Pi shellPath is the X1 H6 wrapper", settings["shellPath"] == str(x.WRAPPER))
            check(f"{ds}: Pi request retry as ablation_final", settings["retry"] == x.PI_RETRY)
            cmd = x.read(a.path / "command.json")
            check(f"{ds}: model, turns, thinking and offline flags", all(v in cmd for v in (
                "gpt-6-luna", "300", "--extra-arg=--thinking high", "--extra-arg=--context-management-level level3",
                "--extra-arg=--offline")) and cmd[cmd.index("--tools") + 1] == "read,bash")
            out[ds] = a.preflight
        finally:
            a.close()
    RESULTS["preflight"] = out


# ---------------------------------------------------------------- 3. the single start

BYPASS = [
    ("plain", f"python3 -c 'print(\"{MARK}\")'"),
    ("bash -c", f"bash -c 'python3 -c \"print(\\\"{MARK}\\\")\"'"),
    ("bash -lc nested", f"bash -lc \"bash -c 'python3 -c \\\"print(1)\\\" && echo {MARK}-chain'\""),
    ("env", f"env python3 -c 'print(\"{MARK}\")'"),
    ("absolute /usr/bin/python3", f"/usr/bin/python3 -c 'print(\"{MARK}\")'"),
    ("absolute /usr/bin/python3.12", f"/usr/bin/python3.12 -c 'print(\"{MARK}\")'"),
    ("python3.12 on PATH", f"python3.12 -c 'print(\"{MARK}\")'"),
    ("python on PATH", f"python -c 'print(\"{MARK}\")'"),
    ("shim by absolute path", f"\"$(command -v python3)\" -c 'print(\"{MARK}\")'"),
    ("exec", f"exec python3 -c 'print(\"{MARK}\")'"),
    ("command builtin", f"command python3 -c 'print(\"{MARK}\")'"),
    ("pipe into python3", f"echo 'print(\"{MARK}\")' | python3"),
    ("pipe into python3 -", f"echo 'print(\"{MARK}\")' | python3 -"),
    ("here-document", f"python3 - <<'PY'\nprint(\"{MARK}\")\nPY"),
    ("xargs", f"echo x | xargs python3 -c 'print(\"{MARK}\")'"),
    ("xargs -I", f"echo x | xargs -I{{}} python3 -c 'print(\"{MARK}\")'"),
    ("find -exec", f"find . -maxdepth 0 -exec python3 -c 'print(\"{MARK}\")' \\;"),
    ("find -execdir", f"find . -maxdepth 0 -execdir python3 -c 'print(\"{MARK}\")' \\;"),
    ("loader + python binary", f"/lib64/ld-linux-x86-64.so.2 /usr/bin/python3.12 -c 'print(\"{MARK}\")'"),
    ("copy python binary", f"cat /usr/bin/python3.12 > \"$TMPDIR/py\"; ls -l \"$TMPDIR/py\"; \"$TMPDIR/py\" -c 'print(\"{MARK}\")'"),
    ("subshell", f"( python3 -c 'print(\"{MARK}\")' )"),
    ("command substitution", f"echo \"$(python3 -c 'print(\"{MARK}\")')\""),
    ("background", f"python3 -c 'print(\"{MARK}\")' & wait"),
    ("PATH reset", f"PATH=/usr/bin:/bin python3 -c 'print(\"{MARK}\")'"),
    ("PATH reset export", f"export PATH=/usr/local/bin:/usr/bin:/bin; python3 -c 'print(\"{MARK}\")'"),
]

INTERPRETERS = [
    ("awk", f"awk 'BEGIN{{print \"{INTERP}\"}}'"), ("mawk", f"mawk 'BEGIN{{print \"{INTERP}\"}}'"),
    ("gawk", f"gawk 'BEGIN{{print \"{INTERP}\"}}'"), ("/usr/bin/awk", f"/usr/bin/awk 'BEGIN{{print \"{INTERP}\"}}'"),
    ("perl", f"perl -e 'print \"{INTERP}\\n\"'"), ("/usr/bin/perl", f"/usr/bin/perl -e 'print \"{INTERP}\\n\"'"),
    ("node", f"node -e 'console.log(\"{INTERP}\")'"), ("ruby", f"ruby -e 'puts \"{INTERP}\"'"),
    ("php", f"php -r 'echo \"{INTERP}\";'"), ("sh", f"sh -c 'echo {INTERP}'"), ("/bin/sh", f"/bin/sh -c 'echo {INTERP}'"),
    ("dash", f"dash -c 'echo {INTERP}'"), ("env awk", f"env awk 'BEGIN{{print \"{INTERP}\"}}'"),
    ("find -exec awk", f"find . -maxdepth 0 -exec awk 'BEGIN{{print \"{INTERP}\"}}' \\;"),
    ("xargs perl", f"echo x | xargs perl -e 'print \"{INTERP}\\n\"'"),
    ("loader + perl", f"/lib64/ld-linux-x86-64.so.2 /usr/bin/perl -e 'print \"{INTERP}\\n\"'"),
    ("loader + awk", f"/lib64/ld-linux-x86-64.so.2 /usr/bin/mawk 'BEGIN{{print \"{INTERP}\"}}'"),
    ("loader + node", f"/lib64/ld-linux-x86-64.so.2 \"{shutil.which('node') or '/usr/bin/node'}\" -e 'console.log(\"{INTERP}\")'"),
]
# The Shell only layer (ablation/sandbox/exec_only.py) grants read on /usr/lib; two interpreters stored there
# can be run through the dynamic loader in Shell only as well. X1's shell layer is the same, so these are
# recorded (not required to fail), and the run audit counts any loader use.
SHARED_LOADER_CAVEAT = [
    ("loader + busybox awk (/usr/lib)", f"/lib64/ld-linux-x86-64.so.2 /usr/lib/initramfs-tools/bin/busybox awk 'BEGIN{{print \"{INTERP}\"}}'"),
    ("loader + java (/usr/lib/jvm)", "/lib64/ld-linux-x86-64.so.2 /usr/lib/jvm/java-21-openjdk-amd64/bin/java -version 2>&1 | head -1"),
]


def check_single_start():
    out = {}
    a = Attempt("single_start")
    try:
        same = a.sh(f"python3 -c 'print(\"A1\")'; python3 -c 'print(\"A2\")'; echo rc=$?")
        check("two starts in one command: first runs, second refused with the fixed message (exit 126)",
              same["stdout"] == "A1\n" + x.REFUSAL + "rc=126\n", same)
        out["same_command"] = same
        routes = {}
        for name, script in BYPASS:
            r = a.sh(script)
            routes[name] = r
            check(f"second start refused via {name}", MARK not in r["stdout"], r)
        out["bypass_routes"] = routes
        s = a.summary()
        check("counter: one grant and one interpreter for the whole attempt", s["granted"] == 1 and s["started"] == 1, s)
        out["counter"] = s
        refused_by_shim = [k for k, r in routes.items() if x.REFUSAL in r["stdout"]]
        out["routes_refused_with_fixed_message"] = refused_by_shim
        out["routes_denied_otherwise"] = sorted(set(routes) - set(refused_by_shim))
    finally:
        a.close()
    RESULTS["single_start"] = out


def check_interpreters():
    out = {}
    a = Attempt("interpreters")
    try:
        for name, script in INTERPRETERS:
            r = a.sh(script)
            out[name] = r
            check(f"{name} blocked", INTERP not in r["stdout"], r)
        caveat = {name: a.sh(script) for name, script in SHARED_LOADER_CAVEAT}
        RESULTS["shared_loader_caveat"] = caveat
        allowed = a.sh("set -e; pwd; ls >/dev/null; cat schema.json | head -n 1; tail -n 1 schema.json; rg dataset schema.json; "
                       "sed -n 1p schema.json; wc -l schema.json; echo x | cut -c 1 | sort; comm /dev/null /dev/null; "
                       "find . -maxdepth 0; echo x | xargs echo; bash -c 'echo ok'")
        check("allowlisted commands work (Shell only check list)", allowed["rc"] == 0, allowed)
        out["allowed"] = allowed
        # Same executable and read rules as Shell only (exec_only.py), plus the shim.
        sys.path.insert(0, str(x.STARK / "ablation/sandbox"))
        import exec_only
        check("allowlist names equal Shell only's", exec_only.ALLOWED == x.ALLOWLIST)
        names = sorted(p.name for p in (a.path / "sandbox/bin").iterdir())
        check("PATH directory = Shell only names + python3", names == sorted(x.ALLOWLIST + ["python3"]), names)
        out["path_dir"] = names
    finally:
        a.close()
    RESULTS["interpreters"] = out


# ---------------------------------------------------------------- 4. the counter cannot be refilled or duplicated

INSIDE = r"""
import os, socket, sys
attempt = os.environ["X1P_ATTEMPT"]
item = os.path.dirname(attempt)
r = {}
def attempt_(name, fn):
    try:
        fn(); r[name] = "ALLOWED"
    except Exception as e:
        r[name] = type(e).__name__
def connect():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(attempt + ".gate.sock")
attempt_("connect counter socket", connect)
def bind_item():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.bind(attempt + ".fake.sock"); s.listen(1)
attempt_("bind socket next to attempt", bind_item)
def bind_attempt():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.bind(os.path.join(attempt, "fake.sock")); s.listen(1)
attempt_("bind+listen socket in attempt dir", bind_attempt)
attempt_("write counter fd 9", lambda: os.write(9, b'{"op":"take"}\n'))
attempt_("fstat request fd 198", lambda: os.fstat(198))
attempt_("open /proc/self/fd", lambda: os.listdir("/proc/self/fd"))
attempt_("read counter log", lambda: open(attempt + ".python_gate.jsonl").read())
attempt_("create file next to attempt", lambda: open(attempt + ".x", "w").close())
attempt_("unlink counter socket", lambda: os.unlink(attempt + ".gate.sock"))
attempt_("read gate source", lambda: open(os.environ["X1P_SRC"]).read())
attempt_("write shim", lambda: open(os.path.join(attempt, "sandbox/bin/python3"), "ab").close())
print("INSIDE " + repr(r))
"""


def check_counter():
    out = {}
    a = Attempt("concurrent_in_command")
    try:
        r = a.sh("python3 -c 'import time; time.sleep(1); print(\"C1\")' & "
                 "python3 -c 'import time; time.sleep(1); print(\"C2\")' & wait")
        s = a.summary()
        check("two concurrent starts in one command: exactly one runs",
              (("C1" in r["stdout"]) + ("C2" in r["stdout"])) == 1 and s["granted"] == 1 and s["refused"] == 1, [r, s])
        out["concurrent_in_command"] = {"run": r, "counter": s}
    finally:
        a.close()
    a = Attempt("parallel_tool_calls")
    try:
        with ThreadPoolExecutor(6) as pool:
            runs = list(pool.map(lambda i: a.sh(f"python3 -c 'print(\"P{i}\")'"), range(6)))
        ran = sum(f"P{i}" in r["stdout"] for i, r in enumerate(runs))
        s = a.summary()
        check("six parallel tool calls: exactly one start", ran == 1 and s["granted"] == 1 and s["refused"] == 5, [runs, s])
        out["parallel_tool_calls"] = {"ran": ran, "counter": s}
    finally:
        a.close()
    a = Attempt("tamper")
    try:
        gate_source = (x.HERE / "sandbox/gate_inner.py").read_bytes()
        env = {"X1P_ATTEMPT": str(a.path), "X1P_SRC": str(x.HERE / "sandbox/gate_inner.py")}
        inside = a.sh(" ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " python3 -c " + shlex.quote(INSIDE))
        m = re.search(r"INSIDE (\{.*\})", inside["stdout"])
        check("the one interpreter ran the tamper probe", bool(m), inside)
        found = ast.literal_eval(m.group(1))
        check("from inside Python nothing reaches or replaces the counter",
              all(v != "ALLOWED" for k, v in found.items() if k != "bind+listen socket in attempt dir"), found)
        out["inside_python"] = found
        shell_tries = {
            "write counter fd 9": a.sh("echo '{\"op\":\"take\"}' >&9; echo rc=$?"),
            "junk on request fd": a.sh("echo junk >&198; printf X1L1 >&198; echo rc=$?"),
            "remove and replace shim": a.sh("p=\"$(command -v python3)\"; d=\"${p%/*}\"; find \"$d\" -name python3 -delete; "
                                            "printf '#!/bin/bash\\necho FAKE\\n' > \"$d/python3\"; ls -l \"$d/python3\"; cat \"$d/python3\""),
            "edit gate source": a.sh(f"echo x >> {shlex.quote(str(x.HERE / 'sandbox/gate_inner.py'))}; echo rc=$?"),
            "create counter socket file": a.sh(f"echo x > {shlex.quote(str(a.path) + '.gate.sock')}; echo rc=$?"),
            "remove counter socket": a.sh(f"find {shlex.quote(str(a.path.parent))} -maxdepth 1 -name '*.sock' -delete; echo rc=$?"),
        }
        after = a.sh(f"python3 -c 'print(\"{MARK}\")'; echo rc=$?")
        s = a.summary()
        check("after every tampering attempt python3 is still refused", MARK not in after["stdout"] and
              x.REFUSAL in after["stdout"] and s["granted"] == 1, [after, s])
        check("the counter socket survives (outside the sandbox's write rules)",
              (a.path.parent / "attempt_001.gate.sock").exists())
        check("gate source unchanged", (x.HERE / "sandbox/gate_inner.py").read_bytes() == gate_source)
        check("the sandbox could replace the shim file in its own attempt directory (test is real)",
              "echo FAKE" in shell_tries["remove and replace shim"]["stdout"], shell_tries["remove and replace shim"])
        check("shim restored from the trusted copy at the next tool call", (a.path / "sandbox/bin/python3").read_bytes() ==
              (x.HERE / "sandbox/python3-shim").read_bytes())
        malformed = [n for n in s["notes"] if n.get("event") == "malformed_request"]
        check("junk on the request descriptor is logged and ignored", len(malformed) >= 1, s["notes"])
        out["shell_tries"] = shell_tries
        out["after"] = after
        out["counter"] = s
    finally:
        a.close()
    a = Attempt("no_counter")
    try:
        a.gate.close()   # the counter is gone (e.g. runner stopped): every start must be refused
        r = a.sh(f"python3 -c 'print(\"{MARK}\")'; echo rc=$?")
        check("no counter -> python3 refused (fail closed)", MARK not in r["stdout"] and x.REFUSAL in r["stdout"], r)
        out["no_counter"] = r
    finally:
        pass
    RESULTS["counter"] = out


# ---------------------------------------------------------------- 5. behaviour of the one start, H6, network, cleanup

def check_one_start_behaviour():
    out = {}
    cases = [
        ("here-document program", f"python3 - <<'PY'\nimport sys, json\nprint('HEREDOC-OK', sys.argv, json.load(open('schema.json'))['dataset'])\nPY",
         lambda r: "HEREDOC-OK ['-']" in r["stdout"] and r["rc"] == 0),
        ("exit status", "python3 -c 'import sys; sys.exit(3)'; echo rc=$?", lambda r: "rc=3" in r["stdout"]),
        ("signal status", "python3 -c 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)'; echo rc=$?",
         lambda r: "rc=143" in r["stdout"]),
        ("pipe in and out", "head -c 100 schema.json | python3 -c 'import sys; print(len(sys.stdin.read()))' | cat",
         lambda r: r["stdout"].strip() == "100"),
        ("shared output file order", "{ echo first; python3 -c 'print(\"second\")'; echo third; } > \"$TMPDIR/o.txt\"; cat \"$TMPDIR/o.txt\"",
         lambda r: r["stdout"] == "first\nsecond\nthird\n"),
        ("exported variable and cwd", "cd \"$TMPDIR\" && export X1VAR=v && python3 -c 'import os; print(os.environ[\"X1VAR\"], os.getcwd())'",
         lambda r: r["stdout"].startswith("v ") and r["stdout"].strip().endswith("/tmp")),
        ("traceback as in Full DKGI's tool sandbox", "python3 -c 'raise ValueError(7)'",
         lambda r: r["stdout"] == FULL_DKGI_TRACEBACK),
        ("H6 cap on python output", "python3 -c 'print(\"x\" * 50000)'", lambda r: "[... harness output limit" in r["stdout"]),
    ]
    global FULL_DKGI_TRACEBACK
    a = Attempt("full_dkgi_reference")
    try:  # the same command in Full DKGI's tool sandbox (fixed harness shell, same outer sandbox), for comparison
        outer = a.command[:a.command.index("--") + 1]
        ref = subprocess.run(outer + [str(x.launcher.SHELL), "-c", "python3 -c 'raise ValueError(7)'"], cwd=a.cwd,
                             env=a.env, capture_output=True, text=True, timeout=60)
        FULL_DKGI_TRACEBACK = ref.stdout + ref.stderr
        check("Full DKGI reference traceback recorded", "ValueError: 7" in ref.stderr, ref.stderr)
        out["full_dkgi_traceback"] = ref.stderr
    finally:
        a.close()
    for number, (name, script, ok) in enumerate(cases):
        a = Attempt(f"b{number}")   # short directory names: Unix socket paths are limited to 107 bytes
        try:
            r = a.sh(script)
            check(f"one start: {name}", ok(r), r)
            out[name] = {**r, "counter": a.summary()}
        finally:
            a.close()
    a = Attempt("background")
    try:  # first start in a background job whose shell exits at once (request arrives during the gate's grace)
        first = a.sh("python3 -c 'import os, time; time.sleep(2); open(os.environ[\"TMPDIR\"] + \"/bg.txt\", \"w\").write(\"bg-done\")' "
                     ">/dev/null 2>&1 & echo launched")
        time.sleep(4)
        second = a.sh("cat \"$TMPDIR/bg.txt\"")
        check("background first start runs after the command returns", "launched" in first["stdout"] and
              second["stdout"] == "bg-done", [first, second, a.summary()])
        out["background"] = {"first": first, "second": second, "counter": a.summary()}
    finally:
        a.close()
    a = Attempt("h6_network")
    try:
        big = a.sh("cat nodes/0000.jsonl; exit 7")
        check("H6 cap active on bash output, exit status kept", "[... harness output limit" in big["stdout"]
              and big["rc"] == 7 and len(big["stdout"]) < 20000, {k: v if k != "stdout" else v[-400:] for k, v in big.items()})
        saved = re.search(r"Complete output saved to (\S+) ", big["stdout"])
        check("H6 full output saved in the attempt scratch directory", bool(saved) and saved.group(1).startswith(str(a.path / "tmp/full_outputs")))
        net = a.sh("exec 3<>/dev/tcp/1.1.1.1/80 && echo NET-OPEN; echo done")
        check("no network from bash", "NET-OPEN" not in net["stdout"], net)
        out["h6"] = {"rc": big["rc"], "notice": re.search(r"\[\.\.\. harness output limit[^\]]*\]", big["stdout"]).group(0)}
        out["network"] = net
    finally:
        a.close()
    a = Attempt("timeout_kill")
    try:
        # Pi's bash tool kills the whole process group with SIGKILL on timeout; the interpreter must go too.
        outer = a.command[:a.command.index("--") + 1]
        p = subprocess.Popen(outer + [str(x.WRAPPER), "-c", "python3 -c 'import time; time.sleep(120)' # x1-sleeper"],
                             cwd=a.cwd, env=a.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        time.sleep(3)
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        time.sleep(1)
        left = subprocess.run(["pgrep", "-f", "time.sleep\\(120\\)"], capture_output=True, text=True).stdout.split()
        check("process-group kill (Pi timeout) also stops the interpreter", not left, left)
        out["timeout_kill"] = {"remaining_processes": left, "counter": a.summary()}
    finally:
        a.close()
    RESULTS["one_start"] = out


def check_logs():
    texts = [x.REFUSAL]
    for path in OFFLINE.rglob("*.python_gate.jsonl"):
        texts.append(path.read_text())
    check("refusal message and counter logs: no 'ablation', no label paths", all(no_leak(t) for t in texts))
    shells = [x.HERE / "sandbox/x1_shell.py", x.HERE / "sandbox/gate_inner.py", x.HERE / "h6/cap_x1.py", x.HERE / "h6/capped-x1.sh"]
    check("no model-visible message in the X1 shell code mentions 'ablation'",
          all("ablation" not in line.lower() for p in shells for line in p.read_text().splitlines()
              if "write(2" in line or "SystemExit(" in line or "b\"sandbox-shell" in line))


def run():
    global ROWS_VAL, ROWS_TEST, SMOKE, PROTOCOL
    started = time.time()
    CHECKS.mkdir(exist_ok=True)
    ROWS_VAL = x.f.load_split("val")[0]
    ROWS_TEST = x.f.load_split("test-0.1")[0]
    SMOKE = {q["dataset"]: f"{q['dataset']}:{q['query_id']}" for q in x.f.val_selection(ROWS_VAL) if q["smoke"]}
    PROTOCOL = x.c.val.protocol()
    status = "failed"
    try:
        check_prompt()
        check_preflight()
        check_single_start()
        check_interpreters()
        check_counter()
        check_one_start_behaviour()
        check_logs()
        status = "passed"
    finally:
        RESULTS.update(status=status, no_model_calls=True, seconds=round(time.time() - started, 1), finished_at=x.now(),
                       checks_passed=sum(i["passed"] for i in RESULTS.get("items", [])),
                       checks_total=len(RESULTS.get("items", [])))
        x.save(CHECKS / "offline.json", RESULTS)
