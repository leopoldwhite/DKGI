"""Add system startup files to the Landlock policy without changing it."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1] / "codex_probe/landlock_exec.py"
SHELL = ROOT / "sandbox-shell.sh"
SYSTEM_READS = ("/etc/profile", "/etc/profile.d", "/etc/bash.bashrc", "/etc/apt/apt.conf.d")


def configure(command, env, attempt, harness):
    """Wrap a prepared command; HOME changes only in tool processes."""
    attempt = Path(attempt).resolve()
    command, env = list(command), dict(env)
    if command[:3] != ["/usr/bin/python3", "-S", str(ORIGINAL)]:
        raise ValueError("Unexpected Landlock launcher")
    home = attempt / "tool_home"
    home.mkdir(exist_ok=True)
    for name in (".profile", ".bashrc"):
        path = home / name
        if not path.exists():
            path.write_text("")
    command[2] = str(ROOT / "launcher.py")
    boundary = command.index("--")
    command[boundary:boundary] = ["--read", str(ROOT / "launcher.py"), "--read", str(SHELL)]
    if harness == "codex":
        boundary = command.index("exec", boundary)
        command[boundary:boundary] = ["-c", "shell_environment_policy.set={HOME=" + json.dumps(str(home)) + "}"]
    elif harness == "pi":
        path = attempt / "agent/settings.json"
        settings = json.loads(path.read_text())
        settings["shellPath"] = str(SHELL)
        path.write_text(json.dumps(settings, indent=2) + "\n")
        env["DCI_LANDLOCK"] = str(ROOT / "launcher.py")
    else:
        raise ValueError("Unknown harness")
    return command, env


def preflight(command, env, attempt, row):
    """Check the actual configured outer sandbox before starting a model."""
    if command[:3] != ["/usr/bin/python3", "-S", str(ROOT / "launcher.py")]:
        raise ValueError("Isolation preflight requires the fixed launcher")
    corpus = Path(row["corpus_dir"])
    denied = [ROOT.parent / "private", ROOT.parent / "public",
              ROOT.parents[2] / "STaRK/data", Path.home() / ".bashrc",
              Path.home() / ".profile", Path.home() / ".codex/auth.json"]
    probe = "import os\nCORPUS=" + repr(str(corpus)) + "\n"
    probe += "DENIED=" + repr([str(path) for path in denied]) + "\n"
    probe += """assert os.listdir(CORPUS)
for path in DENIED:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (PermissionError, FileNotFoundError):
        continue
    else:
        os.close(descriptor)
        raise AssertionError('unexpected readable path: ' + path)
try:
    descriptor = os.open(CORPUS + '/.harness-preflight-write', os.O_WRONLY | os.O_CREAT, 0o600)
except PermissionError:
    pass
else:
    os.close(descriptor)
    raise AssertionError('corpus is writable')
"""
    outer = command[:command.index("--") + 1]
    subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", probe],
                   env=env, cwd=corpus, capture_output=True, check=True, timeout=30)
    result = {"configured_outer_filesystem": "passed",
              "configured_outer_real_home": "denied", "configured_outer_corpus": "read-only"}
    path = Path(attempt) / "preflight.json"
    checks = json.loads(path.read_text()) if path.exists() else {}
    checks.update(result)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(checks, indent=2) + "\n")
    temporary.replace(path)
    return result


def main():
    extra = [part for path in SYSTEM_READS for part in ("--read", path)]
    sys.argv = [str(ORIGINAL), *extra, *sys.argv[1:]]
    runpy.run_path(str(ORIGINAL), run_name="__main__")


if __name__ == "__main__":
    main()
