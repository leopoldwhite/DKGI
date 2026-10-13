#!/usr/bin/python3 -S
"""Base shell of the One program condition (X1), run by the H6 cap for every bash tool call.

Runs in Pi's outer sandbox, outside the tool sandbox, from this read-only directory:
  1. keeps the attempt's PATH directory as configured: the shell-only allowlist symlinks plus `python3`
     (the shim), rewritten from the copy here whenever its content differs;
  2. builds the Landlock rules from trusted locations (never from the attempt directory's symlinks) and passes
     the interpreter lock (x1lock.so) as bytes; the gate loads it from a sealed memory file, never from disk;
  3. connects to the attempt's counter socket (served by the runner, next to the attempt directory, where the
     tool sandbox cannot create or remove files) and hands the connection in on stdin,
     the only way through the fixed harness launcher, which closes descriptors >= 3 (as for A5);
  4. execs the unchanged harness sandbox shell, whose bash moves the connection to fd 9 and starts the gate
     (gate_inner.py, passed as source text, so the sandbox cannot edit the code it runs).
"""
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import sys

HERE = Path(__file__).resolve().parent
STARK = HERE.parents[1]
FIXED_SHELL = STARK / "harness_fixes/sandbox-shell.sh"
ALLOWED = "rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash".split()
LOADER = "/lib64/ld-linux-x86-64.so.2"
PYTHON = "/usr/bin/python3"
PYTHON_READS = ["/usr/share", "/usr/include", "/usr/src", "/usr/local/lib", "/usr/local/share", "/usr/local/include",
                "/etc/apt/apt.conf.d", "/etc/profile", "/etc/profile.d", "/etc/bash.bashrc"]
BOOT = 'exec 9<&0 </dev/null; exec /usr/bin/python3 -S -c "$1" "${@:2}"'


def ensure_file(path, source, mode):
    data = source.read_bytes()
    try:
        if not path.is_symlink() and path.is_file() and path.read_bytes() == data:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_bytes(data)
    temporary.chmod(mode)
    os.replace(temporary, path)


def ensure_link(path, target):
    if path.is_symlink() and os.readlink(path) == target:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")  # atomic replace: parallel tool calls may race here
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, path)


def main():
    attempt = Path(os.environ["DCI_ATTEMPT"]).resolve(strict=True)
    search = f"{os.environ['DCI_CODEX_ROOT']}/codex-path:/usr/bin:/bin"
    local = attempt / "sandbox"
    bin_dir = local / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    targets = {}
    for name in ALLOWED:
        target = shutil.which(name, path=search)
        if not target:
            raise SystemExit(f"sandbox-shell: {name} is missing")
        targets[name] = target
        ensure_link(bin_dir / name, target)
    ensure_file(bin_dir / "python3", HERE / "python3-shim", 0o555)
    executables = [os.path.realpath(t) for t in targets.values()] + [str(bin_dir / "python3"), os.path.realpath(LOADER)]
    # Same readable set as the Shell only allowlist layer (ablation/sandbox/exec_only.py): system libraries,
    # locale, loader cache, time zone, /dev/null, the working directory (corpus) and the attempt directory.
    readable = ["/usr/lib", "/lib", "/lib64", "/usr/share/locale", "/etc/ld.so.cache", "/etc/localtime",
                "/dev/null", os.getcwd(), str(attempt)]
    spec = {"bash_rules": [["x", p] for p in executables] + [["r", p] for p in readable],
            # The one interpreter keeps the reads Python has in Full DKGI's tool sandbox, except the directories
            # of executables (so the loader cannot run interpreter binaries): same execute rules as bash.
            "python_reads": [["r", p] for p in PYTHON_READS],
            "python_binary": [["x", os.path.realpath(PYTHON)]],
            "lock_b64": base64.b64encode((HERE / "x1lock.so").read_bytes()).decode(),
            "bin": str(bin_dir), "python": PYTHON, "grace": 0.25, "confirm_timeout": 30}
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(attempt.parent / f"{attempt.name}.gate.sock"))  # gate_service.socket_path
        os.dup2(connection.fileno(), 0, inheritable=True)
        connection.close()
    except OSError:
        pass  # no counter: stdin stays as it was, and the gate refuses every start
    source = (HERE / "gate_inner.py").read_text()
    os.execv(str(FIXED_SHELL), [str(FIXED_SHELL), "-c", BOOT, "sandbox-shell", source, json.dumps(spec), *sys.argv[1:]])


if __name__ == "__main__":
    main()
