"""Add an executable allowlist inside the fixed harness shell sandbox."""

import ctypes
import os
from pathlib import Path
import sys

ALLOWED = "rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash".split()


def restrict(directory):
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    def checked(value):
        if value < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return value

    rule = Ruleset(1 | 4 | 8)
    descriptor = checked(libc.syscall(444, ctypes.byref(rule), ctypes.sizeof(rule), 0))
    try:
        executable = [(Path(directory) / name).resolve(strict=True) for name in ALLOWED]
        executable.append(Path("/lib64/ld-linux-x86-64.so.2").resolve(strict=True))
        readable = [Path("/usr/lib"), Path("/lib"), Path("/lib64"), Path("/usr/share/locale"),
                    Path("/etc/ld.so.cache"), Path("/etc/localtime"), Path("/dev/null"),
                    Path.cwd(), Path(__file__).resolve().parents[1]]
        for path, rights in [(p, 1 | 4) for p in executable] + [(p, 4 | (8 if p.is_dir() else 0)) for p in readable if p.exists()]:
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                entry = PathRule(rights, handle)
                checked(libc.syscall(445, descriptor, 1, ctypes.byref(entry), 0))
            finally:
                os.close(handle)
        checked(libc.prctl(38, 1, 0, 0, 0))
        checked(libc.syscall(446, descriptor, 0))
    finally:
        os.close(descriptor)
    os.environ["PATH"] = str(directory)


if __name__ == "__main__":
    restrict(sys.argv[1])
    os.execv(sys.argv[2], sys.argv[2:])
