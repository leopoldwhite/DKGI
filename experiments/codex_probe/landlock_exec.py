#!/usr/bin/env python3
"""Restrict this process and descendants to explicit filesystem roots, then exec."""

import argparse
import ctypes
import os
import platform
import stat


class Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--read", action="append", default=[])
parser.add_argument("--write", action="append", default=[])
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or not args.read:
    parser.error("at least one --read path and a command are required")
if platform.machine() != "x86_64":
    parser.error("this probe only supports the verified x86_64 syscall ABI")

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def checked(result):
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


abi = checked(libc.syscall(444, 0, 0, 1))
if abi < 3:
    raise RuntimeError(f"Landlock ABI >= 3 required, found {abi}")
all_rights = (1 << 15) - 1
read_rights = (1 << 0) | (1 << 2) | (1 << 3)
file_rights = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
ruleset = Ruleset(all_rights)
ruleset_fd = checked(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))
try:
    for paths, rights in ((args.read, read_rights), (args.write, all_rights)):
        for path in paths:
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                allowed = rights if stat.S_ISDIR(os.fstat(path_fd).st_mode) else rights & file_rights
                rule = PathRule(allowed, path_fd)
                checked(libc.syscall(445, ruleset_fd, 1, ctypes.byref(rule), 0))
            finally:
                os.close(path_fd)
    checked(libc.prctl(38, 1, 0, 0, 0))
    checked(libc.syscall(446, ruleset_fd, 0))
finally:
    os.close(ruleset_fd)
checked(libc.syscall(436, ctypes.c_uint(3), ctypes.c_uint(0xFFFFFFFF), 0))
os.execvp(command[0], command)
