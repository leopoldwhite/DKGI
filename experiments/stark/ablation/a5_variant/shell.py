#!/usr/bin/python3 -S
"""Carry only this attempt's preconnected KB socket through the fixed sandbox."""

import os
from pathlib import Path
import socket
import stat
import sys

attempt = Path(os.environ["DCI_ATTEMPT"]).resolve(strict=True)
path = attempt / "kb.sock"
if path.is_symlink() or not stat.S_ISSOCK(path.stat().st_mode):
    raise RuntimeError("Expected the attempt's local KB socket")
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(str(path))
    os.dup2(connection.fileno(), 0, inheritable=True)

fixed = Path(__file__).resolve().parents[2] / "harness_fixes/sandbox-shell.sh"
# The fixed launcher closes descriptors >=3. Its final shell restores the one
# trusted connection from stdin to fd3, then leaves stdin available to heredocs.
os.execv(str(fixed), [str(fixed), "-c",
    'exec 3<&0 </dev/null; exec /bin/bash --noprofile --norc "$@"', "kb-shell", *sys.argv[1:]])
