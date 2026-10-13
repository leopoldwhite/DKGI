#!/usr/bin/python3 -S
"""H6 head+tail output cap around a selectable base shell.

Loads the unchanged harness_fixes_v2/cap_filter.py (single-stage head 9,000 bytes / 900 lines, tail
9,000 bytes / 900 lines, full output saved under $DCI_ATTEMPT/tmp/full_outputs/) and only replaces its
hardcoded base shell. The base shell is named by the first argument, which the per-arm wrapper script
fixes; the model cannot choose it. Exit status of the base shell is preserved.
"""
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STARK = HERE.parents[1]
BASES = {
    # A1, A2 (no bash tool is exposed; set for uniformity) and A6: the Pi sandbox shell.
    "default": STARK / "harness_fixes/sandbox-shell.sh",
    # A3: executable allowlist shell (rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash).
    "a3": STARK / "ablation/sandbox/sandbox-shell.sh",
    # A5: preconnected attempt-local KB socket handed to the sandbox on fd3.
    "a5": STARK / "ablation/a5_variant/shell.py",
}
CAP_FILTER = STARK / "harness_fixes_v2/cap_filter.py"


def load():
    spec = importlib.util.spec_from_file_location("h6_cap_filter", CAP_FILTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv):
    if not argv or argv[0] not in BASES:
        sys.stderr.write("cap_select: unknown base shell\n")
        return 2
    cap = load()
    cap.BASE_SHELL = BASES[argv[0]]
    return cap.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
