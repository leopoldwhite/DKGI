#!/usr/bin/python3 -S
"""H6 head+tail output cap around the X1 base shell (same pattern as ablation_final/h6/cap_select.py).

Loads the unchanged harness_fixes_v2/cap_filter.py (head 9,000 bytes / 900 lines, tail 9,000 bytes / 900
lines, full output under $DCI_ATTEMPT/tmp/full_outputs/) and only replaces its base shell with
../sandbox/x1_shell.py. The exit status of the base shell is preserved.
"""
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STARK = HERE.parents[1]
BASE = HERE.parent / "sandbox/x1_shell.py"
CAP_FILTER = STARK / "harness_fixes_v2/cap_filter.py"


def main(argv):
    spec = importlib.util.spec_from_file_location("h6_cap_filter", CAP_FILTER)
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    cap.BASE_SHELL = BASE
    return cap.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
