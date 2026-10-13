#!/bin/bash -p
# Pi shellPath for H6: head+tail cap around the unchanged harness_fixes/sandbox-shell.sh.
exec /usr/bin/python3 -S "$(dirname "$(readlink -f "$0")")/cap_filter.py" "$@"
