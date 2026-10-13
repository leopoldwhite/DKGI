#!/bin/bash -p
# Pi shellPath for X1: single head+tail cap (H6) around the One program base shell (see cap_x1.py).
exec /usr/bin/python3 -S "$(dirname "$(readlink -f "$0")")/cap_x1.py" "$@"
