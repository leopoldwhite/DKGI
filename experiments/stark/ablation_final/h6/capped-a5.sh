#!/bin/bash -p
# Pi shellPath for H6: single head+tail cap around the 'a5' base shell (see cap_select.py).
exec /usr/bin/python3 -S "$(dirname "$(readlink -f "$0")")/cap_select.py" a5 "$@"
