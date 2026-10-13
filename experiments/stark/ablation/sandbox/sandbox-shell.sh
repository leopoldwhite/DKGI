#!/bin/bash -p
set -euo pipefail
HERE="$DCI_ATTEMPT/sandbox"
exec "$(dirname "$(readlink -f "$0")")/../../harness_fixes/sandbox-shell.sh" -c \
  'exec /usr/bin/python3 -S "$1/exec_only.py" "$1/bin" /bin/bash --noprofile --norc "${@:2}"' \
  ablation-shell "$HERE" "$@"
