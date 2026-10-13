#!/bin/bash -p
set -euo pipefail

: "${DCI_CORPUS:?}"
: "${DCI_ATTEMPT:?}"
: "${DCI_CODEX_ROOT:?}"
: "${DCI_LANDLOCK:?}"

exec /usr/bin/env -i PATH="$DCI_CODEX_ROOT/codex-path:/usr/bin:/bin" LANG=C.UTF-8 TMPDIR="$DCI_ATTEMPT/tmp" \
  /usr/bin/python3 -S "$DCI_LANDLOCK" \
  --read /usr --read /bin --read /lib --read /lib64 --read /etc/ld.so.cache \
  --read "$DCI_CORPUS" --write "$DCI_ATTEMPT" --read "$DCI_CODEX_ROOT" \
  --write /dev/null -- \
  /bin/bash --noprofile --norc -c 'exec -a codex-linux-sandbox "$@"' sandbox-shell \
  "$DCI_CODEX_ROOT/bin/codex" --sandbox-policy-cwd "$DCI_CORPUS" \
  --permission-profile '{"type":"external","network":"restricted"}' \
  --apply-seccomp-then-exec -- /bin/bash --noprofile --norc "$@"
