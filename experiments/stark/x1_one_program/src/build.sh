#!/bin/bash
# Build the two X1 helpers into ../sandbox/ (the shim installed as python3, and the interpreter lock).
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
OUT="$HERE/../sandbox"
mkdir -p "$OUT"
gcc -O2 -Wall -Wextra -o "$OUT/python3-shim" "$HERE/shim.c"
gcc -O2 -Wall -Wextra -shared -fPIC -o "$OUT/x1lock.so" "$HERE/lock.c"
chmod 0555 "$OUT/python3-shim"
chmod 0444 "$OUT/x1lock.so"
echo "built $OUT/python3-shim $OUT/x1lock.so"
