"""H6: single-stage head+tail output cap for Pi's bash tool.

Pi (DCI-Agent-Lite, L3) truncated bash output twice in opposite directions: the bash tool kept the last
<=2000 lines / 50 KB (with its notice at the end), then L3 kept the first 20,000 UTF-16 units of that and cut
the notice. Models saw a window starting ~50 KB before the end, with a misleading omitted count.

This filter runs the unchanged harness_fixes/sandbox-shell.sh, saves the complete output under
$DCI_ATTEMPT/tmp/full_outputs/, streams at most HEAD_BYTES / HEAD_LINES immediately (so a SIGKILL on timeout
still leaves the head visible), and at exit prints a notice with the true size and saved path followed by the
last TAIL_BYTES / TAIL_LINES. The printed total stays below both Pi limits, so neither Pi truncation fires.
Output that fits is passed through byte-for-byte. Exit status is preserved."""
import os, subprocess, sys, tempfile
from collections import deque
from pathlib import Path

BASE_SHELL = Path(__file__).resolve().parent.parent / "harness_fixes/sandbox-shell.sh"
HEAD_BYTES, HEAD_LINES = 9000, 900
TAIL_BYTES, TAIL_LINES = 9000, 900
CHUNK = 1 << 16


def main(argv):
    attempt = os.environ["DCI_ATTEMPT"]
    out_dir = Path(attempt) / "tmp/full_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="out-", suffix=".log", dir=out_dir)
    full = os.fdopen(fd, "wb")
    stdout = sys.stdout.buffer
    child = subprocess.Popen([str(BASE_SHELL), *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    total = lines = 0
    head_bytes = head_lines = 0
    head_done = False
    tail = deque()          # recent chunks after the head
    tail_size = 0
    dropped = False
    while True:
        chunk = child.stdout.read1(CHUNK) if hasattr(child.stdout, "read1") else child.stdout.read(CHUNK)
        if not chunk:
            break
        full.write(chunk)
        total += len(chunk)
        lines += chunk.count(b"\n")
        rest = chunk
        if not head_done:
            take = min(len(rest), HEAD_BYTES - head_bytes)
            piece = rest[:take]
            n = 0
            for k, b in enumerate(piece):
                if b == 10:
                    n += 1
                    if head_lines + n >= HEAD_LINES:
                        piece = piece[:k + 1]; break
            stdout.write(piece); stdout.flush()
            head_bytes += len(piece); head_lines += piece.count(b"\n")
            rest = rest[len(piece):]
            if head_bytes >= HEAD_BYTES or head_lines >= HEAD_LINES:
                head_done = True
        if rest:
            head_done = True
            tail.append(rest); tail_size += len(rest)
            while tail and tail_size - len(tail[0]) >= TAIL_BYTES + 4096:
                tail_size -= len(tail.popleft()); dropped = True
    rc = child.wait()
    full.close()
    buf = b"".join(tail)
    if not dropped and head_bytes + len(buf) <= HEAD_BYTES + TAIL_BYTES and \
            head_lines + buf.count(b"\n") <= HEAD_LINES + TAIL_LINES:
        stdout.write(buf)                      # fits: pass through unchanged
    else:
        t = buf[-TAIL_BYTES:]
        if len(buf) > TAIL_BYTES:
            nl = t.find(b"\n")
            if 0 <= nl < min(2000, len(t) - 1):   # align only if it costs little of the budget
                t = t[nl + 1:]                 # start the tail on a line boundary
        parts = t.split(b"\n")
        if len(parts) - 1 > TAIL_LINES:
            t = b"\n".join(parts[-(TAIL_LINES + 1):])
        omitted = total - head_bytes - len(t)
        note = (f"\n[... harness output limit: showing the first {head_lines} lines ({head_bytes} bytes) and the "
                f"last {t.count(b'\n')} lines ({len(t)} bytes) of {lines} lines ({total} bytes); {omitted} bytes "
                f"omitted in between. Complete output saved to {path} - inspect it with rg, sed -n, head or tail. ...]\n")
        stdout.write(note.encode()); stdout.write(t)
    stdout.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
