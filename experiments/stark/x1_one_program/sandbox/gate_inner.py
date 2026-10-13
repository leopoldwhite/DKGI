"""Per-command gate inside the tool sandbox (X1, "One program").

Not run from this file: x1_shell.py passes this source text as an argument, so the code that runs inside the
sandbox is never a file the sandbox can edit. It runs as `/usr/bin/python3 -S -c <this> <spec> -c <command>`
inside the fixed harness sandbox (Landlock + seccomp, no network), before the executable allowlist exists:

  fd 9   connection to the harness-held counter (runner process); never reaches bash or Python.
  bash   started with the shell-only allowlist plus the `python3` shim (Landlock execute rules) and fd 198,
         one end of a private socket pair on which shims send launch requests.
  shim   sends its stdin/stdout/stderr, working directory and argv/env; the gate asks the counter (one token
         per attempt, held in runner memory, only operation "take"); refused -> the shim prints the fixed
         refusal; granted -> a supervisor starts /usr/bin/python3 on the shim's streams, behind a second
         Landlock layer (shell allowlist + the Python binary) and LD_PRELOAD x1lock.so, which re-applies the
         shell allowlist before any Python code runs and confirms. The supervisor relays the exit status.

The gate exits with bash's status once bash has exited and no further request arrives (all request
descriptors closed, or a short grace period for background jobs).
"""
import ctypes
import json
import os
import select
import signal
import socket
import stat
import struct
import sys
import time

SERVICE_FD = 9
GATE_FD = 198
SPEC = json.loads(sys.argv[1])
COMMAND = sys.argv[2:]
LOG_ARG_CHARS = 2000


def restrict(rules):
    """Landlock layer: handled execute|read_file|read_dir; rules [("x", path)] execute+read, [("r", path)] read."""
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    def checked(value):
        if value < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return value

    rule = Ruleset(1 | 4 | 8)
    descriptor = checked(libc.syscall(444, ctypes.byref(rule), ctypes.sizeof(rule), 0))
    try:
        for kind, path in rules:
            try:
                handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except FileNotFoundError:
                if kind == "r":
                    continue
                raise
            try:
                is_dir = os.path.isdir(path)
                rights = (1 | 4) if kind == "x" else (4 | (8 if is_dir else 0))
                entry = PathRule(rights, handle)
                checked(libc.syscall(445, descriptor, 1, ctypes.byref(entry), 0))
            finally:
                os.close(handle)
        checked(libc.prctl(38, 1, 0, 0, 0))
        checked(libc.syscall(446, descriptor, 0))
    finally:
        os.close(descriptor)


def write_all(fd, data):
    """write(2) only: the sandbox's seccomp filter refuses send/sendto/recvfrom on sockets."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def read_exact(fd, n, timeout):
    deadline, data = time.monotonic() + timeout, b""
    while len(data) < n:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise OSError("request timed out")
        chunk = os.read(fd, n - len(data))
        if not chunk:
            raise OSError("short request")
        data += chunk
    return data


class Service:
    """Line-delimited JSON over the inherited counter connection (fd 9), with read(2)/write(2) only."""

    def __init__(self):
        self.fd = None
        try:
            if stat.S_ISSOCK(os.fstat(SERVICE_FD).st_mode):
                self.fd = os.dup(SERVICE_FD)  # non-inheritable copy; bash and Python never receive it
        except OSError:
            self.fd = None
        try:
            os.close(SERVICE_FD)
        except OSError:
            pass
        self.buffer = b""

    def send(self, record):
        if self.fd is None:
            return False
        try:
            write_all(self.fd, (json.dumps(record) + "\n").encode())
            return True
        except OSError:
            self.fd = None
            return False

    def take(self, record):
        if not self.send(dict(record, op="take")):
            return False
        try:
            deadline = time.monotonic() + 60
            while b"\n" not in self.buffer:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([self.fd], [], [], left)[0]:
                    raise OSError("counter did not answer")
                chunk = os.read(self.fd, 4096)
                if not chunk:
                    raise OSError("counter connection closed")
                self.buffer += chunk
            line, _, self.buffer = self.buffer.partition(b"\n")
            return json.loads(line).get("granted") is True
        except (OSError, ValueError):
            self.fd = None
            return False


def clip(text):
    return text if len(text) <= LOG_ARG_CHARS else text[:LOG_ARG_CHARS] + f"...[{len(text)} chars]"


def parse_request(gp):
    """One shim request: header 'X1L1' + 5 descriptors, then argv/env on the private reply socket."""
    message, ancillary, _, _ = gp.recvmsg(16, socket.CMSG_SPACE(8 * 4))
    fds = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds += list(struct.unpack(f"{len(data) // 4}i", data[:len(data) - len(data) % 4]))
    if not message and not fds:
        return "eof", None
    if message != b"X1L1" or len(fds) != 5:
        for fd in fds:
            os.close(fd)
        return "malformed", {"bytes": len(message), "descriptors": len(fds)}
    reply = fds[0]
    try:
        argc, envc, size = struct.unpack("<3I", read_exact(reply, 12, 10))
        if size > (64 << 20) or argc > 100000 or envc > 100000:
            raise OSError("oversized request")
        parts = read_exact(reply, size, 30).split(b"\0")[:-1]
        if len(parts) != argc + envc:
            raise OSError("inconsistent request")
        argv = [p.decode("utf-8", "surrogateescape") for p in parts[:argc]]
        env = {}
        for item in parts[argc:]:
            key, sep, value = item.decode("utf-8", "surrogateescape").partition("=")
            if sep:
                env[key] = value
    except (OSError, ValueError, struct.error):
        for fd in fds:
            os.close(fd)
        return "malformed", {"stage": "payload"}
    return "request", {"reply": reply, "stdin": fds[1], "stdout": fds[2], "stderr": fds[3], "cwd": fds[4],
                       "argv": argv, "env": env}


def cwd_path(fd):
    here = os.open(".", os.O_PATH | os.O_DIRECTORY)
    try:
        os.fchdir(fd)
        return os.getcwd()
    except OSError:
        return None
    finally:
        os.fchdir(here)
        os.close(here)


def lock_spec(rules):
    return "".join(f"{kind}\t{path}\n" for kind, path in rules)


def lock_memfd():
    """Sealed in-memory copy of x1lock.so (bytes from the trusted spec); the sandbox has no path to it."""
    import base64
    import fcntl
    fd = os.memfd_create("x1lock", os.MFD_ALLOW_SEALING)
    write_all(fd, base64.b64decode(SPEC["lock_b64"]))
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
    return fd


def supervise(service, request, launch):
    """Child of the gate: start the one interpreter on the shim's streams and relay its exit status."""
    reply = request["reply"]
    null = os.open("/dev/null", os.O_RDWR)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.dup2(null, 0)
    os.close(null)
    confirm_r, confirm_w = os.pipe()
    memfd = lock_memfd()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(request["stdin"], 0)
            os.dup2(request["stdout"], 1)
            os.dup2(request["stderr"], 2)
            os.fchdir(request["cwd"])
            keep = sorted((confirm_w, memfd))
            os.closerange(3, keep[0])
            os.closerange(keep[0] + 1, keep[1])
            os.closerange(keep[1] + 1, 65536)
            os.set_inheritable(confirm_w, True)
            os.set_inheritable(memfd, True)
            # Layer before exec: the shell allowlist plus the Python binary (to start it) and Python's reads.
            restrict(SPEC["bash_rules"] + SPEC["python_reads"] + SPEC["python_binary"])
            env = {k: v for k, v in request["env"].items() if not k.startswith("LD_") and not k.startswith("X1_LOCK")}
            # Layer inside the interpreter (x1lock.so constructor, before any Python code): the same without
            # the Python binary, so the running program cannot start Python again.
            env.update(LD_PRELOAD=f"/proc/self/fd/{memfd}", X1_LOCK_MEMFD=str(memfd), X1_LOCK_FD=str(confirm_w),
                       X1_LOCK_SPEC=lock_spec(SPEC["bash_rules"] + SPEC["python_reads"]))
            os.execve(SPEC["python"], [SPEC["python"], *request["argv"][1:]], env)
        finally:
            os._exit(127)
    os.close(memfd)
    os.close(confirm_w)
    for name in ("stdin", "stdout", "stderr", "cwd"):
        os.close(request[name])
    ready, _, _ = select.select([confirm_r], [], [], SPEC["confirm_timeout"])
    confirmed = bool(ready) and os.read(confirm_r, 1) == b"L"
    os.close(confirm_r)
    if not confirmed:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        service.send({"op": "note", "event": "interpreter_lock_not_confirmed", "launch": launch})
        try:
            write_all(reply, b"F" + struct.pack("<i", 126))
        except OSError:
            pass
        os._exit(0)
    service.send({"op": "started", "launch": launch, "pid": pid})
    status, shim_left = None, False
    while status is None:
        try:
            ready, _, _ = select.select([reply], [], [], 0.2)
        except InterruptedError:
            ready = []
        if ready and not shim_left:
            try:
                data = os.read(reply, 1)
            except OSError:
                data = b""
            if not data:
                shim_left = True  # the shim was stopped: stop the interpreter it was waiting for
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        done, value = os.waitpid(pid, os.WNOHANG)
        if done:
            status = value
    if os.WIFSIGNALED(status):
        message, report = b"S" + struct.pack("<i", os.WTERMSIG(status)), {"signal": os.WTERMSIG(status)}
    else:
        message, report = b"E" + struct.pack("<i", os.WEXITSTATUS(status)), {"exit_code": os.WEXITSTATUS(status)}
    service.send({"op": "exited", "launch": launch, "seconds": round(time.monotonic() - started, 3),
                  "shim_stopped_first": shim_left, **report})
    try:
        write_all(reply, message)
    except OSError:
        pass
    os._exit(0)


def main():
    service = Service()
    gp, sp = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    service.send({"op": "hello", "gate_pid": os.getpid(), "command_chars": len(COMMAND[-1]) if COMMAND else 0})
    wake_r, wake_w = os.pipe()
    os.set_blocking(wake_w, False)
    signal.set_wakeup_fd(wake_w)
    signal.signal(signal.SIGCHLD, lambda *_: None)
    bash = os.fork()
    if bash == 0:
        try:
            signal.set_wakeup_fd(-1)
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            os.dup2(sp.fileno(), GATE_FD, inheritable=True)
            restrict(SPEC["bash_rules"])
            os.environ["PATH"] = SPEC["bin"]
            os.execv("/bin/bash", ["/bin/bash", "--noprofile", "--norc", *COMMAND])
        finally:
            os.write(2, b"sandbox-shell: cannot start bash\n")
            os._exit(126)
    sp.close()
    bash_status, deadline, launches = None, None, 0
    while True:
        if bash_status is None:
            done, value = os.waitpid(bash, os.WNOHANG)
            if done:
                bash_status, deadline = value, time.monotonic() + SPEC["grace"]
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass  # reap supervisors
        except ChildProcessError:
            pass
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        if deadline is not None and timeout == 0:
            break
        try:
            ready, _, _ = select.select([gp, wake_r], [], [], timeout if timeout is not None else 5.0)
        except InterruptedError:
            continue
        if wake_r in ready:
            try:
                os.read(wake_r, 4096)
            except BlockingIOError:
                pass
        if gp not in ready:
            continue
        try:
            kind, request = parse_request(gp)
        except OSError:
            kind, request = "eof", None
        if kind == "eof":
            # Every holder of the request descriptor is gone: nothing more can be requested. Wait for bash.
            if bash_status is None:
                bash_status = os.waitpid(bash, 0)[1]
            break
        if kind == "malformed":
            service.send({"op": "note", "event": "malformed_request", **request})
            continue
        launches += 1
        argv = request["argv"]
        record = {"argv": [clip(a) for a in argv[:64]], "argc": len(argv), "argv_chars": sum(len(a) for a in argv),
                  "cwd": cwd_path(request["cwd"]), "request": launches}
        granted = service.take(record)
        if not granted:
            try:
                write_all(request["reply"], b"R")
            except OSError:
                pass
            for name in ("reply", "stdin", "stdout", "stderr", "cwd"):
                os.close(request[name])
            continue
        try:
            write_all(request["reply"], b"G")
        except OSError:
            pass
        supervisor = os.fork()
        if supervisor == 0:
            try:
                signal.set_wakeup_fd(-1)
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                gp.close()
                supervise(service, request, launches)
            finally:
                os._exit(0)
        for name in ("reply", "stdin", "stdout", "stderr", "cwd"):
            os.close(request[name])
    service.send({"op": "bye", "launch_requests": launches})
    if os.WIFSIGNALED(bash_status):
        signal.signal(os.WTERMSIG(bash_status), signal.SIG_DFL)
        os.kill(os.getpid(), os.WTERMSIG(bash_status))
    sys.exit(os.WEXITSTATUS(bash_status))


main()
