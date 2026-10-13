"""Harness-held start counter for X1 ("One program"): one Python start per attempt.

One server per attempt, in the runner process, on <item>/<attempt>.gate.sock (outside the attempt directory).
The counter lives only in this process's memory. The sandbox cannot connect to the socket (seccomp denies
connect, and the path is outside its filesystem rules); each bash tool call
reaches it only through the connection that x1_shell.py opens outside the sandbox and the gate keeps on a
descriptor that bash and Python never receive. The protocol has one operation that changes state, "take";
there is no operation that adds a start. Every message is logged to a file outside the attempt directory.
"""
import json
from pathlib import Path
import socketserver
import threading
import time
from datetime import datetime, timezone

INFO_OPS = {"hello", "started", "exited", "note", "bye"}


def socket_path(attempt):
    """Next to the attempt directory, in the item directory, where the tool sandbox can neither read nor create
    files (it may write only inside the attempt directory), so it cannot remove or impersonate the counter."""
    attempt = Path(attempt)
    return attempt.parent / f"{attempt.name}.gate.sock"


def now():
    return datetime.now(timezone.utc).isoformat()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        gate = self.server.gate
        connection = gate.open_connection()
        try:
            for raw in self.rfile:
                try:
                    message = json.loads(raw)
                    op = message.get("op") if isinstance(message, dict) else None
                except ValueError:
                    message, op = {"bytes": len(raw)}, None
                if op == "take":
                    granted = gate.take(connection, message)
                    self.wfile.write((json.dumps({"granted": granted}) + "\n").encode())
                    self.wfile.flush()
                elif op in INFO_OPS:
                    gate.record(connection, op, message)
                else:
                    gate.record(connection, "invalid", {"message": str(message)[:500]})
        finally:
            gate.record(connection, "closed", {})


class Gate:
    def __init__(self, attempt, log_path, starts=1):
        self.attempt = Path(attempt)
        self.socket_path = socket_path(self.attempt)
        self.log_path = Path(log_path)
        self.starts_per_phase = starts
        self.lock = threading.Lock()
        self.remaining = starts
        self.phase = "preflight"
        self.events = []
        self.connections = 0
        self.socket_path.unlink(missing_ok=True)
        self.server = socketserver.ThreadingUnixStreamServer(str(self.socket_path), Handler)
        self.server.daemon_threads = True
        self.server.gate = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.record(None, "counter_ready", {"starts": starts})

    def open_connection(self):
        with self.lock:
            self.connections += 1
            return self.connections

    def record(self, connection, op, message):
        entry = {"t": now(), "phase": self.phase, "connection": connection, "op": op,
                 **{k: v for k, v in message.items() if k != "op"}}
        with self.lock:
            self.events.append(entry)
            with self.log_path.open("a") as out:
                out.write(json.dumps(entry) + "\n")

    def take(self, connection, message):
        with self.lock:
            granted = self.remaining > 0
            if granted:
                self.remaining -= 1
            remaining = self.remaining
        self.record(connection, "take", {**message, "granted": granted, "remaining_after": remaining})
        return granted

    def begin_agent_phase(self):
        """Called by the runner after the preflight probes and before the model starts: one fresh start."""
        with self.lock:
            self.phase = "agent"
            self.remaining = self.starts_per_phase
        self.record(None, "agent_phase", {"starts": self.starts_per_phase})

    def summary(self, phase="agent"):
        with self.lock:
            events = list(self.events)
        return summarize(events, phase, self.log_path)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.socket_path.unlink(missing_ok=True)
        time.sleep(0)


def summarize(events, phase, log_path):
    events = [e for e in events if e.get("phase") == phase]
    takes = [e for e in events if e["op"] == "take"]
    exited = [e for e in events if e["op"] == "exited"]
    return {"phase": phase, "tool_call_connections": sum(e["op"] == "hello" for e in events),
            "launch_requests": len(takes), "granted": sum(bool(e.get("granted")) for e in takes),
            "refused": sum(not e.get("granted") for e in takes),
            "started": sum(e["op"] == "started" for e in events),
            "exit": [{k: e.get(k) for k in ("exit_code", "signal", "seconds", "shim_stopped_first")} for e in exited],
            "notes": [{k: v for k, v in e.items() if k not in ("t", "phase", "connection", "op")}
                      for e in events if e["op"] in ("note", "invalid")],
            "first_grant_argv": next((e.get("argv") for e in takes if e.get("granted")), None),
            "first_grant_cwd": next((e.get("cwd") for e in takes if e.get("granted")), None),
            "log": str(log_path)}


def summarize_log(log_path, phase="agent"):
    events = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    return summarize(events, phase, log_path)
