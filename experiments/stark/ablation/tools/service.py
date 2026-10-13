"""Unix socket access to unchanged ARK retrieval and original node attributes."""

from functools import lru_cache
import json
import os
from pathlib import Path
import socketserver
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
ARK = PROJECT / "baselines/ark"
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
sys.path.insert(0, str(ARK))
from ark_graph import load_graph
from src.agents.graph_explorer.tools.search_in_graph import SearchInGraphTool
from src.agents.graph_explorer.tools.search_in_neighborhood import SearchInNeighborhoodTool

GRAPH_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()  # handlers run in threads; one tool_requests.jsonl line per write, never interleaved


@lru_cache(maxsize=3)
def graph(dataset):
    return load_graph(dataset)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Tools:
    def __init__(self, dataset, arm):
        if dataset not in ("prime", "mag", "amazon") or arm not in ("A1", "A2", "A5"):
            raise ValueError("Invalid tool service scope")
        self.dataset, self.arm = dataset, arm
        self.corpus = ROOT.parent / "corpus" / dataset
        with GRAPH_LOCK:
            self.graph = graph(dataset)

    def call(self, name, arguments):
        allowed = {"search": {"query", "k"}, "neighbors": {"node_id", "relation", "node_type", "query", "k"},
                   "get_node": {"node_id"}, "get_schema": set()}
        if name not in allowed or (name == "neighbors" and self.arm == "A1"):
            raise ValueError("Tool unavailable: " + name)
        if not isinstance(arguments, dict) or arguments.keys() - allowed[name]:
            raise ValueError("Unknown tool argument")
        if "node_id" in allowed[name] and (type(arguments.get("node_id")) is not int or arguments["node_id"] < 0):
            raise ValueError("node_id must be a nonnegative integer")
        for key in ("query", "relation", "node_type"):
            if key in arguments and not isinstance(arguments[key], str):
                raise ValueError(key + " must be a string")
        if name == "search":
            if "query" not in arguments:
                raise ValueError("query is required")
            k = arguments.get("k", 5)
            if type(k) is not int or not 1 <= k <= 30:
                raise ValueError("k must be between 1 and 30")
            return SearchInGraphTool(self.graph)({"query": arguments["query"], "size": k})
        if name == "neighbors":
            k = arguments.get("k", 20)
            if type(k) is not int or not 1 <= k <= 20:
                raise ValueError("k must be between 1 and 20")
            tool = SearchInNeighborhoodTool(self.graph)
            tool.MAX_RESULTS_SHOWN = k
            return tool({"node_index": arguments["node_id"], "edge_type": arguments.get("relation", ""),
                         "node_type": arguments.get("node_type", ""), "query": arguments.get("query", "")})
        if name == "get_schema":
            return self.corpus.joinpath("schema.json").read_text()
        node_id = arguments["node_id"]
        shard = self.corpus / "nodes" / f"{node_id // 10000:04d}.jsonl"
        if shard.exists():
            with shard.open() as source:
                for line in source:
                    row = json.loads(line)
                    if row["node_id"] == node_id:
                        return dumps(row)
        raise ValueError("Unknown node_id")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(1024 * 1024 + 1)
        started = time.time()
        request = None
        try:
            if len(raw) > 1024 * 1024:
                raise ValueError("Request exceeds 1 MiB")
            request = json.loads(raw)
            if not isinstance(request, dict) or set(request) != {"name", "arguments"}:
                raise ValueError("Expected name and arguments")
            output = {"ok": True, "text": self.server.tools.call(request["name"], request["arguments"])}
        except Exception as error:
            output = {"ok": False, "error": str(error)}
        with LOG_LOCK, self.server.log.open("a") as log:
            log.write(dumps({"time": started, "seconds": time.time() - started,
                             "phase": getattr(self.server, "phase", "preflight"),
                             "request": request, "response": output}) + "\n")
        self.wfile.write((dumps(output) + "\n").encode())


class PersistentHandler(Handler):
    def handle(self):
        while self.rfile.peek(1):
            super().handle()


class Server(socketserver.ThreadingUnixStreamServer):
    # One thread per connection; listen backlog 1024, since parallel tool calls in one turn arrive at once.
    request_queue_size = 1024


def start(attempt, dataset, arm):
    path = Path(attempt) / "kb.sock"
    path.unlink(missing_ok=True)
    server = Server(str(path), PersistentHandler if arm == "A5" else Handler)
    server.daemon_threads = True
    server.tools = Tools(dataset, arm)
    server.log = Path(attempt) / "tool_requests.jsonl"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
