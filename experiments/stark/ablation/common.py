"""Shared preparation of the tool variants: corpus views, tool paragraphs, sandbox shells, isolation probes."""

import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STARK = ROOT.parent
FIXES = STARK / "harness_fixes"
sys.path.insert(0, str(FIXES))
import adapter
val = adapter.runner
base, stark = val.base, val.stark
ARMS = ("H", "A1", "A2", "A3", "A5", "A6")
DATASETS = ("prime", "mag", "amazon")
EXPECTED_TOOLS = {"A1": {"search", "get_node", "get_schema"},
                  "A2": {"search", "get_node", "get_schema", "neighbors"},
                  "A3": {"read", "bash"}, "A5": {"read", "bash"}, "A6": {"read", "bash"}}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, value)


def read(path):
    return base.read_json(Path(path))


def key(row):
    return f"{row['dataset']}:{row['query_id']}"


def jobkey(row):
    return f"{row['group']}:{key(row)}"


def result_path(job):
    return ROOT / "runs" / job["group"] / job["dataset"] / "val" / job["query_id"] / "result.json"


def issue(text):
    with (ROOT / "issues.log").open("a") as out:
        out.write(f"\n- {base.now()}: {text}\n")


def assets():
    (ROOT / "runs").mkdir(exist_ok=True)
    for dataset in DATASETS:
        source, target = STARK / "corpus" / dataset, STARK / "kb" / dataset
        (target / "nodes").mkdir(parents=True, exist_ok=True)
        linked = [source / "candidate_ids.txt", *sorted((source / "nodes").glob("*.jsonl"))]
        for path in linked:
            destination = target / path.relative_to(source)
            if not destination.exists():
                os.link(path, destination)
            if not os.path.samefile(path, destination):
                raise ValueError("A6 node is not a source hardlink")
        schema = read(source / "schema.json")
        for field in ("relation_types", "stored_edges_by_type", "graph_semantics"):
            schema.pop(field, None)
        schema["representation"] = schema["representation"].replace(" Stored edge rows preserved.", "")
        save(target / "schema.json", schema)
        lines = (source / "DATA_FORMAT.txt").read_text().splitlines(keepends=True)
        (target / "DATA_FORMAT.txt").write_text("".join(line for line in lines if not line.startswith(
            ("Graph edges:", "RR is the relation", "For adjacency,", "Preserve relation meaning"))))
        save(target / "manifest.json", {"source": str(source), "hardlinks": len(linked),
             "nodes": [{"path": str(p.relative_to(source)), "device": p.stat().st_dev, "inode": p.stat().st_ino,
                        "bytes": p.stat().st_size} for p in linked]})
    workspace = ROOT / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "README.txt").write_text("Knowledge-base data is available through the supplied tools.\n")
    (ROOT / "prompts").mkdir(exist_ok=True)
    file_text = ("Corpus directory: {corpus}\nRead DATA_FORMAT.txt and schema.json for the file format, entity types, relation "
                 "types, and graph interpretation. Nodes and edges are complete local text files. "
                 "Use shell tools, rg, or Python 3 standard-library modules to search and read them. "
                 "The corpus is read-only. Writable scratch directory: {scratch}\n")
    descriptions = {
        "H": file_text,
        "A1": "Available tools: get_schema(), search(query, k=5), get_node(node_id). Call get_schema first to understand node and relation types. "
              "Search returns up to 30 BM25 matches with benchmark IDs, names, types, and summary context. get_node returns the complete record "
              "including attributes and is_candidate. Use node_ids from returned records.\n",
        "A2": "Available tools: get_schema(), search(query, k=5), get_node(node_id), neighbors(node_id, relation, node_type, query, k=20). "
              "Call get_schema first to understand node and relation types. Search returns up to 30 BM25 matches with benchmark IDs, names, "
              "types, and summary context. get_node returns the complete record including attributes and is_candidate. "
              "neighbors examines one hop in either direction, optionally filters relation and node type, and uses keyword matching when query is supplied; "
              "k is at most 20. Optional filters may be omitted or empty. Use node_ids from returned records.\n",
        "A3": file_text + "General-purpose programming languages are unavailable; use only shell text tools and bash syntax.\n",
        "A5": "Available tools: read and bash, including Python 3. Knowledge-base files cannot be read directly. "
              "In Python, use: import sys; sys.path.insert(0, {client_repr}); import kbtools; kb = kbtools.Client({socket_repr}). "
              "Call kb.get_schema() first to understand node and relation types. kb.search(query, k=5) returns up to 30 BM25 matches. "
              "kb.get_node(node_id) returns the complete record including attributes and is_candidate. "
              "kb.neighbors(node_id, relation='', node_type='', query='', k=20) examines one hop in either direction with optional filters "
              "and keyword matching; k is at most 20. All methods return text. Writable scratch directory: {scratch}\n",
        "A6": file_text.replace("Nodes and edges are complete local text files", "Nodes are complete local text files"),
    }
    for arm, text in descriptions.items():
        (ROOT / "prompts" / f"{arm}.txt").write_text(text)


def prompt(row, attempt, arm):
    original = val.variants.prompt_for(row, attempt, "p1")
    prefix, _, _ = original.partition("Corpus directory:")
    corpus = STARK / "kb" / row["dataset"] if arm == "A6" else Path(row["corpus_dir"])
    tail = (ROOT / "prompts" / f"{arm}.txt").read_text().format(
        corpus=corpus, scratch=attempt / "tmp", client_repr=repr(str(attempt)),
        socket_repr=repr(str(attempt / "kb.sock")))
    return prefix + tail


def prepare(attempt, row, arm, protocol):
    harness = "codex" if arm == "H" else "pi"
    chosen = dict(row)
    if arm in ("A1", "A2", "A5"):
        chosen["corpus_dir"] = str(ROOT / "workspace")
    elif arm == "A6":
        chosen["corpus_dir"] = str(STARK / "kb" / row["dataset"])
    settings = dict(protocol)
    if harness == "pi":
        settings["pi_max_turns"] = 300
    command, env = adapter.prepare_attempt(attempt, chosen, settings, harness, prompt(row, attempt, arm))
    if harness == "pi":
        config = read(attempt / "agent/settings.json")
        config["retry"] = {"enabled": True, "maxRetries": 2, "baseDelayMs": 15000, "maxDelayMs": 120000}
        if arm == "A3":
            local = attempt / "sandbox"
            (local / "bin").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "sandbox/exec_only.py", local / "exec_only.py")
            for name in "rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash".split():
                destination = local / "bin" / name
                source = shutil.which(name, path=f"{base.CODEX_ROOT}/codex-path:/usr/bin:/bin")
                if destination.is_symlink() and destination.readlink() != Path(source):
                    destination.unlink()
                if not destination.exists():
                    destination.symlink_to(source)
            config["shellPath"] = str(ROOT / "sandbox/sandbox-shell.sh")
            boundary = command.index("--")
            command[boundary:boundary] = ["--read", config["shellPath"]]
        save(attempt / "agent/settings.json", config)
    if arm in ("A1", "A2"):
        command[command.index("--tools") + 1] = ""
        command += ["--extra-arg=--no-tools", f"--extra-arg=-e {ROOT / 'tools/extension.mjs'}"]
        boundary = command.index("--")
        command[boundary:boundary] = ["--read", str(ROOT / "tools/extension.mjs")]
        env.update(DCI_KB_SOCKET=str(attempt / "kb.sock"), DCI_KB_ARM=arm)
    if arm == "A5":
        shutil.copyfile(ROOT / "tools/kbtools.py", attempt / "_kbtools_base.py")
        shutil.copyfile(ROOT / "a5_variant/kbtools.py", attempt / "kbtools.py")
        config = read(attempt / "agent/settings.json")
        config["shellPath"] = str(ROOT / "a5_variant/shell.py")
        save(attempt / "agent/settings.json", config)
        boundary = command.index("--")
        command[boundary:boundary] = ["--read", config["shellPath"]]
    save(attempt / "command.json", command)
    save(attempt / "arm.json", {"arm": arm, "harness": harness, "cwd": chosen["corpus_dir"],
                               "expected_tools": sorted(EXPECTED_TOOLS.get(arm, {"exec_command", "write_stdin"}))})
    return command, env, Path(chosen["corpus_dir"])


def isolation_check(attempt, row, arm, command, env, cwd):
    denied = [STARK / "private", STARK / "public", STARK.parents[1] / "STaRK/data"]
    if arm in ("A1", "A2", "A5", "A6"):
        denied.extend(STARK / "corpus" / dataset for dataset in DATASETS)
    denied.extend([ROOT / "reservation.json", ROOT / "selection.json", ROOT / "runs"])
    code = "import os\nfor path in " + repr([str(p) for p in denied]) + ":\n"
    code += " try:\n  fd=os.open(path,os.O_RDONLY)\n except (PermissionError, FileNotFoundError):\n  continue\n else:\n  os.close(fd)\n  raise AssertionError('Readable denied path: '+path)\n"
    outer = command[:command.index("--") + 1]
    probe = subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", code], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise RuntimeError("Outer isolation probe failed: " + probe.stderr)
    checks = {"labels_and_other_runs": "denied", "original_corpus": "denied" if arm in ("A1", "A2", "A5", "A6") else "read-only"}
    if arm == "A3":
        shell = str(ROOT / "sandbox/sandbox-shell.sh")
        commands = ["python3 -c 1", "awk 1", "perl -e 1", "node -e 1",
                    "find . -maxdepth 0 -exec python3 -c 1 \\;", "echo x | xargs python3 -c 1",
                    "/usr/bin/python3 -c 1", "/usr/bin/awk 1",
                    "/lib64/ld-linux-x86-64.so.2 /usr/bin/python3 -c 1",
                    "/lib64/ld-linux-x86-64.so.2 /usr/bin/perl -e 1"]
        for item in commands:
            result = subprocess.run(outer + [shell, "-c", item], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
            # find itself returns zero even when its -exec child is denied.
            if result.returncode == 0 and "Permission denied" not in result.stderr and "No such file" not in result.stderr:
                raise AssertionError("Whitelist allowed: " + item)
        allowed = "set -e; pwd; ls >/dev/null; cat schema.json | head -n 1; tail -n 1 schema.json; rg dataset schema.json; " \
                  "sed -n 1p schema.json; wc -l schema.json; echo x | cut -c 1 | sort; comm /dev/null /dev/null; " \
                  "find . -maxdepth 0; echo x | xargs echo; bash -c 'echo ok'"
        permitted = subprocess.run(outer + [shell, "-c", allowed], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
        if permitted.returncode:
            raise RuntimeError("Allowed shell command failed: " + permitted.stderr)
        checks["executable_allowlist"] = "passed"
    if arm == "A5":
        probe = f"import sys; sys.path.insert(0,{str(attempt)!r}); import kbtools; c=kbtools.Client({str(attempt / 'kb.sock')!r}); "
        probe += "assert c.get_schema(); assert c.get_node(0); assert c.search('cell'); assert c.neighbors(0)"
        shell = read(attempt / "agent/settings.json")["shellPath"]
        client = subprocess.run(outer + [shell, "-c", "python3 -S -c " + shlex.quote(probe)], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=240)
        if client.returncode:
            raise RuntimeError("A5 sandbox socket probe failed: " + client.stderr)
        checks["unix_socket_all_four_tools"] = "passed"
    save(attempt / "ablation_preflight.json", checks)
    return checks


def validate_tools(attempt, arm):
    if arm == "H":
        requests = list(base.read_jsonl(attempt / "model_requests.jsonl"))
        if not requests:
            raise AssertionError("No native Codex model requests")
        return {"requests": len(requests), "tools": "native Codex shell tools"}
    events = list(adapter.pi_context.requests(attempt / "artifacts/events.jsonl"))
    if not events:
        raise AssertionError("No provider requests")
    for _, _, event in events:
        names = {tool.get("name", tool.get("function", {}).get("name")) for tool in event["payload"].get("tools", [])}
        if names != EXPECTED_TOOLS[arm]:
            raise AssertionError(f"{arm} provider tools differ: {sorted(names)}")
    return {"requests": len(events), "tools": sorted(EXPECTED_TOOLS[arm]), "status": "passed"}
