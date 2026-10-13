"""Shell and Python command classifier shared by the runners and the analysis (commands are parsed, never executed)."""

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shlex
import sys



ROOT = Path(__file__).resolve().parent
TOOLS = ("search", "python", "inspect", "workspace", "other")
OPERATIONS = ("schema", "nodes", "edges", "candidate_check", "structured_parse",
              "loop", "intersection", "sort", "aggregate", "text_filter", "output_limit",
              "script_write", "scratch_write", "graph_batch")
SEARCH = {"rg", "grep", "egrep", "fgrep"}
INSPECT = {"cat", "head", "tail", "sed", "awk", "wc", "cut", "sort", "uniq", "jq"}
WORKSPACE = {"ls", "pwd", "find", "du", "file", "stat", "which", "mkdir", "cd"}
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w]*)\1")
NODE_ID = re.compile(r'''["']node_id["']\s*:\s*(\d+)''')
MEASURES = ("calls", "seconds", "input_tokens", "output_tokens", "output_chars",
            "search_calls", "python_calls", "edge_calls", "graph_batch_calls",
            "intersection_calls", "repeat_calls", "execution_errors", "search_no_matches",
            "large_outputs", "returned_count", "attempts", "total_agent_seconds")


def read_json(path):
    return json.loads(Path(path).read_text())


def shell_body(command):
    words = shlex.split(command)
    if words and Path(words[0]).name in {"bash", "sh", "dash", "zsh"}:
        for i, token in enumerate(words[1:], 1):
            if token.startswith("-") and "c" in token and i + 1 < len(words):
                return words[i + 1]
    return command


def separate_heredocs(body):
    lines = body.splitlines(keepends=True)
    shell, sources = [], []
    i = 0
    while i < len(lines):
        header = lines[i]
        shell.append(header)
        i += 1
        match = HEREDOC.search(header)
        if not match:
            continue
        code = []
        while i < len(lines) and lines[i].strip() != match[2]:
            code.append(lines[i])
            i += 1
        if i == len(lines):
            raise ValueError("Unclosed heredoc")
        i += 1
        sources.append((header, "".join(code)))
    return "".join(shell), sources


def shell_segments(body):
    lexer = shlex.shlex(body, posix=True, punctuation_chars=";&|()<>\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segment = []
    for token in lexer:
        if token and all(c in ";&|()\n" for c in token):
            if segment:
                yield segment
            segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def program_args(segment):
    words = list(segment)
    if words and words[0] in {"for", "while", "done", "fi", "break"}:
        return "", []
    while words and (re.match(r"^[A-Za-z_]\w*=", words[0]) or words[0] in {"env", "command", "exec", "do", "then", "else", "if"}):
        words.pop(0)
    return (Path(words[0]).name, words[1:]) if words else ("", [])


def path_operations(strings):
    tags = set()
    for s in strings:
        if re.search(r"(?:^|/)nodes(?:/|$)", s) or s.endswith(".jsonl"):
            tags.add("nodes")
        if re.search(r"(?:^|/)edges(?:/|$)", s) or s.endswith(".tsv"):
            tags.add("edges")
        if "schema.json" in s or "DATA_FORMAT.txt" in s:
            tags.add("schema")
        if "candidate_ids.txt" in s or s == "is_candidate":
            tags.add("candidate_check")
    return tags


def python_operations(source):
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    tags = path_operations(n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str))
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
    if calls & {"load", "loads", "reader", "DictReader"}:
        tags.add("structured_parse")
    if any(isinstance(n, (ast.For, ast.While, ast.comprehension)) for n in nodes):
        tags.add("loop")
    if calls & {"intersection", "intersection_update"} or any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd) for n in nodes):
        tags.add("intersection")
    if calls & {"sort", "sorted", "nlargest", "nsmallest"}:
        tags.add("sort")
    if calls & {"Counter", "sum", "most_common"}:
        tags.add("aggregate")
    if calls & {"search", "findall", "match", "fullmatch", "lower", "casefold"}:
        tags.add("text_filter")
    if any(isinstance(n, ast.Slice) and n.upper is not None for n in nodes):
        tags.add("output_limit")
    if calls & {"write", "write_text", "dump"}:
        tags.add("scratch_write")
    if {"edges", "loop"} <= tags and calls & {"open", "read_text", "readlines"}:
        tags.add("graph_batch")
    return tags


def classify(command):
    tools, operations, programs, sources = set(), set(), [], []
    parse_error = 0
    unavailable = 0
    try:
        body = shell_body(command)
        masked, heredocs = separate_heredocs(body)
        segments = list(shell_segments(masked))
    except ValueError:
        return {"tools": ["other"], "operations": [], "programs": [], "shell_parse_errors": 1,
                "python_parse_errors": 0, "python_sources": 0, "python_source_unavailable": 0}
    for segment in segments:
        program, args = program_args(segment)
        if not program:
            continue
        programs.append(program)
        if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", program):
            tools.add("python")
            if "-c" in args and args.index("-c") + 1 < len(args):
                sources.append(args[args.index("-c") + 1])
            elif "<<" not in args and "<<-" not in args:
                unavailable += 1
        elif program in SEARCH:
            tools.add("workspace" if "--files" in args else "search")
        elif program in INSPECT:
            tools.add("inspect")
        elif program in WORKSPACE:
            tools.add("workspace")
        else:
            tools.add("other")
        if program in INSPECT or (program in SEARCH and "--files" not in args):
            operations.update(path_operations(args))
        if program in {"head", "tail", "cut"} or (program in SEARCH and any(x in args for x in ("-m", "--max-count", "-l", "--files-with-matches"))):
            operations.add("output_limit")
        if program == "sort":
            operations.add("sort")
        if program in {"wc", "uniq"}:
            operations.add("aggregate")
        if any(x in segment for x in (">", ">>")) or program == "tee":
            operations.add("scratch_write")
            if any(x.endswith(".py") for x in segment):
                operations.add("script_write")
    for header, source in heredocs:
        if any(re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", program_args(s)[0])
               for s in shell_segments(header)):
            sources.append(source)
    for source in sources:
        try:
            operations.update(python_operations(source))
        except (SyntaxError, ValueError):
            parse_error += 1
    return {"tools": sorted(tools), "operations": sorted(operations), "programs": programs,
            "shell_parse_errors": 0, "python_parse_errors": parse_error,
            "python_sources": len(sources), "python_source_unavailable": unavailable}


