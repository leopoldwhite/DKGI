"""Refine saved Python commands into auditable graph-operation features.

This is conservative, intraprocedural AST analysis, not execution or a Python
interpreter. Labels describe code patterns; branches are not dynamically traced.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re

import analyze_behavior as base


LABELS = {
    "edge_read": "Read edge data", "edge_scan": "Iterate edge rows",
    "relation_sweep": "Loop over relation-file patterns", "multiple_relations": "Read multiple named relations",
    "scalar_seed": "Filter endpoints by one ID", "set_seed": "Filter endpoints by an ID collection",
    "seed_loop": "Repeat lookup over a seed collection", "both_directions": "Check both endpoint directions",
    "neighbor_collect": "Collect graph-derived IDs or rows", "adjacency_index": "Build endpoint-keyed adjacency buckets",
    "chained_expansion": "Use earlier edge results in a later edge filter",
    "intersection": "Intersect graph-derived sets", "union": "Union graph-derived sets",
    "difference": "Subtract graph-derived sets", "node_enrichment": "Look up nodes using graph results",
    "candidate_filter": "Filter candidate eligibility", "node_type_filter": "Filter node types",
    "attribute_filter": "Filter node attributes", "count_neighbors": "Count graph-derived results",
    "sort_results": "Sort graph-derived results", "subprocess_search": "Search edges from Python subprocess",
}


@dataclass(frozen=True)
class Value:
    paths: frozenset = frozenset()
    reads: frozenset = frozenset()
    rows: frozenset = frozenset()
    ends: frozenset = frozenset()
    origins: frozenset = frozenset()
    collection: bool = False
    seed_loop: bool = False


def merged(*values):
    return Value(**{f.name: (any(getattr(v, f.name) for v in values) if f.type is bool else
                            frozenset().union(*(getattr(v, f.name) for v in values))) for f in fields(Value)})


def derived(value, collection=False, seed_loop=False):
    return Value(paths=value.paths, origins=value.origins | value.reads | value.rows,
                 collection=collection, seed_loop=seed_loop)


def name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def root_name(node):
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def path_kind(path, kind):
    return bool(re.search(rf"(?:^|/){kind}(?:/|$)", path))


class GraphCode:
    def __init__(self, source):
        self.tree = ast.parse(source)
        self.lines = source.splitlines()
        self.env = {}
        self.evidence = defaultdict(set)
        self.reads = {}
        self.directions = defaultdict(set)
        self.direction_lines = defaultdict(set)
        self.chains = set()
        self.called = {n.func.id for n in ast.walk(self.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    def mark(self, label, node):
        self.evidence[label].add(node.lineno)

    def value(self, node):
        if node is None:
            return Value()
        if isinstance(node, ast.Name):
            return self.env.get(node.id, Value())
        if isinstance(node, ast.Constant):
            return Value(paths=frozenset([node.value])) if isinstance(node.value, str) and any(path_kind(node.value, k) for k in ("nodes", "edges")) else Value()
        if isinstance(node, ast.JoinedStr):
            pattern = "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "*" for v in node.values)
            values = merged(*(self.value(v) for v in node.values))
            return merged(values, Value(paths=frozenset([pattern])))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            items = node.elts if not isinstance(node, ast.Dict) else [x for x in node.keys + node.values if x]
            return merged(*(self.value(x) for x in items), Value(collection=True))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            old = dict(self.env)
            for gen in node.generators:
                self.bind_loop(gen.target, self.value(gen.iter), gen)
                for test in gen.ifs:
                    self.predicate(test)
            values = [self.value(node.key), self.value(node.value)] if isinstance(node, ast.DictComp) else [self.value(node.elt)]
            self.env = old
            return derived(merged(*values), collection=True)
        if isinstance(node, ast.Subscript):
            receiver, index = self.value(node.value), self.value(node.slice)
            if receiver.rows and isinstance(node.slice, ast.Slice):
                return receiver
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            side = 0 if key in (0, "source_id") else 1 if key in (1, "target_id") else None
            if receiver.rows and side is not None:
                return Value(ends=frozenset((r, side) for r in receiver.rows), origins=receiver.rows)
            if any(path_kind(p, "nodes") for p in receiver.paths) and index.origins:
                self.mark("node_enrichment", node)
            return derived(merged(receiver, index), collection=receiver.collection)
        if isinstance(node, ast.Call):
            method = name(node.func)
            receiver = self.value(node.func.value) if isinstance(node.func, ast.Attribute) else Value()
            args = [self.value(x) for x in node.args]
            v = merged(receiver, *args)
            edge_paths = {p for p in v.paths if path_kind(p, "edges")}
            if edge_paths and method in {"open", "read_text", "read_bytes", "check_output", "run"}:
                read_id = f"{node.lineno}:{node.col_offset}"
                self.reads.setdefault(read_id, set()).update(edge_paths)
                self.mark("edge_read", node)
                if method in {"check_output", "run"}:
                    self.mark("subprocess_search", node)
                if v.seed_loop and any("*" in p or p.rstrip("/") == "edges" for p in edge_paths):
                    self.mark("relation_sweep", node)
                return Value(paths=frozenset(edge_paths), reads=frozenset([read_id]))
            if method in {"open", "read_text"} and any(path_kind(p, "nodes") for p in v.paths) and v.origins:
                self.mark("node_enrichment", node)
            if method == "get" and receiver.rows and node.args and isinstance(node.args[0], ast.Constant):
                key = node.args[0].value
                if key in ("source_id", "target_id"):
                    return Value(ends=frozenset((r, int(key == "target_id")) for r in receiver.rows), origins=receiver.rows)
            if method in {"add", "append", "extend", "update"} and isinstance(node.func, ast.Attribute):
                added = merged(*args)
                if added.origins or added.rows:
                    self.mark("neighbor_collect", node)
                    container = node.func.value
                    bucket = self.value(container.slice) if isinstance(container, ast.Subscript) else Value()
                    if isinstance(container, ast.Call) and name(container.func) == "setdefault" and container.args:
                        bucket = self.value(container.args[0])
                    if bucket.ends and (added.ends or added.origins):
                        self.mark("adjacency_index", node)
                target = root_name(node.func.value)
                if target:
                    self.env[target] = derived(merged(self.env.get(target, Value()), added), collection=True)
            set_methods = {"intersection": "intersection", "intersection_update": "intersection",
                           "union": "union", "difference": "difference", "difference_update": "difference"}
            if method in set_methods and v.origins:
                self.mark(set_methods[method], node)
            if method in {"len", "Counter", "most_common"} and v.origins:
                self.mark("count_neighbors", node)
            if method in {"sort", "sorted", "nlargest", "nsmallest"} and v.origins:
                self.mark("sort_results", node)
            if method in {"set", "list", "dict", "tuple", "range", "enumerate", "zip", "defaultdict", "Counter", "setdefault", "keys", "values", "items", *set_methods}:
                return derived(v, collection=True)
            return v
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            value = merged(left, right)
            op = {ast.BitAnd: "intersection", ast.BitOr: "union", ast.Sub: "difference"}.get(type(node.op))
            if op and value.collection and value.origins:
                self.mark(op, node)
            return value
        if isinstance(node, ast.Compare):
            self.compare(node)
        return merged(*(self.value(child) for child in ast.iter_child_nodes(node)))

    def compare(self, node):
        for left_node, op, right_node in zip([node.left, *node.comparators[:-1]], node.ops, node.comparators):
            left, right = self.value(left_node), self.value(right_node)
            if not isinstance(op, (ast.Eq, ast.In)):
                continue
            pairs = [(left, right, right_node), (right, left, left_node)]
            for endpoint, seed, seed_node in pairs:
                if not endpoint.ends or seed.ends:
                    continue
                if isinstance(seed_node, ast.Constant) and (seed_node.value is None or isinstance(seed_node.value, bool) or
                        isinstance(seed_node.value, str) and not seed_node.value.isdigit()):
                    continue
                label = "set_seed" if isinstance(op, ast.In) and endpoint is left else "scalar_seed"
                self.mark(label, node)
                if seed.seed_loop:
                    self.mark("seed_loop", node)
                for read_id, side in endpoint.ends:
                    self.directions[(read_id, ast.dump(seed_node), label)].add(side)
                    self.direction_lines[(read_id, ast.dump(seed_node), label)].add(node.lineno)
                    for previous in seed.origins - {read_id}:
                        self.mark("chained_expansion", node)
                        self.chains.add((previous, read_id, node.lineno))
            if ((any(path_kind(p, "nodes") for p in left.paths) and right.origins) or
                    (any(path_kind(p, "nodes") for p in right.paths) and left.origins)):
                self.mark("node_enrichment", node)

    def predicate(self, node):
        value = self.value(node)
        strings = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        if "is_candidate" in strings:
            self.mark("candidate_filter", node)
        if "node_type" in strings:
            self.mark("node_type_filter", node)
        methods = {name(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
        if any(path_kind(p, "nodes") for p in value.paths) and (methods & {"search", "match", "lower", "casefold"} or
                any(isinstance(n, ast.Compare) and any(isinstance(o, ast.In) for o in n.ops) and
                    isinstance(n.left, ast.Constant) and isinstance(n.left.value, str) for n in ast.walk(node))):
            self.mark("attribute_filter", node)

    def bind(self, target, value, expression=None):
        if isinstance(target, ast.Name):
            self.env[target.id] = derived(value, True) if value.collection else value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(expression, ast.Call) and name(expression.func) == 'map' and len(expression.args) >= 2:
                expression = expression.args[1]
            if isinstance(expression, (ast.Tuple, ast.List)) and len(expression.elts) == len(target.elts):
                for dest, item in zip(target.elts, expression.elts):
                    self.bind(dest, self.value(item), item)
            elif value.rows and len(target.elts) == 2:
                for side, dest in enumerate(target.elts):
                    self.bind(dest, Value(ends=frozenset((r, side) for r in value.rows), origins=value.rows))
            else:
                for dest in target.elts:
                    self.bind(dest, derived(value))
        elif isinstance(target, ast.Subscript):
            index = self.value(target.slice)
            key = root_name(target.value)
            if index.ends and value.origins:
                self.mark("adjacency_index", target)
            if key:
                self.env[key] = derived(merged(self.env.get(key, Value()), value, index), True)

    def bind_loop(self, target, iterator, node):
        if iterator.reads:
            self.mark("edge_scan", target)
            value = Value(rows=iterator.reads, origins=iterator.reads)
        else:
            file_pattern = any(path_kind(p, 'edges') and ('*' in p or p.rstrip('/') == 'edges') for p in iterator.paths)
            value = derived(iterator, seed_loop=(iterator.collection or file_pattern) and not iterator.origins)
        self.bind(target, value)

    def statements(self, statements):
        for node in statements:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = self.value(node.value)
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    self.bind(target, value, node.value)
            elif isinstance(node, ast.AugAssign):
                self.bind(node.target, self.value(ast.copy_location(ast.BinOp(node.target, node.op, node.value), node)))
            elif isinstance(node, ast.With):
                for item in node.items:
                    value = self.value(item.context_expr)
                    if item.optional_vars:
                        self.bind(item.optional_vars, value)
                self.statements(node.body)
            elif isinstance(node, ast.For):
                self.bind_loop(node.target, self.value(node.iter), node)
                self.statements(node.body)
                self.statements(node.orelse)
            elif isinstance(node, (ast.If, ast.While)):
                self.predicate(node.test)
                self.statements(node.body)
                self.statements(node.orelse)
            elif isinstance(node, ast.Try):
                self.statements(node.body)
                for handler in node.handlers:
                    self.statements(handler.body)
                self.statements(node.orelse)
                self.statements(node.finalbody)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in self.called:
                    old = dict(self.env)
                    for arg in node.args.args:
                        self.env[arg.arg] = Value()
                    self.statements(node.body)
                    self.env = old
            elif isinstance(node, (ast.Expr, ast.Return)):
                self.value(node.value)

    def result(self):
        self.statements(self.tree.body)
        if self.reads:
            paths = set().union(*self.reads.values())
            fixed = {p for p in paths if re.search(r"edges/\d+\.tsv$", p)}
            if len(fixed) > 1:
                self.evidence["multiple_relations"].update(int(k.split(":")[0]) for k in self.reads)
            for key, sides in self.directions.items():
                if sides == {0, 1}:
                    self.evidence["both_directions"].update(self.direction_lines[key])
        else:
            self.evidence.clear()
        return {"labels": sorted(self.evidence),
                "evidence": {label: [{"line": line, "code": self.lines[line - 1].strip()} for line in sorted(lines)]
                             for label, lines in self.evidence.items()},
                "reads": {key: sorted(paths) for key, paths in self.reads.items()},
                "chains": [{"from_read": a, "to_read": b, "line": line} for a, b, line in sorted(self.chains)],
                "imports": sorted({alias.name for n in ast.walk(self.tree) if isinstance(n, ast.Import) for alias in n.names} |
                                  {n.module for n in ast.walk(self.tree) if isinstance(n, ast.ImportFrom) and n.module}),
                "called_function_definitions": sum(isinstance(n, ast.FunctionDef) and n.name in self.called for n in ast.walk(self.tree))}


def classify_source(source):
    return GraphCode(source).result()


def classify_command(command):
    body = base.shell_body(command)
    shell, heredocs = base.separate_heredocs(body)
    sources = []
    for segment in base.shell_segments(shell):
        program, args = base.program_args(segment)
        if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", program) and "-c" in args:
            sources.append(args[args.index("-c") + 1])
    for header, source in heredocs:
        if any(re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", base.program_args(s)[0]) for s in base.shell_segments(header)):
            sources.append(source)
    programs = []
    for source in sources:
        try:
            programs.append({"source": source, **classify_source(source), "parse_error": None})
        except (ValueError, SyntaxError) as error:
            programs.append({"source": source, "labels": [], "parse_error": str(error)})
    return {"labels": sorted({tag for p in programs for tag in p['labels']}), "programs": programs}


def summarize(query_rows, command_rows):
    summary = {}
    for dataset in ["ALL", "amazon", "mag", "prime"]:
        summary[dataset] = {}
        for outcome, hit in [("all", None), ("correct", 1), ("wrong", 0)]:
            queries = [q for q in query_rows if (dataset == "ALL" or q["dataset"] == dataset) and (hit is None or q["hit_at_1"] == hit)]
            commands = [c for c in command_rows if (dataset == "ALL" or c["dataset"] == dataset) and (hit is None or c["hit_at_1"] == hit)]
            summary[dataset][outcome] = {"queries": len(queries), "labels": {
                tag: {"queries": sum(q[tag] > 0 for q in queries), "rate": sum(q[tag] > 0 for q in queries) / len(queries) if queries else None,
                      "calls": sum(tag in c["labels"] for c in commands),
                      "exit0_calls": sum(tag in c["labels"] and c["exit_code"] == 0 for c in commands),
                      "exit0_queries": len({c["query_id"] + ':' + c["dataset"] for c in commands if tag in c["labels"] and c["exit_code"] == 0})}
                for tag in LABELS}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.ROOT / "behavior_710")
    parser.add_argument("--output", type=Path, default=base.ROOT / "behavior_710/graph")
    args = parser.parse_args()
    with (args.input / "per_query.csv").open() as source:
        original = list(csv.DictReader(source))
    queries = {(r['dataset'], r['query_id']): {"dataset": r['dataset'], "query_id": r['query_id'], "hit_at_1": int(r['hit_at_1']),
                                              **dict.fromkeys(LABELS, 0)} for r in original}
    commands, checks, seen, imports = [], Counter(), set(), Counter()
    for line in (args.input / "commands.jsonl").open():
        command = json.loads(line)
        key = command['dataset'], command['query_id']
        identity = (*key, command['step'])
        if identity in seen or key not in queries or command['hit_at_1'] != queries[key]['hit_at_1']:
            raise ValueError('Duplicate command or inconsistent query label')
        seen.add(identity)
        checks['all_shell_calls'] += 1
        checks['upstream_shell_parse_errors'] += command.get('shell_parse_errors', 0)
        if 'python' not in command['tools']:
            continue
        checks['python_shell_calls'] += 1
        try:
            result = classify_command(command['command'])
        except ValueError:
            checks['shell_parse_errors'] += 1
            result = {'labels': [], 'programs': []}
        checks['python_programs'] += len(result['programs'])
        checks['python_parse_errors'] += sum(bool(p.get('parse_error')) for p in result['programs'])
        checks['graph_calls'] += bool(result['labels'])
        for program in result['programs']:
            if program['labels']:
                checks['graph_programs'] += 1
                imports.update(program['imports'])
        for tag in result['labels']:
            queries[key][tag] += 1
        commands.append({**{k: command[k] for k in ('dataset', 'query_id', 'hit_at_1', 'step', 'line', 'trace_path', 'exit_code')}, **result})
    for row in original:
        if sum(1 for d, q, _ in seen if (d, q) == (row['dataset'], row['query_id'])) != int(row['calls']):
            raise ValueError('Incomplete source commands')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'commands.jsonl').open('w') as destination:
        for command in commands:
            destination.write(json.dumps(command, ensure_ascii=False) + '\n')
    with (args.output / 'per_query.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(next(iter(queries.values()))))
        writer.writeheader()
        writer.writerows(queries.values())
    stats = {'checks': dict(checks), 'groups': summarize(list(queries.values()), commands), 'labels': LABELS, 'graph_imports': dict(imports),
             'provenance': [str(p.resolve()) for p in (Path(__file__), Path(base.__file__), args.input / 'commands.jsonl', args.input / 'per_query.csv')],
             'scope': 'Static code patterns; final scored attempts; multiple labels per completed shell call; no execution; intraprocedural dataflow only'}
    (args.output / 'statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Python graph behavior', '', 'Rates use all questions in each outcome group. Labels overlap.', '']
    for dataset, groups in stats['groups'].items():
        lines += [f'## {dataset}', '', '| Behavior | Calls | Exit-0 calls | Queries | Correct queries | Wrong queries |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for tag, title in LABELS.items():
            a, c, w = [groups[g]['labels'][tag] for g in ('all', 'correct', 'wrong')]
            lines.append(f"| {tag}: {title} | {a['calls']} | {a['exit0_calls']} | {a['queries']} | {c['queries']} ({c['rate']:.1%}) | {w['queries']} ({w['rate']:.1%}) |")
        lines.append('')
    (args.output / 'tables.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
