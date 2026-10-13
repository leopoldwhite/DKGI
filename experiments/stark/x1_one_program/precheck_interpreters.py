"""Pre-check: how often Full DKGI (Pi, test subset, 710 questions) runs interpreters other than Python.

Reads only the 710 item directories listed in ../trials/selection.json (written by dci_run/run.py codex_h) under
../pi_full_test/runs/pi_final/<ds>/test/<qid>/ (result.json -> final attempt -> artifacts/conversation.json).
Commands are parsed, never executed. Writes checks/precheck_interpreters.json.
"""

import json
from collections import Counter
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
STARK = HERE.parent
FULL = STARK / "pi_full_test/runs/pi_final"
SELECTION = STARK / "trials/selection.json"

INTERPRETERS = {  # general-purpose interpreters other than Python (awk, perl and node first)
    "awk": "awk", "gawk": "awk", "mawk": "awk", "nawk": "awk", "busybox": "awk",
    "perl": "perl", "node": "node", "nodejs": "node", "ruby": "ruby", "php": "php",
    "lua": "lua", "luajit": "lua", "tclsh": "tcl", "Rscript": "R", "R": "R", "julia": "julia",
    "deno": "deno", "bun": "bun", "java": "java", "jshell": "java", "guile": "guile",
}
OTHER_SHELLS = {"sh", "dash", "zsh", "ksh", "fish", "csh", "tcsh"}
PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")
X1_ALLOWED = set("rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash python3".split())
BUILTINS = set(("cd echo printf test [ [[ true false read export set unset local declare shopt source . eval exec "
                "command type hash alias let return exit break continue shift wait trap ulimit umask pushd popd "
                "dirs times getopts printf mapfile readarray compgen complete builtin enable help history jobs "
                "kill disown fg bg suspend caller").split())
KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
            "in", "function", "select", "time", "!", "{", "}", "]]", "coproc"}
WRAPPERS = {"env", "command", "exec", "nohup", "time", "nice", "ionice", "stdbuf", "builtin", "sudo", "setsid"}
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w]*)\1")


def strip_heredocs(text):
    """Remove here-document bodies (program text fed on stdin is data for the reader, not shell commands)."""
    out, lines, i = [], text.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for match in HEREDOC.finditer(line):
            while i < len(lines) and lines[i].strip() != match[2]:
                i += 1
            i += 1
    return "".join(out)


def segments(text):
    """Split shell text into simple-command strings at unquoted operators; also yield bodies of $( ) and backticks."""
    parts, cur, quote, i = [], [], None, 0
    while i < len(text):
        ch = text[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch == "\\" and i + 1 < len(text):
            cur.extend(text[i:i + 2])
            i += 1
        elif ch == "#" and (not cur or cur[-1] in " \t\n;&|("):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif ch in ";&|()\n`" or (ch == "$" and text[i + 1:i + 2] == "("):
            parts.append("".join(cur))
            cur = []
            if ch == "$":
                i += 1
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def words_of(segment):
    try:
        return shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return segment.split()


def programs(command, depth=0):
    """Program names this command line would launch (lexical approximation)."""
    found = []
    for seg in segments(strip_heredocs(command)):
        words = words_of(seg)
        while words:
            # drop redirections and assignments in command position
            while words and (re.match(r"^[A-Za-z_]\w*=", words[0]) or re.match(r"^\d*[<>]", words[0])):
                words.pop(0)
            if not words:
                break
            head = words[0]
            name = Path(head).name
            if name in ("for", "select", "case"):
                break  # loop / case header: no command word in this segment
            if name in KEYWORDS:
                words.pop(0)
                continue
            if name in WRAPPERS:
                words.pop(0)
                while words and words[0].startswith("-"):
                    words.pop(0)
                continue
            if name == "timeout":
                words.pop(0)
                while words and words[0].startswith("-"):
                    words.pop(0)
                if words:
                    words.pop(0)  # duration
                continue
            found.append(name)
            if name == "xargs":
                rest = words[1:]
                while rest and rest[0].startswith("-"):
                    opt = rest.pop(0)
                    if opt in ("-I", "-n", "-P", "-d", "-L", "-s", "-E", "-a") and rest:
                        rest.pop(0)
                words = rest
                continue
            if name == "find":
                for k, w in enumerate(words):
                    if w in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(words):
                        found.append(Path(words[k + 1]).name)
            if name in ("bash", "sh", "dash", "zsh") and depth < 3:
                for k, w in enumerate(words[1:], 1):
                    if w.startswith("-") and "c" in w and k + 1 < len(words):
                        found.extend(programs(words[k + 1], depth + 1))
                        break
            break
    return found


def bash_commands(attempt):
    conv = json.loads((Path(attempt) / "artifacts/conversation.json").read_text())
    out = []
    for message in conv.get("messages", []):
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list):
            continue
        for item in message["content"]:
            if item.get("type") == "toolCall" and item.get("name") == "bash":
                command = (item.get("arguments") or {}).get("command")
                if isinstance(command, str):
                    out.append(command)
    return out


def self_check():
    cases = {
        "awk -F'\\t' '{print $1}' edges/00.tsv | head": {"awk", "head"},
        "rg x nodes | perl -ne 'print'": {"rg", "perl"},
        "python3 - <<'PY'\nprint('awk perl node')\nPY": {"python3"},
        "find . -name '*.tsv' -exec awk 1 {} \\;": {"find", "awk"},
        "bash -lc \"node -e 1 && echo ok\"": {"bash", "node", "echo"},
        "echo x | xargs -I{} awk 1 {}": {"echo", "xargs", "awk"},
        "for f in a b; do cut -f1 $f; done": {"cut"},
        "LC_ALL=C sort -u x > y": {"sort"},
        "echo $(awk 1 x)": {"echo", "awk"},
        "rg 'awk|perl' nodes": {"rg"},
    }
    for command, expected in cases.items():
        got = set(programs(command))
        assert got == expected, (command, got, expected)


def main():
    self_check()
    selection = json.loads(SELECTION.read_text())
    assert len(selection) == 710, len(selection)
    per_program, per_family, queries_with = Counter(), Counter(), Counter()
    total_cmds = python_cmds = interp_cmds = shell_cmds = outside_cmds = 0
    queries_interp = 0
    examples = {}
    outside_programs = Counter()
    missing = []
    for row in selection:
        item = FULL / row["dataset"] / "test" / str(row["query_id"])
        try:
            result = json.loads((item / "result.json").read_text())
            commands = bash_commands(result["attempt_path"])
        except (OSError, KeyError, ValueError) as error:
            missing.append({"dataset": row["dataset"], "query_id": row["query_id"], "error": repr(error)[:200]})
            continue
        families_here = set()
        for command in commands:
            total_cmds += 1
            progs = programs(command)
            fams = {INTERPRETERS[p] for p in progs if p in INTERPRETERS}
            if any(PYTHON.match(p) for p in progs):
                python_cmds += 1
            if fams:
                interp_cmds += 1
                for fam in fams:
                    per_family[fam] += 1
                    examples.setdefault(fam, []).append(command[:300]) if len(examples.get(fam, [])) < 5 else None
                families_here |= fams
            if any(p in OTHER_SHELLS for p in progs):
                shell_cmds += 1
            outside = {p for p in progs if p not in X1_ALLOWED and p not in BUILTINS and not PYTHON.match(p)}
            if outside:
                outside_cmds += 1
                outside_programs.update(outside)
            per_program.update(set(progs))
        if families_here:
            queries_interp += 1
            queries_with.update(families_here)
    report = {
        "source": "pi_full_test/runs/pi_final/<ds>/test/<qid> for the 710 questions of ablation_final/selection.json",
        "method": "bash tool calls of the final attempt (artifacts/conversation.json), parsed lexically (heredoc bodies "
                  "removed, command words after ; & | ( ) $( ` newline, xargs, find -exec, timeout/env/nohup wrappers, bash -c bodies)",
        "questions_read": 710 - len(missing), "missing": missing,
        "bash_commands": total_cmds,
        "commands_running_python": python_cmds,
        "commands_running_other_interpreters": interp_cmds,
        "share_other_interpreters": interp_cmds / total_cmds if total_cmds else None,
        "by_interpreter_commands": dict(per_family.most_common()),
        "questions_with_other_interpreters": queries_interp,
        "share_questions_with_other_interpreters": queries_interp / (710 - len(missing)),
        "by_interpreter_questions": dict(queries_with.most_common()),
        "commands_running_sh_dash_zsh": shell_cmds,
        "commands_using_any_program_outside_x1_allowlist": outside_cmds,
        "share_outside_x1_allowlist": outside_cmds / total_cmds if total_cmds else None,
        "programs_outside_x1_allowlist_top": dict(outside_programs.most_common(25)),
        "examples": examples,
    }
    (HERE / "checks").mkdir(exist_ok=True)
    (HERE / "checks/precheck_interpreters.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("examples", "missing")}, indent=2))


if __name__ == "__main__":
    main()
