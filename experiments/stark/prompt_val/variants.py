"""Prompt variants: base prompt (p0), initial output policy (p1), rg-only variant (p2); built on the STaRK task prompt."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STARK = ROOT.parent
sys.path.insert(0, str(STARK))
_spec = importlib.util.spec_from_file_location("stark_task", STARK / "task.py")
stark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stark)
if Path(stark.prepare.__file__).resolve() != STARK / "prepare.py":
    raise RuntimeError("Loaded the wrong STaRK prepare module")

VARIANTS = ("p0", "p1", "p2")
OLD_OUTPUT = (
    "Return at most 100 unique integer IDs observed in the corpus. Do not invent or "
    "pad IDs; return an empty list if none are supported."
)
NEW_OUTPUT = (
    "The query may have several correct answers. Return 20 unique integer IDs observed "
    "in the corpus, ranked by how well the evidence supports them: first the candidates "
    "you are most confident in, then the next most plausible candidates you found. "
    "Return fewer than 20 only if you cannot find that many plausible candidates. "
    "Do not invent IDs."
)
OLD_TOOLS = "Use shell tools, rg, or Python 3 standard-library modules to search and read them."
NEW_TOOLS = "Use ripgrep (rg) instead of grep for fast searching."


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new, 1)


def row_for(row, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    result = dict(row)
    dataset = row["dataset"]
    if dataset not in stark.prepare.DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    original = STARK / "corpus" / dataset
    alternate = STARK / "corpus2" / dataset
    allowed = (str(original), str(alternate)) if variant == "p2" else (str(original),)
    if row["corpus_dir"] not in allowed:
        raise ValueError("Unexpected corpus directory")
    result["corpus_dir"] = str(alternate if variant == "p2" else original)
    return result


def prompt_for(row, attempt, variant="p0"):
    prompt = stark.prompt_for(row_for(row, variant), Path(attempt))
    if variant in ("p1", "p2"):
        prompt = replace_once(prompt, OLD_OUTPUT, NEW_OUTPUT)
    if variant == "p2":
        prompt = replace_once(prompt, OLD_TOOLS, NEW_TOOLS)
    return prompt
