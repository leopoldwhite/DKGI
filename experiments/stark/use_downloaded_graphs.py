"""Point the question files of the Hugging Face download (yuyangbai/DKGI-STaRK-Graphs) at this checkout.

The download stores corpus_dir in public/<dataset>/*.jsonl as the relative path corpus/<dataset>. The runners and
the prompt use the absolute path, as prepare.py writes it. This script rewrites corpus_dir in place to the absolute
path of experiments/stark/corpus/<dataset>. Running it again changes nothing.
"""

import argparse
import json
from pathlib import Path

STARK = Path(__file__).resolve().parent


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    changed = 0
    for path in sorted((STARK / "public").glob("*/*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            target = str(STARK / "corpus" / row["dataset"])
            if row["corpus_dir"] != target:
                row["corpus_dir"] = target
                changed += 1
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    missing = [name for name in ("amazon", "mag", "prime") if not (STARK / "corpus" / name / "schema.json").exists()]
    print(f"{changed} question rows updated" + (f"; no corpus for {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
