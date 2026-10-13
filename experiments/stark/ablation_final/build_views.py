"""Build the corpus views of the tool variants from the export (run once after prepare.py; no model call).

    python3 experiments/stark/ablation_final/build_views.py

  kb/<dataset>/                 nodes and candidate list as hardlinks of corpus/<dataset>/; edge-free schema.json
                                and DATA_FORMAT.txt (ablation/common.py assets())
  corpus_views/v1/<dataset>/    Shell only (A3): the full corpus; DATA_FORMAT.txt without the Python line
  corpus_views/v2/<dataset>/    No edges (A6): nodes, candidate list, edge-free schema.json and DATA_FORMAT.txt
  corpus_views/workspace/       working directory of the graph-tool variants (README.txt only)
"""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final as f  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    f.c.assets()
    for arm in ("A3", "A6"):
        for dataset in f.DATASETS:
            print(f.build_view(arm, dataset), flush=True)
    print("workspace:", f.build_workspace(), flush=True)


if __name__ == "__main__":
    main()
