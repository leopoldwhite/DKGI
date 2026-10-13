"""Regenerate every analysis output in analysis_final/out/ with one command.

    python3 analysis_final/make_all.py            # uses the per-query feature cache when still valid
    python3 analysis_final/make_all.py --no-cache # re-extract every trajectory

Each step runs in its own process; a failing step is recorded in out/INDEX.md and the others still run.
No model or API calls; experiment directories are only read.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
STEPS = [
    ("final_numbers.py", "Final main-table and ablation numbers",
     ["final_numbers.md", "final_numbers.json", "tex/final_main_rows.tex", "tex/final_ablation_table.tex"]),
    ("errors.py", "Error taxonomy", ["errors.md", "errors_by_config.csv", "errors_per_query.csv",
                                                 "errors_seen_sensitivity.csv", "tex/errors_table.tex",
                                                 "tex/appendix/errors_seen_id_table.tex", "figures/errors_710.pdf",
                                                 "figures/appendix/errors_full.pdf", "figures/appendix/errors_dev.pdf"]),
    ("behavior.py", "Behavior metrics", ["behavior.md", "behavior_by_config.csv", "tex/behavior_table.tex",
                                                     "figures/behavior_intents_710.pdf",
                                                     "figures/appendix/behavior_intents_full.pdf",
                                                     "figures/appendix/behavior_intents_dev.pdf"]),
    ("strata.py", "Text-constraint and closed-book probe strata", ["strata.md", "strata_text_constraint.csv", "strata_a0.csv",
                                                                   "a0_probe_summary.csv", "tex/strata_text_table.tex",
                                                                   "tex/strata_a0_table.tex",
                                                                   "tex/appendix/strata_a0_h20_table.tex"]),
    ("coding_effect.py", "Code effect, cost of code, prompt-driven coding (2x2 ablation; prompt selection P1 vs P2)",
     ["coding_effect.md", "coding_effect.json"]),
    ("coevolution.py", "Tool coding and tool calling within one query (behaviour study; own cache in coevolution/cache)",
     ["coevolution/REPORT.md", "coevolution/coevolution.json", "coevolution/tex/coevolution_table.tex",
      "coevolution/tex/appendix/coevolution_table_codex.tex", "coevolution/tex/coevolution_figure.tex",
      "coevolution/figures/coevolution_figure.pdf",
      "coevolution/figures/coevolution_phases.pdf", "coevolution/figures/coevolution_feedback.pdf",
      "coevolution/lineage_pairs_sample.md"]),
    ("x1_stats.py", "X1 One program: starts, program length, cost, metrics (descriptive; after coevolution.py and "
     "coding_effect.py)", ["x1_one_program.md", "x1_one_program.json"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these step scripts, e.g. errors.py")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.only:
        # figures/ and tex/ are fully regenerated; clearing them removes outputs of earlier layouts.
        import shutil
        for sub in ("figures", "tex"):
            shutil.rmtree(OUT / sub, ignore_errors=True)
    status = []
    first = True
    for script, title, outputs in STEPS:
        if args.only and script not in args.only:
            continue
        cmd = [sys.executable, str(HERE / script)]
        # The first cache-using step fills the shared per-query cache; later steps reuse it.
        # final_numbers.py, coding_effect.py and x1_stats.py take no --no-cache flag (they never write the cache).
        if script == "coevolution.py":
            # coevolution.py keeps its own cache (out/coevolution/cache), so it always gets the flag.
            if args.no_cache:
                cmd.append("--no-cache")
        elif script not in ("final_numbers.py", "coding_effect.py", "x1_stats.py"):
            if args.no_cache and first:
                cmd.append("--no-cache")
            first = False
        t0 = time.time()
        print(f"== {script}", flush=True)
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        (OUT / "logs").mkdir(exist_ok=True)
        (OUT / "logs" / f"{script}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        ok = proc.returncode == 0
        print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "", flush=True)
        if not ok:
            print(proc.stderr[-2000:], file=sys.stderr, flush=True)
        status.append((script, title, ok, time.time() - t0, outputs))
    lines = ["# Analysis outputs", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `make_all.py`. "
             "Each report starts from its own coverage table; partial runs are analysed as they are.", "",
             "| Step | Content | Status | Seconds | Outputs |", "|---|---|---|---:|---|"]
    for script, title, ok, secs, outputs in status:
        present = [o for o in outputs if (OUT / o).exists()]
        lines.append(f"| `{script}` | {title} | {'ok' if ok else 'FAILED (see logs/)'} | {secs:.0f} | "
                     + ", ".join(f"`{o}`" for o in present) + " |")
    lines += ["", "Main-text material: `figures/*.pdf`, `tex/*.tex`. Appendix material: `figures/appendix/`, `tex/appendix/`.",
              "Logs: `logs/<step>.log`. Per-query feature cache: `cache/` (safe to delete)."]
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")
    failed = [s for s, _, ok, _, _ in status if not ok]
    print("done" + (f"; failed steps: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
