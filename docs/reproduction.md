# Reproducing experiments and analysis

[Overview](../README.md) · [Setup and data](setup.md) · [Reproduction](reproduction.md) · [Repository reference](reference.md)

Complete [setup and data preparation](setup.md) first. All commands below start from the repository root unless
noted. `--check` performs an offline preflight; `--smoke` and experiment runs make model requests and write results.

## A first offline check

```bash
python3 experiments/stark/dci_run/run.py pi_full_test --check --datasets prime
```

This validates prompts, the harness wiring and isolation for Prime. `MODEL_BASE_URL` must be configured even for
this offline check; no API key or model call is needed. To run a small live check with the same configuration,
also set `MODEL_API_KEY`:

```bash
python3 experiments/stark/dci_run/run.py pi_full_test --smoke --datasets prime --workers 1
```

This runs **two questions**: the first Prime test query and the first Prime human-generated query. The
`pi_full_test` preset includes the 7,108 synthetic test queries and 263 human-generated queries; the paper's main
results report only the synthetic test queries. Other presets and their analysis names are listed in the
[configuration map](reference.md#names-used-in-the-paper-and-in-the-code).

## Running the configurations

All runners resume: a question whose `result.json` is terminal is never run again. Attempts that fail for a
transient reason (timeout, rate limit, connection error, invalid output) are retried with exponential backoff
within a fixed per-question attempt budget (`dci_run/core.py`, `ablation_final/final.py`,
`contamination/runner.py`); the closed-book probe runner allows up to six attempts per question, and
`dci_run/run.py` ends with one further pass in which questions that failed for a technical reason (not a refusal)
get up to three more attempts. A provider refusal ends the question, and failed questions score 0. The Full DKGI, tool-variant and One program runners provide
an offline `--check` (no model call; it needs `MODEL_BASE_URL` set and the Codex CLI installed) and a `--smoke`
mode. The closed-book probe runner instead provides `--status` and `--execute`. The validation questions of the tool-variant and One program smoke runs (`ablation_final/selection_val.json`,
20 per dataset, written on first use; the first of each dataset is its smoke question) are drawn with a fixed seed
from the STaRK validation questions, leaving out those of the prompt-validation rounds when their selection files
(`prompt_val/selection.json`, `prompt_val_r1/selection.json`) are present. These files are not included, so the
draw differs from the one of our smoke runs.

```bash
cd experiments/stark
# Full DKGI (main table): Pi + GPT-6-Luna on the full test (and the human-generated set, not reported in the paper)
python3 dci_run/run.py pi_full_test --check
python3 dci_run/run.py pi_full_test --workers 8
# Full DKGI (Codex) + GPT-6-Luna: test subset, then the full test
python3 dci_run/run.py codex_h --workers 8
python3 dci_run/run.py codex_luna_full --workers 8
# GPT-6-Astra
python3 dci_run/run.py pi_astra --workers 8 && python3 dci_run/run.py pi_astra_full --workers 8
python3 dci_run/run.py codex_astra --workers 8
# Tool variants on the test subset (with the ARK virtualenv; see docs/setup.md)
python3 ablation_final/build_views.py
python3 ablation_final/run.py --check
python3 ablation_final/run.py --smoke
python3 ablation_final/run.py --split test-0.1 --arms A1,A2,A3,A5,A6 --workers 8
# One program
bash x1_one_program/src/build.sh
python3 x1_one_program/run.py --check && python3 x1_one_program/run.py --smoke
python3 x1_one_program/run.py --split test-0.1 --workers 8
# Closed-book probes (full test, no tools): build the name indexes, then run
python3 contamination/name_index.py
python3 contamination/runner.py --execute
```

Results go to `pi_full_test/runs/`, `trials/` (Full DKGI (Codex) with GPT-6-Luna on the test subset and the full
test, the tool variants, One program), `astra_runs/runs/` and `contamination/runs/`, one directory per question with
all attempts and the agent's recorded requests. The model never sees a path containing the word "ablation" (checked
by the runners).

The prompt-validation queue scripts are not included; the validation is described in the paper's appendix. Its two
rounds on STaRK validation questions used the harness of the test runs with the prompt variants in
`prompt_val/variants.py` (`p0`, `p1`, `p2`) and the final prompt, with one difference: in the prompt-selection round
Pi ran without head-and-tail truncation (`harness_fixes_v2/`), with the uncapped sandbox shell. The comparison
with the base prompt on 750 further validation queries used it.

## Analysis

The Hugging Face trajectory dataset is a normalized export; these scripts read raw runner result trees
(`result.json` and, for behavior analyses, attempt artifacts) rather than the Hugging Face export directly.

```bash
cd experiments/stark
python3 analysis_final/make_all.py              # final_numbers, errors, behavior, strata, coding_effect,
                                                # coevolution, x1_stats -> analysis_final/out/
python3 contamination/analyze.py               # closed-book probe summary (ID and name probes)
python3 contamination/probe_strata.py --refresh # probe strata (H@1 on questions the probe answered or missed)
python3 analysis_final/compute_numbers.py       # appendix numbers (validation rounds, probes, costs)
python3 analysis_final/compute_numbers.py --edges  # also the distinct and reversed edge-pair counts of the
                                                # export-size table caption (needs pandas)
python3 analysis_final/cases/case_stats.py      # statistics of the case studies
python3 analysis_final/cases/trace.py mag 9090  # one case query across configurations: outcome, rank, tool
                                                # calls and the model-visible trajectory (the paper's cases are
                                                # MAG 9090, Prime 9996 and Amazon 6321)
```

The scripts read the result trees where the runners write them (`experiments/stark/pi_full_test/runs/`, `trials/`,
`astra_runs/runs/`, `contamination/runs/`), the labels in `private/`, the probes' name indexes in
`contamination/index/` (written by `contamination/name_index.py`), and the validation rounds from
`prompt_val/results_raw.jsonl` (prompt-selection round, one row per final result) and `prompt_val_r1/runs/<group>/`
(comparison with the base prompt); `compute_numbers.py --selection-round DIR --confirmatory-round FILE` reads the
validation-round files from another place. To analyse results stored elsewhere, link these directories into
`experiments/stark/`; the scripts only read them and write to `analysis_final/out/`, `analysis_final/cases/out/`
and `contamination/`.

All reported numbers are point estimates. Failed runs score 0. The text-constraint strata of `strata.py` and
`coding_effect.py` read the tags in `data/query_tags/tags_effective.jsonl` (see [Text-constraint tags](reference.md#text-constraint-tags); `strata.py
--text-constraint-tags FILE` reads another file). `final_numbers.py` writes the main-table and
ablation-table numbers; `coevolution.py` the tool creation and tool use analysis; `x1_stats.py` the One program numbers.
`download_original_behavior.py` and `align_intent_distribution.py` apply the same command classification to the
public DCI traces; the paper's appendix on behavior measures reports this comparison.
