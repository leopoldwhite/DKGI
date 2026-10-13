# Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve

Yuyang Bai, Yu Wang, Yu Zhang (Texas A&M University, University of Oregon)

[Project page](https://yuyangbai.com/dkgi/) | [Graphs and trajectories on Hugging Face](https://huggingface.co/collections/yuyangbai/dkgi)

DKGI (direct knowledge graph interaction) is a harness in which LLM agents create graph access tools directly over
the complete node and edge files of a knowledge graph. The agent writes Python code to follow relations, combine
node sets and select textual evidence, interleaving execution with shell search and file inspection. On the three
text-rich knowledge graphs of STaRK, training-free DKGI exceeds the strongest published baseline by at least 6.7
Hit@1 points on every graph.

This repository contains the code that exports the STaRK knowledge graphs to files, runs every configuration
reported in the paper, and computes the numbers of the paper's tables and most numbers in its text from the run
results, plus the exact prompt texts. The exported graphs and the trajectories are on Hugging Face (see Data on
Hugging Face). The repository itself holds one data file, the per-query text-constraint tags used by the analysis
(`data/query_tags/`, see Text-constraint tags).

## Names used in the paper and in the code

The code uses short internal identifiers for directories, presets and groups. Paper name -> code:

| Paper | Code |
|---|---|
| Full DKGI (Pi), GPT-6-Luna | `dci_run/run.py` preset `pi_full_test` (group `pi_final`); analysis names `pi_final_full` (full test; `luna_full` in `coevolution.py`) and `pi_final_human` (human-generated set, split `human_generated_eval`, not reported in the paper) |
| Full DKGI (Codex), GPT-6-Luna | presets `codex_h` (test subset) and `codex_luna_full` (full test; reuses the test-subset results of `codex_h`), group `codex_final`; analysis names `H` (test subset; the Full DKGI (Codex) row of the ablation table) and `codex_luna_full` (full test; `codex_full` in `coevolution.py`) |
| Full DKGI (Pi), GPT-6-Astra | presets `pi_astra` (test subset) and `pi_astra_full` (full test; reuses the test-subset results of `pi_astra`); analysis names `astra_pi_full` (full test; `astra_full` in `coevolution.py`) and `astra_pi_710` (test subset; `Astra-Pi` in `final_numbers.py`) |
| Full DKGI (Codex), GPT-6-Astra | preset `codex_astra` (test subset); analysis name `astra_codex_710` (`Astra-Codex` in `final_numbers.py`) |
| Full DKGI on the test subset (ablation table) | `A4` in the analysis: the test-subset part of the `pi_full_test` run |
| Retrieval only | arm `A1` |
| Graph tools | arm `A2` |
| Shell only | arm `A3` |
| Graph tools + code | arm `A5` |
| No edges | arm `A6` |
| One program | `x1_one_program/`, arm `O1` (analysis name `X1`) |
| tool section of the final prompt (Full DKGI, Pi and Codex) | `ablation/prompts/H.txt` (prompt file, `H` in the tool-variant code; in the analysis scripts `H` is the Full DKGI (Codex) run on the test subset, see above) |
| closed-book ID probe / name probe | `A0-ID` / `A0-Name` (`contamination/`) |
| base prompt / earlier ranking instruction / rg prompt | `p0` / `p1` / `p2` (`prompt_val/variants.py`; `P1`, `P2` in `coding_effect.py`) |
| final prompt | `P1-R1` (`dci_run/core.py` `P1R1_OUTPUT`) |
| test subset (710 questions) | split `test-0.1` (the official STaRK test-0.1 split) |
| head-and-tail truncation | `H6` (`harness_fixes_v2/cap_filter.py`) |
| Pi context management level | `L3` (Pi option `--context-management-level level3`) |
| prompt-selection round / comparison with the base prompt on 750 further validation queries | analysis names `t11_*` / `r1_*` |
| small wrapper around DCI-Agent-Lite's runner that keeps a run going while Pi retries a failed model request | `prompt_val/transport/pi_rpc_retry.py`, installed by `prompt_val/pi_transport.py`; the same module preloads `prompt_val/transport/blocking_stdout.cjs` into Pi's Node process (`NODE_OPTIONS=--require`) only to make Pi's RPC output blocking, so that no event is lost while the recorder reads; no Pi code is changed |
| the agent of this paper (Full DKGI and its variants) | the prefix `dci` / `DCI` in code, variable and directory names (`dci_run/`, `DCI_ATTEMPT`, `DCI_CORPUS`, `--dci`, `dci_*` result fields), chosen because the harness wraps the public DCI-Agent-Lite runner; the DCI method of Li et al. (2026) is meant only where the text names it: DCI-Agent-Lite, DCI's tool instruction (the one sentence of the rg prompt), the original DCI wording and the public DCI traces |

## Layout

```
README.md, LICENSE (MIT, code), LICENSE-DATA.md (CC BY 4.0, derived data)
prompts/                         exact prompt texts (see prompts/README.md)
data/query_tags/tags_effective.jsonl
                                 per-query text-constraint tags read by the analysis (see Text-constraint tags)
experiments/stark/
  verify_data.py, prepare.py     check the STaRK download; export the graphs to files
  task.py                        base prompt, scoring, retry classification
  dci_run/                       Full DKGI runner (Pi and Codex presets)
  harness_fixes/, harness_fixes_v2/, prompt_val/
                                 harness: Landlock launcher and sandbox shell, Pi transport, observability,
                                 head-and-tail truncation, prompt variants, STaRK adapter
  ablation/, ablation_final/     tool variants (graph-tool service, tool sections, sandboxes) and their runner
  x1_one_program/                One program (sources of its two small C helpers in src/)
  contamination/                 closed-book ID and name probes, their analysis and strata
  analysis_final/                analysis scripts that produce the paper's tables and numbers
  *.py (analyze_behavior, align_intent_distribution, plot_intent_distribution, analyze_graph_behavior)
                                 shell/Python command classifiers shared by runners and analysis;
                                 download_original_behavior.py downloads the public DCI-Agent-Lite traces
experiments/agent_base/          agent process runner: isolation, command lines, collection; protocol.json
experiments/codex_probe/landlock_exec.py   Landlock launcher used for every agent process
third_party/                     DCI-Agent-Lite patch; ARK overlay for the graph-tool service
```

## Requirements

- Linux x86_64 with Landlock ABI 3 or later (kernel 6.2 or later) and `/usr/bin/python3` (3.10 or later; the
  sandbox launcher runs it with `-S`). The sandbox grants read access to a fixed list of system files of a
  Debian-style `/etc` (for example `/etc/bash.bashrc`, `/etc/profile.d`, `/etc/apt/apt.conf.d`); each listed path
  must exist. The runners need Python 3.10+ with `numpy`; the export needs `torch`; figures need `matplotlib`;
  `compute_numbers.py --edges` needs `pandas`.
- A short install path. The tool variants and One program create Unix sockets inside the result tree
  (`<attempt dir>/kb.sock`, `<question dir>/attempt_NNN.gate.sock`), and a Unix socket path has at most 107 bytes,
  so the absolute path of the package root must have at most 38 characters (for example `~/dkgi` or `/opt/dkgi`);
  move or rename the extracted directory otherwise.
- Codex CLI 0.156.1 (standalone release). Its `codex` binary also provides the seccomp sandbox used by the Pi
  shell. Put it on `PATH`, install it at the default standalone location, or set `CODEX_BIN`. Codex runs need
  `~/.codex/installation_id` (written by Codex on its first run), which is the one file outside the attempt
  directory that Codex may write.
- Node.js and the public DCI-Agent-Lite release (https://github.com/DCI-Agent/DCI-Agent-Lite, commit
  271f37e71f053bf0c99c05ce6d2fb53b841d922e), cloned into `DCI-Agent-Lite/` in the package root and set up with its
  `setup.sh`: a Python virtualenv in `DCI-Agent-Lite/.venv`, and Pi 0.65.0 in `DCI-Agent-Lite/pi-mono`, cloned and
  built from the Pi branch that the release's README and `setup.sh` name (`codex/context-management-ablation`; it
  adds the context-management levels and the provider-request events that the harness records). Our runs used commit
  a6be5eb4cce278de31ac05792af3dfc0883215dc of that branch; check it out before building if the branch has moved.
  Apply `third_party/DCI-Agent-Lite.diff` to that checkout (`git apply`). Pi itself is not modified. The diff makes
  three changes to DCI-Agent-Lite's runner script (`src/dci/benchmark/pi_rpc_runner.py`); the paper's appendix
  describes the first, which affects scoring: the final answer is read from the agent's last assistant message, and a run that ends on a tool
  call or an error fails; with `--compact-tool-updates`, progress updates of a running tool are logged by size only;
  and the endpoint of its answer judge can be set with `OPENAI_BASE_URL` (the judge is not used). Its only other
  hunk changes how `setup.sh` loads an optional `.env` file; this affects setup only.
- For the graph-tool variants (Retrieval only, Graph tools, Graph tools + code): ARK
  (https://github.com/mims-harvard/ark, commit 8d3191b9de1ebf05a69ff88821edd08fc050760b) cloned into
  `baselines/ark/` in the package root, the files of `third_party/ark_overlay/` copied into that checkout, a
  virtualenv at `baselines/ark/.venv` with ARK's requirements and `requirements-tools.txt`, and the graphs prepared
  with `python prepare_local.py` there after the export below (ablation_final/run.py re-executes itself with that
  virtualenv; the graph-tool service builds each graph's BM25 index on first use).
- For One program: `gcc` (run `bash experiments/stark/x1_one_program/src/build.sh` once; it builds the `python3`
  shim and the interpreter lock into `x1_one_program/sandbox/`).

## Environment variables

| Variable | Meaning |
|---|---|
| `MODEL_BASE_URL` | OpenAI-compatible endpoint (Responses API) serving `gpt-6-luna` and `gpt-6-astra`; required, no default |
| `MODEL_API_KEY` | API key for that endpoint; passed to the agents by name only |
| `CODEX_BIN` | optional path of the Codex CLI 0.156.1 binary |

`experiments/agent_base/protocol.json` holds the fixed model settings (reasoning effort high, 272,000-token
context, 1,800 s per attempt, at most 300 Pi turns). Pi requests at most 32,000 output tokens per model request (Pi
0.65.0's default when no limit is set: the smaller of the model's `maxTokens` and 32,000; the 128,000 in the model
entries of `dci_run/core.py` and `prompt_val/harness.py` is therefore never reached). The endpoint is read from
`MODEL_BASE_URL` at run time.

## Data on Hugging Face

| Dataset | Content |
|---|---|
| [`yuyangbai/DKGI-STaRK-Graphs`](https://huggingface.co/datasets/yuyangbai/DKGI-STaRK-Graphs) | the exported graphs, questions and labels, as written by `prepare.py` (8.3 GiB) |
| [`yuyangbai/DKGI-Trajectories`](https://huggingface.co/datasets/yuyangbai/DKGI-Trajectories) | for every configuration in the paper's tables and every question: the agent's trajectory, its answer and its scores |

To use the exported graphs instead of exporting them yourself (next section), download them into
`experiments/stark/` and point the question files at this checkout:

```bash
hf download yuyangbai/DKGI-STaRK-Graphs --repo-type dataset --local-dir experiments/stark \
  --include "corpus/*" "public/*" "private/*" "prepared/*"
python3 experiments/stark/use_downloaded_graphs.py
```

## Data and export

```bash
# STaRK data release (CC BY 4.0), about 19 GiB with its archives
hf download snap-stanford/stark --repo-type dataset \
  --revision 88269e23e90587f99476c5dd74e235a0877e69be --local-dir STaRK/data
# extract the archives in place (idx_title_abs.zip goes to processed/paper100M_cache)
(cd STaRK/data/skb && for z in */*.zip; do d=$(dirname "$z"); \
  [ "$(basename "$z")" = idx_title_abs.zip ] && d="$d/processed/paper100M_cache"; unzip -n -q "$z" -d "$d"; done)
python3 experiments/stark/verify_data.py > STaRK/verification.json
python3 experiments/stark/prepare.py            # writes experiments/stark/{corpus,public,private,prepared}/
```

`prepare.py` writes, per dataset, `corpus/<dataset>/` (nodes/NNNN.jsonl, edges/RR.tsv, schema.json,
DATA_FORMAT.txt, candidate_ids.txt; this is what the agent reads), the questions without labels in `public/`, and
the labels in `private/` (never readable by the agent: every attempt runs under Landlock and a preflight checks
that `private/`, `public/` and the STaRK source data cannot be opened). The derived views of the variants
(Shell only, No edges, and the working directory of the graph-tool variants, which holds only a README) are
hardlinks of `corpus/` with edited `DATA_FORMAT.txt` / `schema.json`, built once with
`python3 experiments/stark/ablation_final/build_views.py`. The rg prompt of the prompt-selection round
reads `corpus2/<dataset>/`, a hardlink copy of `corpus/<dataset>/` whose `DATA_FORMAT.txt` lacks its last line (as
`prompts/DATA_FORMAT_shell_only.txt`), built with `python3 experiments/stark/prompt_val/views.py`.

## Running the configurations

All runners resume: a question whose `result.json` is terminal is never run again. Attempts that fail for a
transient reason (timeout, rate limit, connection error, invalid output) are retried with exponential backoff
within a fixed per-question attempt budget (`dci_run/core.py`, `ablation_final/final.py`,
`contamination/runner.py`); the closed-book probe runner allows up to six attempts per question, and
`dci_run/run.py` ends with one further pass in which questions that failed for a technical reason (not a refusal)
get up to three more attempts. A provider refusal ends the question, and failed questions score 0. Each runner has
an offline `--check` (no model call; it needs `MODEL_BASE_URL` set and the Codex CLI installed) and a `--smoke`
mode. The validation questions of the tool-variant and One program smoke runs (`ablation_final/selection_val.json`,
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
# Tool variants on the test subset (with the ARK virtualenv; see Requirements)
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
`coding_effect.py` read the tags in `data/query_tags/tags_effective.jsonl` (see below; `strata.py
--text-constraint-tags FILE` reads another file). `final_numbers.py` writes the main-table and
ablation-table numbers; `coevolution.py` the tool creation and tool use analysis; `x1_stats.py` the One program numbers.
`download_original_behavior.py` and `align_intent_distribution.py` apply the same command classification to the
public DCI traces; the paper's appendix on behavior measures reports this comparison.

## Text-constraint tags

`data/query_tags/tags_effective.jsonl` has one JSON line per query of the full test set (7,108 queries), the
human-generated set (263, not reported in the paper) and the STaRK validation questions selected for prompt validation (1,350), with the
fields `dataset`, `split`, `query_id`, `has_text_constraint` (true: the query adds a text constraint to its
relational constraints; null: no tag), `stratify_by_label` and `n_relational_spans`. It contains no STaRK answers.

How the tags were produced: a model annotator (GPT-6-Luna, reasoning effort high, no tools) saw only the query and
the dataset's node-type and relation-type names. Its instructions are `prompts/text_constraint_annotation_prompt.txt`
and its input is `prompts/text_constraint_annotation_input_<dataset>.txt` with the query in place of `<QUERY>`. It
returns a JSON object with `has_text_constraint`, `relational_spans` and `text_spans`. An output was accepted only
with exactly these keys, a boolean label, spans that are non-empty verbatim substrings of the query, and text spans
present exactly when the label is true; an output that failed these checks was requested once more. Two Prime test
queries received no valid output; they have no tag and are excluded (`strata.py` reports the count). The file keeps
the label and the number of relational spans, not the span texts. The script that sent the annotation requests is
not included.

How the tags are applied: Amazon and Prime use the annotator's label (`stratify_by_label` true). Every MAG query
counts as text-constrained, since its topical descriptions require reading the paper text; the annotator's MAG labels
are not used and MAG is not stratified (`has_text_constraint` true, `stratify_by_label` false, `n_relational_spans`
null). `strata.py` reports H@1 by text constraint on Amazon and Prime; `coding_effect.py` uses the same labels for the
code effect by text constraint, and `n_relational_spans` (Amazon, Prime) as a descriptive proxy for relational
complexity.

## Citation

```bibtex
@misc{bai2026dkgi,
  title         = {Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve},
  author        = {Bai, Yuyang and Wang, Yu and Zhang, Yu},
  year          = {2026},
}
```

## Licenses

Code: MIT (`LICENSE`). Data derived from STaRK with this code, the text-constraint tags in `data/` and the prompt
texts: CC BY 4.0 (`LICENSE-DATA.md`), following the STaRK data license. The exported graphs and the trajectories on Hugging Face are released under CC BY 4.0 as well. Third-party code (DCI-Agent-Lite, Pi,
Codex CLI, ARK, STaRK) keeps its own license and is not included. The patch and overlay files in `third_party/` are
part of this package's code and are released under MIT; they modify or extend DCI-Agent-Lite and ARK, both
MIT-licensed.
