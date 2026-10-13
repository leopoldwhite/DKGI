# Repository reference

[Overview](../README.md) · [Setup and data](setup.md) · [Reproduction](reproduction.md) · [Repository reference](reference.md)

Paths in this guide are relative to the repository root. Directory names, presets and prompt identifiers are kept
as used by the released code so that existing commands and result trees remain compatible.

## Layout

```
README.md                        paper overview and quickstart
docs/                            setup, reproduction and repository reference
assets/                          method figure used in the README
LICENSE, LICENSE-DATA.md         MIT code; CC BY 4.0 derived data
prompts/                         exact prompt texts (see prompts/README.md)
data/query_tags/tags_effective.jsonl
                                 per-query text-constraint tags read by the analysis (see Text-constraint tags below)
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
