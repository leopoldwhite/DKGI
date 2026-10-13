# Prompts

Exact prompt texts as the runners build them. Each file ends with the newline the runner writes to the attempt's
`prompt.txt`; DCI-Agent-Lite's runner strips white space at both ends before Pi sends the file as the user message,
so the paper's prompt boxes end without that newline. Placeholders (paper name in parentheses): `<QUERY>` the STaRK
question (`<query>`), `<CORPUS_DIR>` the agent's working directory (`<corpus dir>`), `<ATTEMPT>` the attempt
directory (`<attempt dir>`), `<ATTEMPT>/tmp` the writable scratch directory (`<scratch dir>`).

The code that builds them: `experiments/stark/task.py` (base prompt),
`experiments/stark/prompt_val/variants.py` (ranking instructions and tool instructions), `experiments/stark/dci_run/core.py`
(final prompt), `experiments/stark/ablation/prompts/*.txt` with `experiments/stark/ablation_final/final.py`
(tool sections of the variants), `experiments/stark/x1_one_program/x1.py` (One program sentence) and
`experiments/stark/contamination/prompts.py` (closed-book probes). The text-constraint annotation files are the
exact texts of that annotation step, whose code is not included.

| File | Paper name | Used by |
|---|---|---|
| `final_prompt.txt` | final prompt (Full DKGI) | Full DKGI with Pi and with Codex, GPT-6-Luna and GPT-6-Astra; comparison with the base prompt on 750 further validation queries |
| `base_prompt.txt` | base prompt | validation rounds (the prompt used before validation) |
| `initial_output_policy_prompt.txt` | earlier ranking instruction with our tool instruction | prompt-selection round |
| `rg_only_prompt.txt` | rg prompt | prompt-selection round |
| `graph_tools_prompt.txt` | Graph tools | tool variant A2 |
| `graph_tools_code_prompt.txt` | Graph tools + code | tool variant A5 |
| `shell_only_prompt.txt` | Shell only | tool variant A3 |
| `retrieval_only_prompt.txt` | Retrieval only | tool variant A1 |
| `no_edges_prompt.txt` | No edges | tool variant A6 |
| `one_program_prompt.txt` | One program | final prompt plus one sentence (X1) |
| `id_probe_prompt_*.txt` | closed-book ID probe | A0-ID (Amazon, MAG, Prime; the knowledge-base line differs per dataset) |
| `name_probe_prompt_*.txt` | closed-book name probe | A0-Name (MAG, Prime) |
| `DATA_FORMAT.txt` | data-format file of the export | Full DKGI, One program |
| `DATA_FORMAT_shell_only.txt` | data-format file without the Python line | Shell only; rg prompt (`corpus2/`) |
| `DATA_FORMAT_no_edges.txt` | data-format file without the edge lines | No edges |
| `text_constraint_annotation_prompt.txt` | text-constraint tags (instructions) | model annotator of the analysis tags |
| `text_constraint_annotation_input_*.txt` | text-constraint tags (user message) | one per dataset (Amazon, MAG, Prime) |

The two text-constraint annotation files produced the per-query tags in `data/query_tags/tags_effective.jsonl`
(README, Text-constraint tags). The instructions file was sent as the `instructions` (system message) of a
Responses API request and the input file as its user message, with `<QUERY>` replaced by the query text as a JSON
string value (escaped as JSON); the input files have no final newline, as sent.

The harness system prompts described in the paper's appendix are not copied here; both are used unchanged: Pi's
default system prompt (built by Pi 0.65.0 at run time; it states the enabled tools, a few guidelines, the location of
Pi's documentation, the date and the working directory) and the built-in instructions of Codex CLI 0.156.1.
