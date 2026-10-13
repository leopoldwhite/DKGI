# Setup and data

[Overview](../README.md) · [Setup and data](setup.md) · [Reproduction](reproduction.md) · [Repository reference](reference.md)

Run commands from the repository root unless a command changes directory. Agent execution requires Linux;
this repository does not include third-party harness checkouts or model access.

## Clone into a short path

```bash
git clone https://github.com/leopoldwhite/DKGI.git ~/dkgi
cd ~/dkgi
```

## Requirements

- Linux x86_64 with Landlock ABI 3 or later (kernel 6.2 or later) and `/usr/bin/python3` (3.10 or later; the
  sandbox launcher runs it with `-S`). The sandbox grants read access to a fixed list of system files of a
  Debian-style `/etc` (for example `/etc/bash.bashrc`, `/etc/profile.d`, `/etc/apt/apt.conf.d`); each listed path
  must exist. The runners need Python 3.10+ with `numpy`; the export needs `torch`; figures need `matplotlib`;
  `compute_numbers.py --edges` needs `pandas`.
- The Hugging Face CLI (`hf`) for the download commands below.
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
  Apply `third_party/DCI-Agent-Lite.diff` in the `DCI-Agent-Lite/` checkout
  (`git -C DCI-Agent-Lite apply ../third_party/DCI-Agent-Lite.diff` from the repository root). Pi itself is not modified. The diff makes
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

Set `MODEL_BASE_URL` before offline checks as well as live runs. `MODEL_API_KEY` is needed for live model requests.

`experiments/agent_base/protocol.json` holds the fixed model settings (reasoning effort high, 272,000-token
context, 1,800 s per attempt, at most 300 Pi turns). Pi requests at most 32,000 output tokens per model request (Pi
0.65.0's default when no limit is set: the smaller of the model's `maxTokens` and 32,000; the 128,000 in the model
entries of `dci_run/core.py` and `prompt_val/harness.py` is therefore never reached). The endpoint is read from
`MODEL_BASE_URL` at run time.

## Download the exported graphs

| Dataset | Content |
|---|---|
| [`yuyangbai/DKGI-STaRK-Graphs`](https://huggingface.co/datasets/yuyangbai/DKGI-STaRK-Graphs) | the exported graphs, questions and labels, as written by `prepare.py` (8.3 GiB) |
| [`yuyangbai/DKGI-Trajectories`](https://huggingface.co/datasets/yuyangbai/DKGI-Trajectories) | recorded test configurations: the agent's trajectory, its answer and its scores |

The trajectory release covers the test configurations; the two prompt-validation rounds are not included.

To use the exported graphs instead of [exporting them yourself](#export-from-the-original-stark-data), download them into
`experiments/stark/` and point the question files at this checkout:

```bash
hf download yuyangbai/DKGI-STaRK-Graphs --repo-type dataset --local-dir experiments/stark \
  --include "corpus/*" "public/*" "private/*" "prepared/*"
python3 experiments/stark/use_downloaded_graphs.py
```

## Export from the original STaRK data

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
