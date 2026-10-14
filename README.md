<div align="center">

<h1>DKGI</h1>
<h3>Knowledge Graphs as Files:<br>An Agent Harness Where Tool Creation and Tool Use Co-Evolve</h3>

<p>
  <a href="https://yuyangbai.com/">Yuyang Bai</a><sup>1</sup>,
  <a href="https://yuwang0103.github.io/">Yu Wang</a><sup>2</sup>,
  <a href="https://yuzhimanhua.github.io/">Yu Zhang</a><sup>1</sup>
</p>
<p><sup>1</sup>Texas A&amp;M University &nbsp; <sup>2</sup>University of Oregon</p>

<p>
  <a href="https://arxiv.org/abs/2610.18616">Paper</a> &nbsp;·&nbsp;
  <a href="https://yuyangbai.com/dkgi/">Project</a> &nbsp;·&nbsp;
  <a href="https://github.com/leopoldwhite/DKGI">Code</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/collections/yuyangbai/dkgi">Data</a>
</p>

</div>

**Direct knowledge graph interaction (DKGI)** lets language model agents write and revise graph-access tools
directly over complete node and edge files. The agent interleaves Python, shell search and file inspection:
execution feedback guides code revisions, and the evolving code guides the next search.

![A predefined neighbors tool compared with DKGI, where the agent writes and revises graph-access code during the task.](assets/fig1.png)

*Tool creation and tool use co-evolve during graph search. In this STaRK-MAG example, DKGI extends its search over
the graph files and ranks the gold paper first.*

The repository contains graph export code, runners for Full DKGI and its controlled variants, closed-book probes,
exact prompts, and scripts for the paper's analysis. The exported graphs and recorded test trajectories are hosted
on Hugging Face.

## Results

Hit@1 (%) on the three STaRK synthetic test sets, covering 7,108 queries:

| Method | Amazon | MAG | Prime |
|:--|--:|--:|--:|
| GRASP (published baseline) | 71.2 | 82.8 | 67.8 |
| DKGI (Pi, GPT-6-Luna) | 57.8 | 88.0 | 63.8 |
| DKGI (Codex, GPT-6-Luna) | 59.8 | 88.0 | 66.6 |
| **DKGI (Pi, GPT-6-Astra)** | **79.4** | **91.8** | **74.5** |

Training-free DKGI with GPT-6-Astra improves over GRASP by **8.2 / 9.0 / 6.7** Hit@1 points on Amazon / MAG / Prime.
Baseline numbers are published results with different backbones, budgets and evaluation code. See the
[paper](https://arxiv.org/abs/2610.18616) for Recall@20, MRR and controlled ablations.

## Quickstart

Clone into a short path on **Linux**:

```bash
git clone https://github.com/leopoldwhite/DKGI.git ~/dkgi
cd ~/dkgi
```

Complete the [harness and dependency setup](docs/setup.md#requirements), including the pinned Pi/DCI-Agent-Lite
and Codex CLI versions, and set [`MODEL_BASE_URL`](docs/setup.md#environment-variables). Then download the exported
graphs and run an offline preflight:

```bash
hf download yuyangbai/DKGI-STaRK-Graphs --repo-type dataset --local-dir experiments/stark \
  --include "corpus/*" "public/*" "private/*" "prepared/*"
python3 experiments/stark/use_downloaded_graphs.py
python3 experiments/stark/dci_run/run.py pi_full_test --check --datasets prime
```

The preflight checks prompts and execution isolation without a model call or API key. Live runs additionally
require `MODEL_API_KEY`; [reproduction instructions](docs/reproduction.md) cover smoke checks, full runs and analysis.

## Documentation and data

| Guide or resource | Contents |
|:--|:--|
| [Setup and data](docs/setup.md) | Linux requirements, pinned dependencies, endpoint configuration and graph preparation |
| [Reproduction](docs/reproduction.md) | Offline checks, experiment commands, trajectory locations and analysis |
| [Repository reference](docs/reference.md) | Code layout, paper-to-code configuration map and text-constraint tags |
| [Exact prompts](prompts/README.md) | Final prompt, controlled variants, probes and annotation instructions |
| [Exported graphs](https://huggingface.co/datasets/yuyangbai/DKGI-STaRK-Graphs) | Node and edge files, questions and labels for Amazon, MAG and Prime |
| [Test trajectories](https://huggingface.co/datasets/yuyangbai/DKGI-Trajectories) | Recorded trajectories, answers and scores; prompt-validation rounds are excluded |

## Citation

```bibtex
@misc{bai2026dkgi,
  title  = {Knowledge Graphs as Files: An Agent Harness Where Tool Creation and Tool Use Co-Evolve},
  author = {Bai, Yuyang and Wang, Yu and Zhang, Yu},
  year   = {2026}
}
```

## License

Code is released under [MIT](LICENSE). Derived graph data, text-constraint tags, prompts and the trajectories on
Hugging Face are released under [CC BY 4.0](LICENSE-DATA.md). Third-party dependencies retain their own licenses;
the patch and overlay files in `third_party/` are MIT-licensed.
