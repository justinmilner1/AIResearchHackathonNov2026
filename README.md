# Remember

A continual-learning test bench. Built at the SF AI Research Hackathon, November 14–15, 2026.

The project is called Remember. The Python package is still `clharness` and its command is
`clh`; those names predate the project name and are kept so that nothing running breaks.

A framework for running continual-learning experiments on small open language models, and for
measuring what each forgetting-reduction technique buys: train a model on a stream of new
domains, score a fixed set of protected capabilities at every checkpoint, and compare
techniques against a no-mitigation baseline.

## Objectives

- **Science:** improve on existing forgetting-reduction results.
- **Product:** a framework that makes continual-learning experiments easier to run.

The plan, priorities and division of work are in [`docs/DIRECTION.md`](docs/DIRECTION.md).

## What is in this repository

| Path | Contents |
|---|---|
| `src/clharness/` | The harness: training loop over an ordered domain stream, hook interface, evals, activation and SAE instruments, analysis and report commands |
| `configs/` | Model, eval-suite and run configs. Every result-affecting field is required |
| `manifests/` | Frozen eval item lists and sha256 manifests for the training data |
| `tests/` | Fast unit tests, plus an end-to-end smoke test that downloads the model |
| `scripts/` | GPU-box setup and sync helpers, throughput probe |
| `results/baselines/` | Base-model scores on the protected eval suite |
| `results/runs/` | Four no-mitigation runs: scores, per-question outcomes and configs at every checkpoint |
| `docs/DIRECTION.md` | Goal, priorities, workstreams and the rules for reported experiments |
| `docs/hackathon-brief.md` | The official event brief: tracks, judging, rules, schedule |
| `docs/harness.md` | Harness reference: setup, CLI, config format, layout, caveats |
| `docs/baselines.md` | The numbers to beat, with where each comes from |
| `docs/background/` | Literature review, field overview and pretraining overview |

## Prior work and hackathon work

The first commit of this repository imports work that existed before the event. Everything
after that commit was built during the hackathon.

Imported, built September 5–12, 2026:

- the harness (`src/`, `tests/`, `configs/`, `scripts/`, `manifests/`)
- the base-model baselines and the four no-mitigation runs under `results/`
- the background documents under `docs/background/`

## The baseline to beat

Gemma 3 1B (pretrained), trained in sequence on code, biomedical, legal and math text with no
mitigation. Scores are on a frozen protected set: ARC-Easy, HellaSwag, PIQA and PubMedQA
(500 questions each), and MBPP (200 problems).

| Run | Learning rate | Tokens per domain | Multiple-choice mean, base → end | Change (points) | MBPP, base → end |
|---|---|---|---|---|---|
| `pilot-lr1e-5-50M` | 1e-5 | 50M | 0.683 → 0.673 | −1.0 | 0.110 → 0.095 |
| `pilot-lr3e-5-50M` | 3e-5 | 50M | 0.683 → 0.649 | −3.4 | 0.110 → 0.055 |
| `pilot-lr1e-4-50M` | 1e-4 | 50M | 0.683 → 0.564 | −11.9 | 0.110 → 0.040 |
| `pilot-chosen-200M` | 3e-5 | 195M | 0.683 → 0.609 | −7.4 | 0.110 → 0.065 |

Details, per-task numbers and caveats are in [`docs/baselines.md`](docs/baselines.md).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), and for anything beyond the unit tests a
Hugging Face token with access to `google/gemma-3-1b-pt` and a GPU or Apple silicon.

```bash
cp .env.example .env      # put HF_TOKEN in it
make setup                # create the environment from the lockfile
make test                 # fast unit tests, no downloads
make smoke                # end-to-end run on a tiny data set (downloads model + SAEs)
```

Real data and a baseline run:

```bash
uv run clh build-data --set real --seq-len 1024      # ~6.7 GB, rebuilt from pinned Hugging Face revisions
uv run clh verify configs/pilot/base.yaml            # hash every file against manifests/real.json
uv run clh train configs/pilot/sweep/lr1e-4-50M.yaml
uv run clh analyze results/runs/<run>                # analysis.json + text summary
uv run clh items results/runs/<run>                  # which protected questions were lost, and when
```

The full command list and config format are in [`docs/harness.md`](docs/harness.md).

## Running on Modal

The repository targets Python 3.12. Install the optional Modal CLI in the project environment
instead of using the separate system `modal` command (which may use an older Python):

```bash
uv sync --extra dev --extra modal
uv run modal profile activate bkbrohon795
uv run modal profile current
```

Create the Hugging Face secret once if it is not already in the active Modal workspace, then
launch validated run specs on cloud GPUs:

```bash
uv run modal secret create huggingface HF_TOKEN=<your-hugging-face-token>
uv run modal run src/clharness/executor/modal_app.py::data
uv run clh run --modal --gpu H100 configs/bench/specs/*.json
```

Create the Hugging Face secret and build/cache the data once per Modal workspace; rerun the data
step only when you need to rebuild it. The Modal app uses Python 3.12 in its remote image too.
The Modal CLI reads the active local profile (`bkbrohon795` above). To use another GPU type,
change `--gpu` to a type enabled for your Modal workspace.

## Shared memory between agents (past.dev)

The agents keep a shared memory on [past.dev](https://past.dev) next to the repository's own
files. Which one holds what:

| Store | Holds | Role |
|---|---|---|
| Files in this repository (`results/**/rows.jsonl`, `results/agent/explore/hypotheses.jsonl`, the `decisions.jsonl` logs) | Every result row, hypothesis and decision | The truth. Every number in a report comes from here |
| past.dev | Plain-language statements of what each arm measured, which published claims the Scout found, and what each exploratory hypothesis predicted and concluded | A shared copy the agents recall from by question before they plan |

Who writes: the analyst step (one data point per arm, selection half only), the Scout (one per
claim card) and the explorer (one per hypothesis, replaced by its verdict). Who reads: the
explore loop recalls as `explorer` before it proposes hypotheses, so it plans from what the
analyst and the Scout recorded; the recalled text goes into the model's prompt marked as data,
and each recall and write is in the explore decision log. Any other agent can recall with
`python -m clharness.agent.pastdev recall "<question>" --as <agent>`.

Rules: nothing from the report half is ever written there, and the mirror is best effort (if
past.dev is down the run continues from the files). Code: `src/clharness/agent/pastdev.py`. Turn
it on with `--memory pastdev` on `clharness.agent.explore`; the key is read from `PAST_API_KEY`.

## Exa in the loop

The research agent uses the Exa API at every step of its literature loop, all through
`ExaSearch` in `src/clharness/agent/search.py` (`EXA_API_KEY` in `.env`):

1. **Scout** (`LLMScout` in `src/clharness/agent/agents.py`): three `ExaSearch.search` calls a
   step; the model drafts technique candidates and claim cards, each citing the sources it saw.
   Written to `results/agent/loop-*/candidates/*.json`.
2. **Proposer** (`src/clharness/agent/explore.py`): the Scout's candidates and their papers are
   quoted into the next prompt (`scout_sources`); the proposer may build one as an
   agent-written technique (`origin: "literature"`, `source` = the paper).
3. **Prior-work check**: for each hypothesis the gate accepts, one `ExaSearch.answer` call asks
   whether it was already tested, before any compute is spent (`ask_prior_work`; stored on the
   hypothesis as `prior_work` with its citations). Informational, never a gate.
4. **Model-chosen search**: the proposer also names the one web search it wants next
   (`next_search`); code runs it through Exa before the following cycle and shows the results.

The evidence lives in `results/agent/explore-*/{hypotheses.jsonl,decisions.jsonl,sources.jsonl,
next_search.json,rows.jsonl}` and `results/agent/loop-*/candidates/*.json`. One readable page
of the whole trail:

```bash
clh exa-trail results/agent/explore-himani results/agent/loop-himani   # -> results/agent/exa_trail.md
```

It reads committed files only, makes no network calls, and is byte-identical on a rerun.
The Exa challenge entry is [`docs/exa-challenge.md`](docs/exa-challenge.md).

## Adding a technique

Techniques plug into the training loop as hooks (`src/clharness/train/hooks.py`). A hook can:

- replace, filter or skip each micro-batch (`select_batch`), which covers replay and example selection
- read logits, per-sequence losses and captured activations after each forward pass (`on_micro_batch`)
- act after every optimizer step (`on_step_end`), which covers weight averaging
- measure at every checkpoint and domain boundary (`on_checkpoint`, `on_domain_start`, `on_domain_end`)

Three things the loop cannot do yet, each needed by a common technique:

- **Change the model before the optimizer is built.** The optimizer is created before
  `on_run_start`, so adapters (LoRA) and layer freezing cannot be applied from a hook.
- **Add a term to the loss.** The loss is fixed to next-token cross-entropy, so distillation and
  weight penalties have nowhere to attach.
- **Re-warm the learning rate per domain.** The schedule is global over the whole run.

## Data and licences

No third-party text is stored here. Training corpora are rebuilt from their public Hugging
Face sources by `clh build-data`, and `data/` is ignored by git. The corpora are FineWeb,
StarCoder data (Python), PubMed abstracts, Pile of Law court opinions and OpenWebMath. Each has
its own licence and terms; StarCoder data and Gemma require accepting terms on Hugging Face.

`results/runs/pilot-chosen-200M/items.json` contains the text of protected-set questions from
the public evals listed above.
