# Exa challenge entry

This repository is a continual-learning test bench: a small language model is trained on a
stream of domains while a fixed eval suite measures what it forgets, and a research agent
proposes, gates and launches the experiments itself. Exa is the agent's only access to
the live web, and it is in the main loop, not a side tool: a Scout step finds methods the
agent was not told about and the proposer may rebuild one as new plugin code; every
hypothesis the gate accepts gets an `answer` call asking whether it was already tested,
before any GPU time is spent; and after each cycle the model names the one search it wants
run next. Search is central because it closes the loop between the literature and the
runs — what Exa returns becomes candidates, hypothesis sources, prior-work verdicts and
the next query.

## Where Exa is called

All Exa calls go through `src/clharness/agent/search.py`, the only module that reads
`EXA_API_KEY`:

| Function | Who calls it | For what |
|---|---|---|
| `ExaSearch.search` | `agents.py` `LLMScout.gather` | Three queries per Scout step (`agents.SCOUT_QUERIES`); the sources returned back the Scout's technique candidates and claim cards |
| `ExaSearch.search` | `explore.py` `_search_sources` | The `--search` argument, and `run_next_search`, which runs the one search the model chose after a cycle (`next_search.json` → `sources.jsonl`) |
| `ExaSearch.answer` | `explore.py` `ask_prior_work` | One grounded answer with citations per accepted hypothesis, before compute is spent; stored on the hypothesis as `prior_work`. Informational, never a gate |
| `ExaSearch.answer` | `agent/exa_check.py` (`clh exa-check`) | The same prior-work call, run live on one claim for the demo |
| `ExaSearch.find_similar` | the explore cycle | Pages like a paper on other sites, run on the source paper of each accepted `literature` hypothesis (`Hypothesis.similar`). Added in #24; no paper-sourced hypothesis was accepted after it landed, so no call is on record yet |

## The loop

1. **Scout.** `LLMScout.gather` runs three Exa searches; a model drafts technique candidates
   and claim cards, each citing the sources it saw (`results/agent/loop-*/candidates/*.json`).
2. **Propose.** The Scout's candidates and their papers are quoted into the explore prompt
   (`explore.py` `scout_sources`); the proposer may adopt one as an agent-written technique
   (`origin: "literature"`, `source` = the paper).
3. **Check.** An agent-written technique passes a static allow-list check, a tiny-model
   training check and a four-step GPU probe before it may cost a full run.
4. **Gate.** Plain code refuses what does not build, sits outside the declared ranges,
   repeats a tested arm or exceeds the budget.
5. **Prior work.** For each accepted hypothesis, `ask_prior_work` sends one
   `ExaSearch.answer` call — has this already been tested, and with what result? The answer
   and its citations are stored on the hypothesis.
6. **Run.** Accepted specs launch on Modal GPUs at two paired seeds; per-item files are
   fetched back.
7. **Verdict.** A fixed rule resolves each hypothesis against its control (confirmed /
   refuted / inconclusive). Under `--skeptic`, a confirmation is replicated at two more
   seeds before it may be a control.
8. **Next search.** The proposer names `next_search`; code runs it through Exa before the
   following cycle and quotes the results back to the model.

## Evidence

- [`results/agent/exa_trail.md`](../results/agent/exa_trail.md) — the whole Exa trail as
  one page: call counts, Scout candidates with their papers, every hypothesis's prior-work
  answer and citations, and the searches the agent chose itself.
- [`results/agent/explore-himani/`](../results/agent/explore-himani/) —
  `hypotheses.jsonl` (`prior_work` on each accepted hypothesis), `decisions.jsonl` (the
  hash-chained decision log; `explore/search` entries carry the URLs Exa returned),
  `sources.jsonl` (every source fetched, one record per result).
- [`results/agent/loop-himani/candidates/`](../results/agent/loop-himani/candidates/) —
  the Scout's result files: candidates and claim cards whose `source_ids` point into the
  sources Exa returned.

## Without search

What the loop could and could not do without Exa, from the committed files:

| Without Exa, the loop has no... | Evidence with Exa (instance himani, 4 cycles) |
|---|---|
| Methods from papers it was not given | The Scout's 3 searches returned 24 sources; 2 candidates came from 2025-2026 papers (SCALE, "When to Review"). The agent built 3 techniques from Exa-found papers; one (h007, from "Revisiting Replay and Gradient Alignment") passed all three checks and ran: inconclusive |
| Prior-work verdicts before compute | 8 answers, 64 citations, newest August 2026 (table below) |
| Searches of its own | 3 searches the model chose after a cycle, run at the next cycle's start, 30 URLs returned |

A controlled ablation of the proposer alone (`results/agent/ablation/`, dry runs, 3 with and 3
without the Scout's Exa material, from the cycle history and from a fresh start, same focus):
with a neutral focus the proposer built **0 of 12** hypotheses from the papers in either
condition. It used the papers when its focus asked it to, as in the live cycles. So Exa's
effect on what gets proposed depends on the instruction; its effect on what is known before
a run (the prior-work answers) and on what the agent reads next (its own searches) does not.

## Prior work

What Exa answered, before any GPU time, for each accepted hypothesis (8 citations each; full
answers and URLs in `results/agent/exa_trail.md`):

| Hypothesis | What it tested | Exa said | Run verdict |
|---|---|---|---|
| h002 | spaced-repetition code review (agent-written, from an Aug 2026 paper) | not specifically tested | refused by the GPU probe |
| h003 | freeze top 4 blocks + embeddings | not tested at this granularity | did not replicate |
| h004 | freeze bottom 13 blocks | not tested at 13 blocks; bottom-layer freezing known to help | **confirmed, replicated at 4 seeds** |
| h005 | final weight averaging | related work exists | refuted (it did protect) |
| h006 | embedding-only freezing (skeptic check) | related freeze experiments exist | confirmed, replicated |
| h007 | half-space projection (agent-written, from a 2025 paper) | no published validation of this surrogate | inconclusive |
| h009 | distillation | closely related methods tested | confirmed with cost (learns less) |
| h010 | lr 1e-5 vs distillation (skeptic check) | not directly tested | inconclusive (a run failed) |

5 of 8 were new as stated; 3 had related published work. The live demo below shows the same
call on the bench's strongest finding: Exa returns published work that freezing bottom layers
reduces forgetting, which the agent's replicated h004 agrees with.

## Live demo (60 s)

```bash
clh exa-check "freezing the bottom layers of a small transformer reduces catastrophic forgetting during continual pretraining"
```

This is the same `ExaSearch.answer` call the loop makes before spending GPU time: it prints
the question sent to Exa, Exa's answer, then the numbered citations (title, date, URL) the
answer is grounded in. Then rebuild the trail:

```bash
clh exa-trail results/agent/explore-himani results/agent/loop-himani   # writes results/agent/exa_trail.md
```

Open `results/agent/exa_trail.md`: the Exa call counts at the top, the Scout candidates each
with the papers behind them, each hypothesis's prior-work answer and citations, and the
searches the agent chose itself with the URLs Exa returned. The page reads committed files
only, makes no network calls, and is byte-identical on a rerun.

## Built during the hack

`git log -- src/clharness/agent/search.py src/clharness/agent/explore.py src/clharness/agent/trail.py`:
`3e29415` added the Exa-backed Scout on November 14; `cb16368` and `ea325d2` put Exa inside
the explore loop (Scout candidates reach the proposer, prior work before launch, the
model-chosen next search, `find_similar` on source papers) and `6615414` added `clh
exa-trail`, all on November 15. The harness itself predates the hackathon and is credited
in the README ("Prior work and hackathon work"); the Exa loop does not.
