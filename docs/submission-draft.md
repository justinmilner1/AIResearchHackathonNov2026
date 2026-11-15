# Submission draft

For brendan, who owns the submission (due Sunday 14:00; plan to submit by 13:45). Drafted by
Member A's agent on Saturday evening so that Sunday's work is filling in numbers and recording,
not writing from scratch. Edit freely.

**Every "TO FILL" is a number or sentence that does not exist yet.** Do not submit with one left
in. Numbers already filled in come from committed rows and are marked with their file.

## What the platform asks for

| Item | Status |
|---|---|
| Track | Open Track (Track 5) |
| Challenges entered | Modal, Exa, past.dev: TO CONFIRM as a team. Originator: not entered unless the team agrees to upload agent sessions |
| Repository link | https://github.com/justinmilner1/AIResearchHackathonNov2026 |
| 2-minute demo video | Outline below. TO RECORD |
| Short project description | Draft below |

## Project description (draft, about 230 words)

**Remember: a test bench and research agent for forgetting in continually trained language
models.**

When a language model keeps training on new data, it forgets things it knew. Dozens of papers
propose fixes, each tested on a different model, data stream and compute budget, so their
results cannot be compared and some conflict.

Remember holds everything constant. One frozen bench trains Gemma 3 1B on code and then legal
text in about 20 minutes for about $1.30 on a single cloud GPU, and scores what the model kept,
what it learned and what the update cost. Six published techniques plug in as switches. An
agent launches the experiments on Modal and logs every decision; a pre-registration written
before the results fixes what counts as a win.

What we found:

- The published recipe (replay plus learning-rate re-warming) halves forgetting: −3.1 points
  against −6.4 with no mitigation.
- Simply lowering the learning rate does better: −1.1 points, with the same learning of the
  new domain. The recipe's schedule is, in effect, a learning-rate cut.
- Of 14 technique combinations screened, only two kept learning the new domain within the
  pre-registered margin. Replicated at 3 seeds, the better one (freezing, re-warming, replay and
  weight averaging together) beat the published recipe by 2.1 points (95% CI 0.8 to 3.4) but
  was level with simply lowering the learning rate (+0.1 points, 95% CI −1.1 to +1.2). No
  configuration met the pre-registered win rule.
- Run at the lower learning rate itself (exploratory follow-up, 3 seeds), that combination
  forgets less than lowering the learning rate alone: +1.3 points (95% CI 0.1 to 2.6). But it
  learns the legal text less (perplexity ratio 0.816 against 0.754), outside the pre-registered
  margin. Every gain in retention we found came with less learning of the new domain.
- Of six published claims tested on the bench, two reproduced (replay plus re-warming beats no
  mitigation; the peak learning rate is the main knob), one reproduced in part (LoRA forgets
  less and learns less, but rank 256 learned the legal text less than rank 8, the opposite of
  the claim), and three did not (replay alone at 5% or 25%, distillation on replay tokens,
  freezing the top half instead of the bottom).
- On a second, published task sequence (LoRA fine-tuning of Qwen3-8B on SST-2, then MNLI, then
  AG News, on Tinker; exploratory), plain sequential training barely forgets: 1.8 points (95%
  CI 1.0 to 2.9). Replaying 200 examples of each earlier task did not reduce it: 3.2 points
  (95% CI 1.5 to 5.4). So replay alone did not help here either. With so little to recover,
  the pre-registered "gap closed" (−1.4, 95% CI −5.5 to +0.5) is unstable and is not a
  finding on its own.
- Overnight, three instances of the research agent proposed 89 hypotheses from the results, each
  with a prediction written before its runs. Code refused 6, 158 full runs tested the rest at two
  seeds, and a fixed rule gave the verdicts: 10 confirmed, 5 confirmed at a cost in learning, 2
  refuted, 62 inconclusive (4 were still running at this count). 30 of the tested configurations
  were techniques the agent wrote itself. All of it is exploratory. The clearest leads:
  - Giving the new domain its own adapter weights protects old knowledge where a training
    constraint alone does not. Distillation with LoRA lost 15% of the 58 most fragile questions;
    the same distillation on full-weight training lost 69%, at the same legal perplexity (0.810
    and 0.808).
  - The damage sits in the MLP blocks, and so does the new learning. An agent-written technique
    that protected them lost 23% of the fragile questions against 53% for its control; protecting
    attention instead lost 55%. Undoing only the legal-stage MLP changes recovered the questions
    but also undid the legal learning (perplexity ratio 1.067).
  - A lower learning rate has a floor: at 1e-5 the model forgets 1.8 points less than at 3e-5 but
    learns the legal text less, as the agent predicted.

Everything reported rebuilds from committed result rows with `make reproduce`.

The harness and four September baseline runs predate the event and are imported in a labelled
first commit; everything else was built this weekend.

## Facts behind the description

So that nothing in it is typed from memory. Check each against its file before submitting.

| Claim | Value | Where it comes from |
|---|---|---|
| No mitigation, report half | −6.44 points, SD 1.15, 5 seeds | `results/rows.phase0.remeasured.jsonl` |
| Reference arm, report half | −3.13 points, 3 seeds | `results/rows.phase0.jsonl`, `results/rows.reference.remeasured.jsonl` |
| No mitigation at lr 3e-5, report half | −1.07 points, 3 seeds | `results/rows.phase0.remeasured.jsonl`, `results/rows.phase0.jsonl` |
| Legal perplexity after legal training | 0.787 (1e-4), 0.744 (reference), 0.754 (3e-5) | same files, `acquisition.own_boundary_ppl_ratio` |
| Run length and cost | 17 to 27 minutes, about $1.20 to $1.45 | `cost` in any row |
| Runs on Modal | 100 runs with committed rows (85 full runs, 15 short plumbing and sanity checks), 31.4 H100-hours of training, about $124 | every row file under `results/` except fake and tinker, counted once per spec hash (the `remeasured` files re-score runs already counted); `cost.wall_clock_s`, `cost.est_usd`. Re-measurement jobs and uncommitted checks are not included |
| Mean learning rate | reference arm 5.54e-5, no mitigation 9.20e-5, no mitigation at 3e-5 2.76e-5 | computed from the schedule itself (`lr_at` in `src/clharness/train/loop.py`, `configs/bench/fast.yaml`, `lr_rewarm` defaults); matches peter's review. The committed run files do not log `train/lr` |
| Screen | 14 technique combinations; 2 pass the acquisition gate | `results/analysis/grid_analysis.md`, "Promotion" |
| Best configuration vs reference arm, report half | +2.13 points (95% CI +0.80 to +3.43), Holm p = 0.006, 3 seeds; vs the 3e-5 control +0.07 (−1.07 to +1.23) | `results/analysis/grid_analysis.md`, "Confirmatory tests" |
| Best configuration at lr 3e-5 (follow-up, exploratory, not pre-registered) | vs no mitigation at 3e-5, 3 paired seeds, item bootstrap: report half +1.33 points (95% CI +0.10 to +2.63), selection half +1.77 (+0.73 to +2.80); legal perplexity ratio 0.816 vs 0.754 (fails the 1.05 acquisition margin; code 0.990 vs 1.027 passes) | `results/rows.followup.jsonl`, specs in `configs/bench/specs/followup/`; `attribution.paired_bootstrap` |
| Published claims | 6 tested: 2 reproduced, 1 in part, 3 not; claim 4 not tested (no Reptile plugin) | `results/analysis/grid_analysis.md`, "Published claims" |
| Second task sequence (C5, Tinker, exploratory) | Qwen3-8B, LoRA rank 32, SST-2 → MNLI → AG News, 3 seeds per arm. Average forgetting: naive 1.8 points (95% CI 1.0 to 2.9), replay-200 3.2 (1.5 to 5.4), joint 0.0. Final average accuracy: 0.896 / 0.882 / 0.908. Gap closed by replay-200 −1.40 (−5.48 to +0.48), with the section 10 warning: joint − naive = 0.012 < 0.03. $8.57 estimated for the 9 runs (`est_usd`; a first attempt that stalled in a network outage is not counted). The top-2 technique arms were dropped before any C5 result (amendment 7) | `results/tinker/analysis.md`, `results/tinker/rows.jsonl`; rebuild with `clh tinker-analyze` |
| Research agent overnight | 3 instances (justin, peter, peter2), 89 hypotheses: 6 refused by the code gate, 10 confirmed, 5 confirmed with cost (held, but learned legal more than 5% worse than the control), 2 refuted, 62 inconclusive, 4 still running. 158 full runs at 2 seeds, none failed, about $235 by the rows' estimates. 34 agent-written techniques proposed, 30 passed the three checks and ran. Leads: justin h033 (core lost 69% vs 15%, legal ratio 0.808 vs 0.810), h034 (23% vs 53%, legal 0.878 vs 0.812), h038 (55% vs 23%), h052 (28% vs 53%, legal 1.067 vs 0.812), h002 (+1.8 points, threshold 1.2, legal 0.777 vs 0.754). Selection half; exploratory | `results/agent/explore_summary.md`, rebuilt by `python -m clharness.agent.explore_summary` from `results/agent/explore-*/hypotheses.jsonl` and `rows.jsonl` (as of commit 3d911ce; rerun it if more explore rows are pushed, and recopy these numbers) |
| Research agent with Exa in the loop (instance himani, Sunday morning, exploratory) | 4 cycles, 13 hypotheses, 23 runs at lr 3e-5 vs no mitigation at 3e-5 ($30 cap reached): freezing the bottom 13 blocks cut fragile-core forgetting by 17.2 points, replicated at 4 seeds, legal learning kept (h004); freezing the top 4 blocks + embeddings did not replicate (h003); distillation cut it 18.1 points but learned legal worse (h009); the method built from an Exa-found paper (h007) was inconclusive (+4.3, threshold 13.1). 14 Exa calls: 8 prior-work answers with 64 citations, 3 agent-chosen searches. 2 runs failed and are kept as rows (a GPU-probe device error; an OSError) | `results/agent/explore-himani/`, `results/agent/exa_trail.md`, `results/agent/ablation/` |

The pre-registered win rule cannot be met on this bench: a technique would have to beat the
3e-5 control by more than 2.30 points, and that control loses only 1.07. Say this plainly. The
honest result is "no technique beat lowering the learning rate", reported with the measured
differences and intervals, never as "no technique helps".

## How it maps to the five judging criteria

| Criterion | What to point at |
|---|---|
| Research rigour | Frozen bench; noise floor from 5 seeds; pre-registration with a logged amendment history; a published reference arm and lower-learning-rate controls; selection and report halves of the eval items kept apart; three rounds of independent review with the fixes in git |
| Novelty | All techniques under one protocol at matched compute; a table of published claims tested for reproduction; the finding that learning rate dominates and that the published recipe is partly a learning-rate cut |
| Results | The control table above; the confirmatory table (best configuration +2.1 points over the published recipe, level with the lower-learning-rate control); the claims table (2 of 6 reproduced, 1 in part); the second task sequence on Tinker (replay-200 did not reduce forgetting). Live results page |
| Autonomy and self-improvement | The launcher runs the pre-registered grid unattended and resumably; the explore loop proposes hypotheses from results, a code gate accepts or refuses them, and verdicts feed the next cycle. Be exact: tonight's screen was launched by a deterministic launcher on a person's go, and the Conductor reads real data but launches nothing. Overnight three instances of the explore loop proposed 89 hypotheses; the code gate refused 6 and a fixed rule gave every verdict (10 confirmed, 5 confirmed with cost, 2 refuted, 62 inconclusive). The agent wrote 34 techniques of its own; 30 passed a syntax allow-list, a tiny-model check and a four-step GPU run, and were tested. Later cycles built on earlier verdicts: see the themes in `results/agent/explore_summary.md` |
| Interpretability and reproducibility | Hash-chained decision log; every row carries its spec, commit and cost; failed and invalid runs are rows too; `make reproduce`; this honest limits list |

## Sponsor challenges

- **Modal.** One GPU per experiment, 10 in parallel, launched by name from a deployed app that
  refuses callers at a different commit. 100 runs with committed rows, 31.4 H100-hours of
  training, about $124 (facts table above).
- **Exa.** Exa runs in the research agent's main loop, three ways (`ExaSearch.search`,
  `ExaSearch.answer`, `ExaSearch.find_similar` in `src/clharness/agent/search.py`):
  the Scout's searches find methods in recent papers and hand them to the proposer
  (`--scout-candidates`); before any GPU time, an Exa answer with citations says whether each
  accepted hypothesis was already tested (`ask_prior_work`), and later cycles see that verdict;
  after each cycle the agent writes its own next search, which runs at the next cycle's start
  (`run_next_search`). Evidence, all committed: `results/agent/exa_trail.md` (rebuild with
  `clh exa-trail results/agent/explore-himani results/agent/loop-himani`),
  `results/agent/explore-himani/` and `results/agent/loop-himani/`. In instance himani's 4 cycles:
  14 Exa calls (3 Scout searches, 3 searches the agent chose, 8 prior-work answers with 64
  citations, the newest from August 2026); the agent built 3 methods from Exa-found papers, and
  one (h007, from "Revisiting Replay and Gradient Alignment") passed the static, tiny-model and
  GPU checks and ran: inconclusive. Say plainly what an ablation showed
  (`results/agent/ablation/`): with a neutral focus the proposer did not pick up the papers on
  its own (0 of 12 hypotheses with or without the Scout's sources); it built from them when its
  focus asked it to.
- **past.dev.** `src/clharness/agent/pastdev.py`; README section "Shared memory between agents".
  The analyst, the Scout and the explorer write; the explorer reads what the other two wrote
  before it proposes. The Scout now recalls too (`_recall` in `src/clharness/agent/agents.py`).
  TO CONFIRM: that a Scout run with a recall is in a committed log before entering.

## Limits to state in the pitch

- One small model, a 20M-token stream, a harsh learning rate.
- The screen is one seed per configuration; only the top three and the controls are replicated.
- Two screen runs had been seen before the pre-registration froze, and several analysis fixes
  were logged after the grid launched (see the amendment log in `analysis/PREREG.md`).
- The harness and September baselines are prior work.

## Video outline (2:00)

| Time | On screen | Say |
|---|---|---|
| 0:00 to 0:15 | Title card, then one line of the bench config | The problem: models forget when they keep learning, and published fixes cannot be compared. We built a bench that holds everything constant: 20 minutes, about a dollar a run |
| 0:15 to 0:40 | Terminal: the launcher's dry run, then the decision log scrolling, then the Modal dashboard with 10 GPUs | One command launches the pre-registered grid on Modal. Every wave and every row is logged in a hash-chained file. It resumes if interrupted and stops at a budget cap |
| 0:40 to 1:05 | Results page: the runs table with the per-domain perplexity columns, then the frontier figure | What each configuration kept and what it learned. The controls: no mitigation loses 6.4 points, the published recipe 3.1, a lower learning rate 1.1 |
| 1:05 to 1:25 | The claims figure or table | We tested six published claims on the same bench: two reproduced, one in part, three did not |
| 1:25 to 1:45 | Results page, section "What the research agents found": the totals row, then themes 3 and 4 | Overnight the research agent proposed 89 hypotheses, each with a prediction made before the run. Code decides what runs and code gives the verdict. It wrote 30 techniques of its own that passed the checks. Its clearest lead: the forgetting sits in the MLP blocks, and separate adapter weights protect what a training constraint alone does not |
| 1:45 to 2:00 | `make reproduce` finishing; the limits list | Everything rebuilds from committed rows in seconds. One small model, a short stream: here is what we do not claim |

Record the screen segments on Sunday morning once the page shows real results. The first 40
seconds can be recorded tonight.

## Open questions for the team

- Which sponsor challenges are entered.
- Whether "tune the learning rate first" is the title line of the pitch.
- Who speaks in the 3-minute pitch and who takes harness questions.
