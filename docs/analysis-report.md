# Analysis report: what the project produced

Written on Sunday 15 November 2026, late morning, by justin's agent, from a working session with
justin on what the results mean. It is a reading of the committed results, for the pitch and
for anyone picking the work up later. It adds no new numbers: each figure names the file it
comes from, and anything relayed from a teammate and not re-derived here says so.

Where this report and a results file disagree, the results file is right.

## 1. Conclusion

We built a cheap, controlled bench for forgetting in continually trained language models and
a research agent that runs it. With them we showed that, on this bench, the standard published
fix is mostly a learning-rate cut: no pre-registered technique or combination beat simply
training at a lower learning rate. We did not find a technique that wins, or a confirmed
account of where forgetting happens. We came away with one lead that replicated at four seeds,
a description of how forgetting is distributed over questions, and a specific next experiment.

## 2. What was built

- **The bench.** Gemma 3 1B trains on code and then legal text, 10M tokens each; about 20
  minutes and about $1.30 a run on one Modal GPU. It scores what was kept (four multiple-choice
  tasks), what was learned (held-out legal perplexity) and what the run cost.
- **Six techniques as plug-ins:** replay, learning-rate re-warming, LoRA, freezing,
  distillation, weight averaging (`docs/TECHNIQUES.md`).
- **A pre-registration** (`analysis/PREREG.md`): the win rule, the noise floor and the split of
  eval items into a selection half and a report half, fixed before results were read.
- **An executor and launcher:** one command runs a planned grid on Modal, resumes, stops at a
  budget cap and refuses a caller at a different commit from the deployed code.
- **A research agent** (`docs/research-agent.md`): a model writes a hypothesis with a
  prediction before any run; code decides what may run; a fixed rule gives the verdict. The
  agent can write its own technique, which must pass three checks first. A skeptic step
  replicates confirmed results at two more seeds and tests the cheapest alternative explanation.
- **Analysis, figures, a results page and `make reproduce`.**

The training harness and four September baseline runs are prior work. The rest is from the
weekend.

## 3. Pre-registered results

Report half; source `results/analysis/grid_analysis.md` and the facts table in
`docs/submission-draft.md`.

| Finding | Numbers |
|---|---|
| Lowering the learning rate beats the published recipe | Points lost: no mitigation −6.44 (5 seeds); published recipe, replay plus re-warming, −3.13 (3 seeds); no technique at lr 3e-5, −1.07 (3 seeds). Legal perplexity ratio 0.787, 0.744, 0.754 |
| The best combination is level with the lower learning rate | Of 14 combinations, 2 passed the learning gate. The better one (freeze, re-warm, replay, weight averaging) beat the recipe by +2.13 (95% CI +0.80 to +3.43) and was +0.07 (−1.07 to +1.23) against lr 3e-5 |
| Nothing met the win rule | The rule required beating the lr 3e-5 control by more than 2.30 points; that control loses only 1.07 |
| Published claims | 6 tested: 2 reproduced, 1 in part, 3 not |

Wording that is accurate: "no pre-registered technique beat lowering the learning rate". Not
"no technique helps": several beat no mitigation.

### Learning rate

Low settings still learn, down to a point. At 3e-5 the legal ratio is slightly better than at
1e-4 (0.754 against 0.787); at 1e-5 it is worse (0.777; `results/agent/explore_summary.md`,
justin h002, replicated at four seeds). The learning rate sets a point on a trade-off curve
between keeping and learning. It does not remove the conflict.

The recommended way to judge any forgetting technique follows from this: plot it against the
plain learning-rate curve at matched new learning and ask whether it sits above the curve.

### LoRA

LoRA freezes the model and trains small added matrices. It is not distillation, which adds a
penalty keeping outputs close to the base model's.

- **Pre-registered:** rank 8 at the plugin's default adapter learning rate (1e-3) forgot much
  less than no mitigation, but every combination containing LoRA learned legal too little and
  failed the gate. Rank 256 at 1e-3 collapsed (about −18 to −19 points; `docs/TECHNIQUES.md`).
- **Exploratory, peter:** at a matched adapter learning rate of 1e-4, three seeds each, the
  legal ratio is 0.860, 0.806 and 0.767 at ranks 8, 64 and 256 (`docs/TECHNIQUES.md`). Higher
  rank learns more, which is the published claim's direction; the "opposite" verdict in the
  pre-registered table comes from the collapsed 1e-3 arm.
- **Reading:** LoRA has its own learning-rate question. Tuned, rank 256 lands about on the
  plain learning-rate curve, not clearly above it. No test of LoRA against the lr 3e-5 control
  was run.
- **A limit justin raised:** a fixed small adapter may not hold a long stream of new domains.
  Our rank results agree that capacity limits learning. Fresh adapters per domain, merged in,
  were not tested.

### Replay

No detectable benefit anywhere we tested it (`results/analysis/grid_analysis.md`): replay 5%
alone against no mitigation −0.03 (−1.53 to +1.50); 25% alone −1.10 (−2.73 to +0.50);
distillation on replay tokens against plain replay −0.10 (−1.07 to +0.87); choosing replay
examples by loss or novelty no better than random. On the second task sequence (Tinker,
`results/tinker/analysis.md`) replay did not reduce forgetting either.

The recipe that contains replay beat no mitigation by +2.83, but it loses to the plain low
learning rate, so its benefit most likely comes from the schedule. Say "no detectable benefit
on this bench": the published effect (1 to 3 points) is close to our noise (about 1.2), most
runs replayed general text and not code, and the budget is tiny.

## 4. What the research agents found (exploratory)

Source: `results/agent/explore_summary.md`, rebuilt by
`python -m clharness.agent.explore_summary`. As of commit 33706f8: four instances, 106
hypotheses, 198 full runs finished and 1 failed, 31 agent-written techniques run. Verdict
labels have shifted under stricter rules, so the raw counts of "confirmed" are not a result
and should stay out of the pitch.

### Explanations of forgetting that were tested

| Explanation | What it says | Where it stands |
|---|---|---|
| Capacity | New knowledge overwrites old; separate weights remove the conflict | Best supported: justin h033 and h024 replicated at four seeds. Rank and layer variations inconclusive |
| Directional interference | Only movement in particular directions harms old knowledge | Partial: distillation helped, usually at a cost in legal learning (h015 replicated, with cost), and far less on full weights than with LoRA. Projection and replay-selection tests showed nothing |
| Displacement | Forgetting depends only on how far the weights move | Not sufficient alone (LoRA moved slightly further than full weights at 3e-5 and forgot less), but "less adaptation" explains much of the protection seen |
| Localized damage | The harm is in particular layers or modules | Open. See below |
| Readout drift | The knowledge is intact; only the top of the model moved | No support: every test inconclusive with small effects |

Narrower ideas: freezing normalization gains, freezing attention query and key projections,
and protecting token embeddings each made no detectable difference; smoothing optimizer updates
over time was refuted (forgetting got worse and legal learning collapsed, h011); a moving
average of the weights at lr 3e-5 helped at two seeds (h008) and is being replicated.

### The strongest lead: separate weights

justin h033, four seeds: with the same distillation and the same legal learning (ratio 0.808
and 0.810), full weights lost 69% of the fragile questions and LoRA adapters 15%. It is scoped
to learning rate 1e-4, and nobody has checked the literature for it.

### Forgetting is concentrated in questions

Source: `manifests/eval-protected-v1-core.json`, `results/analysis/forgetting/summary.md`.

- Of 683 questions the base model answers correctly on the selection half, 58 are forgotten by
  plain training at 1e-4 in at least 4 of 5 seeds: the "fragile core" (23 ARC-Easy, 18
  HellaSwag, 12 PubMedQA, 5 PIQA).
- Seeds lose the same questions far more often than chance, and switch to the same wrong
  answer 87% to 98% of the time where chance is about 47%.
- In many arms a quarter to a half of the questions lost after the code stage are answered
  correctly again at the end.
- Whether these are "hard" questions is unknown: the column for the base model's confidence
  on lost questions is empty in the committed summary.

### Whether it is concentrated in weights: not established

- justin h038, four seeds: with attention frozen and MLPs trainable the model lost 55% of the
  fragile questions, against 23% with MLPs frozen and attention trainable.
- That does not show the MLPs are special. Freezing them removes far more trainable weights
  and the model learns legal less (ratio 0.878 against 0.810).
- The skeptic checked this. Plain training at about lr 1e-5 with the same blocks frozen, and no
  MLP-specific step, lost 14% against the MLP-protected arm's 23% at a better legal ratio
  (0.851 against 0.878; h062), and 16% against the restored-MLP arm's 28% (0.852 against
  1.067; h060). So how much the model adapts explains the protection at least as well as where.
  h034 is marked challenged. h060 is labelled "refuted" in the file because it predicted the two
  arms would be about equal; the direction is that plain slow training did better.
- justin h017 and h036 added adapters to the MLPs as well as attention and saw no extra
  forgetting.

So: more forgetting was seen when the MLPs were trainable, and it could not be separated from
the model adapting more. Do not headline "the damage sits in the MLPs".

### Relayed from teammates, not re-derived here

- **brendan's review** of the overnight hypotheses (team channel, 15 November): nothing
  overnight sits clearly above the plain learning-rate curve at matched legal learning; the
  large fragile-core wins at lr 1e-4 are beaten on both axes by plain lr 1e-5; on a fragile set
  rebuilt from held-out questions, 5 of 13 core confirmations survive (justin h024, h033, h034,
  h038, h052); the core threshold counts item sampling only, and seed-to-seed spread on the
  core is about 5.6 points.
- **brendan's four-seed batch** (team channel; rows not on main when this was written):
  re-warming at lr 3e-5 loses 12.9 points less of the core than plain 3e-5 at the same legal
  ratio, but matches plain 2e-5 on every axis, so its decay is a learning-rate cut and not a
  new mechanism. Plain 2e-5 and 1e-5 beat the reference arm on the report half by more than
  the 2.30 margin. These are not pre-registered arms.
- **himani's Exa instance** (`results/agent/explore-himani`): freezing the bottom 13 blocks
  reduced fragile-core forgetting, replicated at four seeds.

## 5. Novelty, stated honestly

- **Not claimed:** a technique that beats the baseline; a located mechanism.
- **A negative result with an edge:** these techniques are usually reported without a tuned
  learning-rate baseline; with one, their advantage disappears here. The source paper itself
  says peak learning rate is the main knob, so the direction is not new.
- **A replication table:** three of six published claims did not reproduce at this scale, and
  one flips with the learning rate.
- **The method:** predictions before runs, code-decided launches and verdicts, agent-written
  techniques behind checks, and a skeptic step that challenged one of the agent's own findings.
- No literature check was run on the separate-weights lead or the same-wrong-answer pattern.

## 6. Worth exploring next

1. **Capacity over a long stream.** Four or five domains; a fresh adapter per domain merged
   afterwards, against one reused adapter and against full weights at a low learning rate, at
   matched new learning.
2. **Location, done fairly.** MLPs against attention with both arms matched on legal
   perplexity or on total weight movement. The agent queued this (h061) and it was refused at
   the budget cap.
3. **Schedule shape and endpoint averaging.** brendan's batch suggests re-warming is a
   learning-rate cut; the weight-averaging replication (h008) is still open.
4. **What the fragile questions share.** No new runs: does the base model's confidence on a
   question predict whether it is lost?

Not worth more runs on this bench: readout drift, normalization, attention routing, embeddings,
update smoothing, and replay variants (the published effect is below our noise).

## 7. What we would change

- Calibrate the bench before pre-registering. The harsh learning rate created most of the
  forgetting, and the lower-learning-rate control made the win rule nearly unreachable.
- More seeds, fewer arms. Noise of about 1.2 points against effects of 1 to 3.
- Start the agent with the skeptic rules on and the mechanism question from the first cycle.
- Show the proposer the threshold. brendan's review found half the accepted hypotheses
  predicted effects below it.
- Score the fragile core on held-out questions.
- One shared compute budget with a guard in the executor.
- Finish the integrations: the Conductor reads real data but launches nothing.

## 8. Limits

One small model, one domain pair, a 20M-token stream. The screen is one seed per
configuration. Agent results are two seeds unless stated, on the selection half, and most core
tests score on the questions used to pick the core. The harness and September baselines are
prior work.
