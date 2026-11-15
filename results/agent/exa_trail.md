# Exa evidence trail

Where Exa is called (the only module that holds the key):
- `src/clharness/agent/search.py` `ExaSearch.search` — one web search with highlights; called by the Scout (agents.py LLMScout.gather, the queries in agents.SCOUT_QUERIES), by the explore loop's --search argument, and by run_next_search for the search the model chose itself
- `src/clharness/agent/search.py` `ExaSearch.answer` — one grounded answer with citations per accepted hypothesis, before compute is spent (explore.py ask_prior_work -> Hypothesis.prior_work); informational, never a gate

Exa calls on record: **14** (3 Scout searches, 3 explore searches, 8 prior-work answers). Source records: 118, unique URLs: 76.

Every title, snippet and answer below is text fetched from the web: it is quoted as data, escaped, and truncated. The only links are the URLs the files recorded.

## Scout: technique candidates found through Exa search

### `2026-11-15t17-11-37-00-00-0-claims-2-candidates.json`

- **SCALE width expansion** — SCALE inserts lightweight width expansions into linear modules while freezing all pretrained parameters to preserve the base model's functionality during continual pre-training. It needs a model-construction hook to install the expansion modules and a trainable-parameter selection hook to optimize only the added parameters.
  - SCALE: Upscaled Continual Learning of Large Language Models — <https://arxiv.org/pdf/2511.03270v1.pdf>
- **Spaced Repetition Training** — Spaced Repetition Training treats continual pre-training as adaptive review scheduling, selecting when individual historical examples return instead of relying on a fixed mixture and uniform sampling. It needs a stateful sampler hook that maintains per-example review state and schedules subsequent rehearsals.
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models (2026-08-18) — <https://arxiv.org/html/2608.17530>

## Hypotheses — `results/agent/explore-himani`

### h001 — rejected

- **Claim:** Tests the capacity account: adding 256 trainable hidden units per MLP while freezing every original parameter should reduce core forgetting relative to plain training at 3e-5. Predict approximately 33% core forgotten versus the control's 48.3%, with legal perplexity ratio at most 0.79. Core forgetting remaining near or above 48% despite reaching that legal-learning threshold would count against this concrete capacity-relief hypothesis. Protection accompanied by little legal learning would not su…
- **Origin:** literature — SCALE: Upscaled Continual Learning of Large Language Models — https://arxiv.org/pdf/2511.03270v1.pdf
- **Agent-written technique:** SCALE-inspired always-on gated MLP width expansion (checks passed: none recorded)
- **Gate:** rejected — does not build: agent-written technique refused: does not parse: invalid character '—' (U+2014) (line 1)
- **Prior work (Exa `answer`, before launch):** not asked

### h002 — rejected

- **Claim:** Tests directional interference: periodically reinforcing old-domain behavior should redirect learning away from damaging directions without adding capacity. Predict core forgetting near 38%, versus 48.3% in the control, with legal perplexity ratio at most 0.80. Core forgetting remaining within 5 percentage points of control despite meeting that learning threshold would count against this code-review version of directional protection. Success would challenge unavoidable capacity-driven overwrite,…
- **Origin:** literature — When to Review: Spaced Repetition for Continual Pre-Training of Language Models — https://arxiv.org/html/2608.17530
- **Agent-written technique:** SM-2-inspired code review with a fixed 5 percent token budget (checks passed: static allow-list, tiny-model training)
- **Gate:** rejected — probe run did not finish: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
- **Prior work (Exa `answer`, before launch):** Research has not specifically tested the claim regarding directional interference as a mechanism to achieve the cited perplexity metrics. While recent literature explores parameter-space mechanisms for catastrophic forgetting, current findings emphasize that forgetting in language models is a consequence of drift on the pretraining distribution rather than an unavoidable capacity-driven overwrite &#91;1&#93;. Studies show that forgetting is controllable through data selection, replay, and optimization choices &#91;1&#93;. Research on Fisher-aware selectors has demonstrated that targeting specific parameter coordinates can improve target-domain quality while bounding forgetting more efficiently than replay &#91;2&#93;. However, there is no published evidence confirming that periodically reinforcing old-domain behavior can redirect learning to meet the specific 38% forgetting target or the 0.80 legal perplexity ratio mentioned in your query.
  - Forgetting in Language Models: Capacity, Optimization, and Self-Generated Replay (2026-05-25) — <https://arxiv.org/abs/2605.26097>
  - Fisher-Guided Submodular Data Selection for Continual Pre-Training of Large Language Models — <https://arxiv.org/html/2610.02593>
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models | Paperlayer (2026-08-18) — <https://paperlayer.ai/abs/2608.17530v1>
  - Simple and Scalable Strategies to Continually Pre-train Large Language Models (2024-09-04) — <https://arxiv.org/abs/2403.08763>
  - An Efficient Rehearsal Scheme for Catastrophic Forgetting Mitigation during Multi-stage Fine-tuning (2024-02-12) — <https://arxiv.org/abs/2402.08096>
  - GeRe: Towards Efficient Anti-Forgetting in Continual Learning of LLM via General Samples Replay (2025-08-06) — <https://arxiv.org/abs/2508.04676>
  - TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining — <https://aclanthology.org/2025.acl-long.1551.pdf>
  - Reuse, Don't Retrain: A Recipe for Continued Pretraining of Language Models (2024-07-09) — <https://arxiv.org/abs/2407.07263>

### h003 — not_replicated

- **Claim:** Tests readout drift: if residual forgetting principally reflects a moving output-side computation, protecting the top four layers and embeddings should substantially recover the core. Predict little change, with core forgetting remaining near 48%. A reduction of at least 15 percentage points while legal perplexity ratio remains at most 0.80 would count against that prediction and favor readout drift over lower-layer localized damage. Conversely, no protection at comparable legal learning would c…
- **Origin:** follow_up
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** No. The provided continual-pretraining intervention studies test different freeze/rollback locations but none match the specific “protect top four layers and embeddings” readout-drift prediction or report the required “core forgetting near 48%” / “legal perplexity ratio ≤ 0.80” thresholds. In fact, component-level interventions find a strong V/O-specific recovery pattern and little for Q/K &#91;1&#93;, while freezing in continual pretraining is explored at coarse granularities (first/last 10 layers and modules) without your top-four-layer criterion &#91;2&#93;.
  - Not All Components Forget Alike: Component-Level Intervention Profiles in Continual Pretraining — <https://aclanthology.org/2026.findings-emnlp.1583.pdf>
  - Examining Forgetting in Continual Pre-training of Aligned Large Language Models (2024-01-06) — <https://arxiv.org/abs/2401.03129v1>
  - Spurious Forgetting in Continual Learning of Language Models | ML Anthology — <https://mlanthology.org/iclr/2025/zheng2025iclr-spurious/>
  - Catastrophic Forgetting as Accessibility Collapse: A Three-Level Framework for Knowledge Persistence in Continual Learning — <https://arxiv.org/pdf/2606.06032>
  - Continual Pre-Training Mitigates Forgetting in Language and Vision (2022-05-19) — <https://arxiv.org/abs/2205.09357>
  - Small Language Models through Strategic — <https://ijaem.net/issue_dcp/Mitigating%20Catastrophic%20Forgetting%20in%20Small%20Language%20Models%20through%20Strategic%20Layer%20Freezing.pdf>
  - { xidicai067,shengjieqiu6} — <https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf>
  - Gradient Localization Improves Lifelong Pretraining of Language Models — <https://aclanthology.org/2024.findings-emnlp.949.pdf>
- **Runs:** `7fcbc2da97f1` explore-h003-s0 finished; `7b1d84736816` explore-h003-s1 finished; `f0c9d77b2e91` control: calibration-code-legal-lr3e-5-s0 finished; `2bfe1a3b0256` control: no-mitigation-lr3e-05-s1 finished

### h004 — confirmed

- **Claim:** Tests localized damage in the lower transformer layers: freezing the bottom 13 layers should reduce core forgetting to approximately 33%, even though the upper layers remain trainable. Core forgetting within 5 percentage points of control while legal perplexity ratio is at most 0.80 would count against this lower-layer localization hypothesis. Protection with preserved legal learning would favor localized damage over a pure top-side readout account, although reduced total displacement remains an…
- **Origin:** follow_up
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** Published research has not specifically tested your 13-layer freezing hypothesis. While studies confirm that freezing bottom layers helps mitigate catastrophic forgetting &#91;1&#93;&#91;2&#93;, the literature suggests that the specific localized damage and performance thresholds you described have not been empirically validated or addressed as a distinct lower-layer localization hypothesis. Existing research focuses on broader findings, such as the effectiveness of freezing bottom layers to preserve task alignment &#91;1&#93; or the marked reduction in forgetting when freezing exceeds 95% of total parameters &#91;2&#93;.
  - { xidicai067,shengjieqiu6} — <https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf>
  - Low-Rank Adaptation Reduces Catastrophic Forgetting in Sequential Transformer Encoder Fine-Tuning (2026-03-29) — <https://arxiv.org/html/2603.27707>
  - Examining Forgetting in Continual Pre-training of Aligned Large Language Models (2024-01-06) — <https://arxiv.org/abs/2401.03129v1>
  - Not All Components Forget Alike: Component-Level Intervention Profiles in Continual Pretraining - ACL Anthology — <https://aclanthology.org/2026.findings-emnlp.1583/>
  - On the Retention of Edited Knowledge in Fine-Tuned Language Models — <https://arxiv.org/html/2507.14198v1>
  - Continual Pre-Training Mitigates Forgetting in Language and Vision (2022-05-19) — <https://arxiv.org/abs/2205.09357>
  - Gradient Localization Improves Lifelong Pretraining of Language Models — <https://aclanthology.org/2024.findings-emnlp.949.pdf>
  - Continual Pre-training of Language Models (2023-02-07) — <https://arxiv.org/abs/2302.03241>
- **Verdict:** **confirmed** — effect +17.24 points, threshold 8.88 (replication over seeds &#91;0, 1, 2, 3&#93;: outcome core_forgotten (share of 58 questions, control minus test): effect +17.24 points over 4 paired seeds, threshold 8.88)
- **Runs:** `faf70f905d93` explore-h004-s0 finished; `7df4ea2b6ad8` explore-h004-s1 finished; `f0c9d77b2e91` control: calibration-code-legal-lr3e-5-s0 finished; `2bfe1a3b0256` control: no-mitigation-lr3e-05-s1 finished

### h005 — refuted

- **Claim:** Tests displacement as the explanation of residual forgetting: final averaging contracts the effective update without selecting particular layers or directions. Predict no material core rescue, at most 5 percentage points, because the residual errors may depend more on damaging directions than on global movement. A reduction of at least 10 percentage points with legal perplexity ratio at most 0.80 would count against this prediction and favor a displacement explanation. Conversely, negligible cor…
- **Origin:** follow_up
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** Yes—published research has tested continual pretraining/fine-tuning interventions that probe which parts of the update cause residual forgetting. In “Not All Components Forget Alike” (EN/Findings EMNLP 2026), selective component rollback/freeze shows forgetting is sharply component-specific rather than explained by global residual update displacement: V/O rollback improved prior-domain Wiki perplexity (about −2.98 to −5.72) with no measurable current-domain learning cost, while Q/K rollback showed essentially no recovery and MLP rollback harmed new domains (+13.95 to +20.75). These results do not support a displacement-only explanation for residual forgetting. &#91;1&#93;
  - Not All Components Forget Alike: Component-Level Intervention Profiles in Continual Pretraining — <https://aclanthology.org/2026.findings-emnlp.1583.pdf>
  - { xidicai067,shengjieqiu6} — <https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf>
  - Efficient Continual Pre-training by Mitigating the Stability Gap (2024-06-21) — <https://arxiv.org/pdf/2406.14833>
  - An Empirical Study of Catastrophic Forgetting in LargeLanguage Models During Continual Fine-tuning (2025-01-05) — <https://arxiv.org/abs/2308.08747>
  - Learning Dynamics in Continual Pre-Training for Large Language Models (2025-05-12) — <https://arxiv.org/pdf/2505.07796>
  - ICLR Poster Spurious Forgetting in Continual Learning of Language Models (2025-01-29) — <https://www.iclr.cc/virtual/2025/poster/29593>
  - Continual Pre-Training of Large Language Models: How to (re)warm your model? (2023-08-08) — <https://arxiv.org/abs/2308.04014>
  - Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning (2025-04-09) — <https://arxiv.org/pdf/2504.07097>
- **Verdict:** **refuted** — effect +16.38 points, threshold 12.87 (outcome core_forgotten (share of 58 questions, control minus test): effect +16.38 points over 2 paired seeds, threshold 12.87)
- **Runs:** `0548c1bf4557` explore-h005-s0 finished; `1f1fafc4bbab` explore-h005-s1 finished; `f0c9d77b2e91` control: calibration-code-legal-lr3e-5-s0 finished; `2bfe1a3b0256` control: no-mitigation-lr3e-05-s1 finished

### h006 — confirmed

- **Claim:** Skeptic check of h003: Run embedding-only freezing at lr 3e-5 with seeds 0 and 1 against freeze@lr3e-05. Removing top-layer freezing while retaining embedding freezing isolates the bundled intervention's cheapest non-localization explanation. If embeddings account for the benefit, core forgetting should remain near the control's 35%, ideally with comparable or better legal learning.
- **Origin:** skeptic
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** Yes—related continual pretraining/fine-tuning “freeze” experiments are reported in published research. In ICLR 2025 “Free Lunch for Mitigating Spurious Forgetting,” freezing bottom layers during sequential fine-tuning (keeping parts including embeddings fixed) substantially reduces forgetting, improving Task 0 performance from ~11% to ~44% while updating &lt; half the parameters &#91;1&#93;&#91;2&#93;. However, the provided sources don’t report the exact embedding-only freezing ablation at lr=3e-5 with seeds 0/1 or your specific ~35% “core forgetting”/legal-learning outcome, so that precise claim isn’t directly confirmed here. &#91;1&#93;&#91;3&#93;
  - { xidicai067,shengjieqiu6} — <https://proceedings.iclr.cc/paper_files/paper/2025/file/a774503daed55eb53c634847ae071ec7-Paper-Conference.pdf>
  - Spurious Forgetting in Continual Learning of Language Models — <https://iclr.cc/media/iclr-2025/Slides/29593.pdf>
  - Examining Forgetting in Continual Pre-training of Aligned Large Language Models (2024-01-06) — <https://arxiv.org/abs/2401.03129v1>
  - Not All Components Forget Alike: Component-Level Intervention Profiles in Continual Pretraining - ACL Anthology — <https://aclanthology.org/2026.findings-emnlp.1583/>
  - Gradient Localization Improves Lifelong Pretraining of Language Models — <https://aclanthology.org/2024.findings-emnlp.949.pdf>
  - { phi} { timothy.anderson.20} — <https://www.cs.jhu.edu/%7Ekevinduh/papers/thompson18freezing.pdf>
  - Exploring Continual Fine-Tuning for Enhancing Language Ability in Large Language Model (2024-10-21) — <https://arxiv.org/pdf/2410.16006>
  - (no title) — <https://www.arxiv.org/pdf/2009.14109v1>
- **Verdict:** **confirmed** — effect -4.74 points, threshold 8.84 (replication over seeds &#91;0, 1, 2, 3&#93;: outcome core_forgotten (share of 58 questions, control minus test): effect -4.74 points over 4 paired seeds, threshold 8.84)
- **Runs:** `694f39626f59` explore-h006-s0 finished; `67ebba481108` explore-h006-s1 finished; `7fcbc2da97f1` control: explore-h003-s0 finished; `7b1d84736816` control: explore-h003-s1 finished

### h007 — inconclusive

- **Claim:** Tests directional interference using a deliberately coarse, gradient-alignment-inspired surrogate: after code training, project each legal optimizer update onto the half-space that does not reverse the accumulated code update. Predict 12 percentage points less core forgetting than plain training at 3e-5, provided legal perplexity ratio remains at most 0.80. Improvement below 5 points at that learning threshold would count against this aggregate-direction version of directional interference, alth…
- **Origin:** literature — Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models, https://arxiv.org/html/2508.01908
- **Agent-written technique:** Code-displacement half-space projection (checks passed: static allow-list, tiny-model training, four-step run on a GPU)
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** There is no published research validating your specific proposed surrogate of projecting optimizer updates onto a half-space to mitigate directional interference. While research exists on using orthogonal projections to prevent catastrophic forgetting in continual pretraining &#91;1&#93;&#91;2&#93;&#91;3&#93;&#91;4&#93;, these methods typically constrain updates to a subspace orthogonal to historical gradients or functional directions &#91;1&#93;&#91;4&#93;, rather than using the specific half-space projection approach described in your claim. Published studies emphasize that projecting the final applied update—rather than the raw gradient—is essential to prevent stateful optimizers like Adam from leaking updates into protected directions &#91;1&#93;.
  - Towards Continual Learning: Mitigating Forgetting With Subspace Tuning — <https://borowiecki.dev/pdf/2610.22293>
  - Continuous Subspace Optimization for Continual Learning — <http://www.lamda.nju.edu.cn/chengq/pdf/NeurIPS-2025-Cheng.pdf>
  - Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning (2025-04-09) — <https://arxiv.org/pdf/2504.07097>
  - Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings (2025-09-28) — <https://arxiv.org/html/2509.23893v1>
  - Orthogonal Subspace Learning for Language Model Continual Learning (2023-01-01) — <https://doi.org/10.18653/v1/2023.findings-emnlp.715>
  - Gradient Projection For Parameter-Efficient Continual Learning — <https://arxiv.org/html/2405.13383v1>
  - Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language Models — <https://aclanthology.org/2025.acl-long.940.pdf>
  - Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs (2026-01-26) — <https://www.arxiv.org/pdf/2601.18255>
- **Verdict:** **inconclusive** — effect +4.31 points, threshold 13.09 (outcome core_forgotten (share of 58 questions, control minus test): effect +4.31 points over 2 paired seeds, threshold 13.09)
- **Runs:** `7b7b12d04802` explore-h007-s0 finished; `ea809a12eadd` explore-h007-s1 finished; `f0c9d77b2e91` control: calibration-code-legal-lr3e-5-s0 finished; `2bfe1a3b0256` control: no-mitigation-lr3e-05-s1 finished

### h008 — rejected

- **Claim:** Tests the vocabulary-interface version of readout drift against damage within transformer blocks. Freeze embeddings while leaving every transformer block trainable. Predict approximately 3 percentage points less core forgetting, substantially short of sufficient protection. A reduction of at least 12 points with legal perplexity ratio at most 0.80 would count against that prediction and favor interface-level readout drift. Improvement below 5 points at comparable legal learning would count again…
- **Origin:** follow_up
- **Gate:** rejected — this configuration already has a row or is in the pre-registered design
- **Prior work (Exa `answer`, before launch):** not asked

### h009 — confirmed_with_cost

- **Claim:** Tests directional interference through functional rather than parameter-space protection: distillation penalizes behaviorally consequential movement while leaving many parameter directions unconstrained. Predict 12 percentage points less core forgetting than plain training at 3e-5, conditional on legal perplexity ratio at most 0.80. Improvement below 5 points despite that legal learning would count against the hypothesis that preserving teacher outputs on the encountered training distribution pr…
- **Origin:** follow_up
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** Yes—closely related continual fine-tuning/distillation-style methods have been tested, and they generally support “directional/functional interference control” over naive unconstrained training. For continual pretraining/fine-tuning, lifelong pretraining (domain-incremental) finds distillation-based approaches have the strongest forgetting prevention (negative forgetting rates), implying better retention than non-distillation baselines &#91;1&#93;. For explicitly functional-direction interference control, DOC continual fine-tuning tracks drifting functional directions and enforces orthogonality via modified gradients, achieving significantly reduced catastrophic forgetting versus prior methods &#91;2&#93;. These results indicate protection can come from constraining update directions/directions affecting functions, not necessarily parameter-space distance.
  - Lifelong Pretraining: Continually Adapting Language Models to Emerging Corpora — <https://aclanthology.org/2022.naacl-main.351.pdf>
  - Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings — <https://arxiv.org/pdf/2509.23893v1.pdf>
  - Towards Continual Learning: Mitigating Forgetting With Subspace Tuning — <https://borowiecki.dev/pdf/2610.22293>
  - Does Continual Learning Equally Forget All Parameters? — <https://proceedings.mlr.press/v202/zhao23n/zhao23n.pdf>
  - Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning (2025-04-09) — <https://arxiv.org/html/2504.07097>
  - Continual Pre-training of Language Models (2023-02-07) — <https://arxiv.org/abs/2302.03241>
  - Preventing Catastrophic Forgetting in Continual Learning of New Natural Language Tasks (2023-02-22) — <https://arxiv.org/abs/2302.11074>
  - SDC-LoRA: Singular-Subspace Drift Controlled LoRA to Mitigate Knowledge Forgetting — <https://aclanthology.org/2026.findings-acl.1207.pdf>
- **Runs:** `0fbd25463537` explore-h009-s0 finished; `e5b511744a86` explore-h009-s1 finished; `f0c9d77b2e91` control: calibration-code-legal-lr3e-5-s0 finished; `2bfe1a3b0256` control: no-mitigation-lr3e-05-s1 finished

### h010 — inconclusive

- **Claim:** Skeptic check of h009: Run plain training at lr=1e-5 with no mitigation on seeds 0 and 1 against distill@lr3e-05. This lowers adaptation without teacher-output protection, targeting the distillation arm's legal perplexity ratio of approximately 0.793. If weaker legal learning explains the protection, the plain run should have comparable core forgetting. Interpret this as a learning-matched check only if the legal ratios are approximately matched.
- **Origin:** skeptic
- **Gate:** accepted
- **Prior work (Exa `answer`, before launch):** There is no published research that directly tests your specific hypothesis regarding a 1e-5 learning rate plain training run against a 3e-5 distillation run to isolate legal perplexity ratios or core forgetting mechanisms. While research has investigated catastrophic forgetting &#91;1&#93;&#91;2&#93; and the impact of learning rates on upstream versus downstream performance &#91;3&#93;, these studies typically focus on re-warming strategies, replay, or scaling laws rather than using legal perplexity ratios to control for learning speed in distillation-based comparisons &#91;4&#93;&#91;3&#93;&#91;5&#93;. Existing work confirms that forgetting is a non-trivial challenge &#91;2&#93; and that simple methods often fail to mitigate it &#91;2&#93;, but your proposed control experiment remains untested in the literature.
  - Continual Pre-Training Mitigates Forgetting in Language and Vision (2022-05-19) — <https://arxiv.org/abs/2205.09357>
  - Examining Forgetting in Continual Pre-training of Aligned Large Language Models (2024-01-06) — <https://arxiv.org/abs/2401.03129v1>
  - Continual Pre-Training of Large Language Models: How to (re)warm your model? (2023-08-08) — <https://arxiv.org/pdf/2308.04014>
  - Simple and Scalable Strategies to Continually Pre-train Large Language Models (2024-09-04) — <https://arxiv.org/abs/2403.08763>
  - Learning Dynamics in Continual Pre-Training for Large Language Models — <https://arxiv.org/html/2505.07796v2>
  - Investigating Continual Pretraining in Large Language Models: Insights and Implications — <https://arxiv.org/pdf/2402.17400v2.pdf>
  - Improving Representational Continuity via Continued Pretraining (2023-02-26) — <https://arxiv.org/abs/2302.13289>
  - Less is More: Task-aware Layer-wise Distillation for Language Model Compression — <https://export.arxiv.org/pdf/2210.01351v3.pdf>
- **Verdict:** **inconclusive** (a test run did not finish (failed): OSError: &#91;Errno 22&#93; Invalid argument)
- **Runs:** `a575f87e1fcb` explore-h010-s0 failed; `2c6f21ced4ed` explore-h010-s1 finished; `0fbd25463537` control: explore-h009-s0 finished; `e5b511744a86` control: explore-h009-s1 finished

### h011 — rejected

- **Claim:** Tests directional interference using a diagonal old-domain sensitivity surrogate, rather than the previously tested cumulative code-displacement direction. During code training, estimate gradient second moments; during legal training, suppress updates to sensitive coordinates and restore the original global step norm. Predict 12 percentage points less core forgetting, conditional on legal perplexity ratio at most 0.80. Improvement below 5 points at that learning threshold would count against thi…
- **Origin:** literature — Inspired by the old-task sharp-direction mechanism in On the Implicit Adversariality of Catastrophic Forgetting in Deep Continual Learning, https://arxiv.org/html/2510.09181 . The hook uses a diagonal gradient-second-moment surrogate, not the paper's exact algorithm.
- **Agent-written technique:** Step-norm-preserving old-gradient coordinate shielding (checks passed: static allow-list, tiny-model training)
- **Gate:** rejected — costs $3.08, only $2.35 left under the caps
- **Prior work (Exa `answer`, before launch):** not asked

### h012 — rejected

- **Claim:** Tests localized damage, specifically whether the replicated benefit of freezing the bottom 13 blocks is concentrated within the bottom four. Freeze only those four blocks, leaving embeddings and all upper blocks trainable. Predict 10 percentage points less core forgetting with legal perplexity ratio at most 0.80. Improvement below 5 points at comparable legal learning would count against bottom-four localization and suggest that protection requires a broader or higher region. A substantial benef…
- **Origin:** follow_up
- **Gate:** rejected — costs $2.63, only $2.35 left under the caps
- **Prior work (Exa `answer`, before launch):** not asked

### h013 — rejected

- **Claim:** Tests directional interference through direct old-domain rehearsal: replace 5 percent of training rows with randomly sampled code replay, keeping batch shapes and optimizer steps unchanged. Predict 10 percentage points less core forgetting, conditional on legal perplexity ratio at most 0.80. Improvement below 5 points at that learning threshold would count against the hypothesis that accessible code-domain gradients sufficiently constrain the directions responsible for fragile-core loss. Protect…
- **Origin:** literature — Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models, https://arxiv.org/abs/2508.01908
- **Gate:** rejected — costs $2.63, only $2.35 left under the caps
- **Prior work (Exa `answer`, before launch):** not asked

## Searches the agent chose itself

- **Continual language model training: does diagonal empirical Fisher importance predict damaging update directions, and have norm-matched Fisher-filtered updates protected knowledge without reducing final weight displacement?** — asked by model, after h010, h011, h012, h013; not run yet
- **SCALE width expansion continual pretraining ablations frozen backbone always-on expansion versus routed expansion matched new-domain perplexity catastrophic forgetting** — asked by model, after h001; run through Exa
  - SCALE: Upscaled Continual Learning of Large Language Models — <https://aclanthology.org/2026.findings-acl.2037.pdf>
  - SCALE: Upscaled Continual Learning of Large Language Models - ACL Anthology — <https://aclanthology.org/2026.findings-acl.2037/>
  - SCALE: Upscaled Continual Learning of Large Language Models (2025-12-11) — <https://arxiv.org/html/2511.03270v2>
  - SCALE: Upscaled Continual Learning of Large Language Models (2026-01-01) — <https://doi.org/10.18653/v1/2026.findings-acl.2037>
  - SCALE: Upscaled Continual Learning of Large Language Models — <https://gengo.sotaro.io/2026.findings-acl.2037>
  - SCALE: Upscaled Continual Learning of Large Language Models | alphaXiv (2025-12-11) — <https://www.alphaxiv.org/abs/2511.03270>
  - Continual Pre-training of Language Models (2023-02-07) — <https://arxiv.org/pdf/2302.03241>
  - Investigating Continual Pretraining in Large Language Models: Insights and Implications — <https://arxiv.org/pdf/2402.17400v2.pdf>
  - Lifelong Pretraining: Continually Adapting Language Models to Emerging Corpora — <https://arxiv.org/pdf/2110.08534v3.pdf>
  - Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning (2025-03-27) — <https://arxiv.org/html/2403.18886v3>
- **When to Review spaced repetition continual pretraining SM-2 ablations adaptive scheduling versus uniform code replay matched replay tokens and new-domain perplexity** — asked by model, after h002, h003, h004, h005; run through Exa
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models (2026-08-18) — <https://arxiv.org/html/2608.17530>
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models | alphaXiv (2026-08-18) — <https://www.alphaxiv.org/abs/2608.17530>
  - The Flow - AI News Aggregator (2026-08-19) — <https://theflow.svmpsp.dev/story/when-to-review-spaced-repetition-for-continual-pre-training-of-language-c23f77f5>
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models — Large Language Models — <https://awesomepapers.io/llm-papers/papers/2608.17530>
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models — <https://arxiv.org/pdf/2608.17530v1.pdf>
  - Simple and Scalable Strategies to Continually Pre-train Large Language Models (2024-09-04) — <https://arxiv.org/html/2403.08763v4>
  - Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models — <https://arxiv.org/html/2508.01908>
  - &#91;PDF&#93; When to Review: Spaced Repetition for Continual Pre-Training of Language Models | Semantic Scholar — <https://www.semanticscholar.org/paper/When-to-Review%3A-Spaced-Repetition-for-Continual-of-Atreya-Batra/d205e7e61d41a89a06ca8400c813e51fffdc62f8>
  - When to Review: Spaced Repetition for Continual Pre-Training of Language Models (2026-08-18) — <https://arxiv.org/abs/2608.17530>
  - Which Pretrain Samples to Rehearse when Finetuning Pretrained Models? — <https://arxiv.org/html/2402.08096v1>
  - cited by hypothesis h002, h007
- **Revisiting Replay and Gradient Alignment for Continual Pre-Training: does alignment with cumulative previous-task weight displacement predict forgetting, compared with alignment to previous-task gradients?** — asked by model, after h006, h007, h008, h009; run through Exa
  - Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models — <https://arxiv.org/html/2508.01908>
  - Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models (2026-04-06) — <https://proceedings.mlr.press/v330/abbes26a.html>
  - (no title) (2018-10-29) — <https://arxiv.org/pdf/1810.11910>
  - On the Implicit Adversariality of Catastrophic Forgetting in Deep Continual Learning — <https://arxiv.org/html/2510.09181>
  - Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models | alphaXiv (2025-08-03) — <https://www.alphaxiv.org/abs/2508.01908>
  - Replay &amp; Gradient Alignment in LLM Pre-Training (2025-08-03) — <https://www.emergentmind.com/papers/2508.01908>
  - chandar-lab/continual-pretraining (2024-07-26) — <https://github.com/chandar-lab/continual-pretraining>
  - Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models (2025-08-03) — <https://arxiv.org/abs/2508.01908>
  - Gradient Alignment &amp; Meta-Experience Replay — <https://www.emergentmind.com/topics/gradient-alignment-meta-experience-replay-mer>
  - Meta-Experience Replay (MER) — <https://www.emergentmind.com/topics/meta-experience-replay-mer>
  - cited by hypothesis h007, h011, h013
