# Pitch script: 2 minutes, four parts of 30 seconds

About 75 words per part at a pitch pace. Every number is on a slide; slide numbers follow
`deck.json` (Intro, Phase 1, Phase 2, Results).

## 1. Intro (0:00–0:30) · slides 1–3

We’re team Remember. When a language model keeps training on new data, it forgets what it knew.
Dozens of papers propose fixes, but each uses its own model, data and budget, so nobody can say
which fix works, and some disagree. We built two layers: a frozen bench that makes every
forgetting experiment comparable, and a research agent that runs experiments on it by itself.
**Takeaway: one bench, so results finally compare.**

## 2. Phase 1: the bench and the grid (0:30–1:00) · slides 4–5

Phase one is standardization. Gemma 3 1B trains on code, then legal text: the same stream every
run, twenty minutes on one Modal GPU. We score what it kept, what it learned, and what it cost.
Six published techniques plug in as switches. Before any results, we froze a pre-registration,
then ran each technique alone, fourteen combinations, and six published claims.
**Takeaway: rules fixed before the data, so no cherry-picking.**

## 3. Phase 2: the research agent (1:00–1:30) · slides 6–7

Phase two asks why the model forgets. Our agent runs unattended. It searches Exa for recent
papers, proposes a hypothesis with a prediction, and can write its own technique. Code, not the
model, decides what runs, and Exa checks prior work, with citations, before any compute is spent.
A fixed rule gives the verdict, and a skeptic re-tests what held. 108 hypotheses, 208 runs.
**Takeaway: an agent that does research and doubts itself.**

## 4. Results and key takeaways (1:30–2:00) · slides 8–12

Three takeaways. **One: tune the learning rate first.** With no fix the model loses 6.4 points;
with the published recipe, 3.1; with just a lower learning rate, 1.1, while learning just as
well. **Two: most published claims didn’t hold here.** Only two of six reproduced. **Three:
separate weights protect.** The agent found distillation with LoRA lost 15% of the most fragile
questions, against 69% on full weights. And everything rebuilds from committed results with one
command.

Pace: five slides in the last 30 seconds, about one every six seconds (one takeaway each on 8, 9
and 11, a glance at 10, close on 12). Slides 8–10 carry the eyebrow “Phase 1”: they are Phase 1’s
results.

## If asked

- Takeaway three is exploratory, at lr 1e-4, on the selection half, held at four seeds.
- The fragile-core measure is selected and scored on the same items; a held-out rebuild kept 5
  of 13 core confirmations. Say so if pressed.
