# Pitch slides

The round 1 pitch deck (3 minutes), as presented on 15 November 2026. Twelve slides.

- `slides.html`: every slide on one page, in order. Open it in a browser. Speaker notes are in each
  slide's `<aside>` (hidden on the page; read them in `slides/<id>.html`).
- `deck.json`, `deck.css`, `slides/<id>.html`: the deck's source files, in the order `deck.json` gives.

Numbers on the slides come from `results/analysis/grid_analysis.json` (the claims table, the grid,
the confirmatory comparison) and `results/agent/explore_summary.md` at commit be76eff (the research
agent). The reading behind the slides is `docs/analysis-report.md`.

`slides.html` is built by concatenating the slide files with the stylesheet. If a slide changes,
rebuild it the same way.

`assets/agent-diagram.webp` is the research-agent diagram on slide 9. In the live deck the slide loads it from the deck's own store; the copies here point at this file.
