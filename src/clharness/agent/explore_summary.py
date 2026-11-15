"""What the research-agent instances found, as one readable page built from their files.

    python -m clharness.agent.explore_summary            # writes results/agent/explore_summary.md

Reads every `results/agent/explore-*/` folder (hypotheses, rows, model calls), the row files the
agents planned from, and `analysis/explore_themes.json`. Writes one Markdown file.

Every number in the output is computed here from committed files. The themes file holds prose
and hypothesis ids only: a theme says in words what a group of experiments was about, and this
module prints the measured numbers of those experiments under it. Nothing in it is a result.

Everything on the page is EXPLORATORY: two seeds per arm, selection half of the eval items,
verdicts by the fixed rule stored with each hypothesis. It is not part of the pre-registered
analysis and must not be mixed into it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clharness import spec as a1
from clharness.agent import explore
from clharness.config import REPO_ROOT

# The row files the instances planned from, in the order they were given to `--have`.
BASE_ROWS = ("results/rows.phase0.remeasured.jsonl", "results/rows.phase0.jsonl",
             "results/rows.reference.remeasured.jsonl", "results/agent/rows.jsonl", "results/rows.explore.jsonl")
THEMES = "analysis/explore_themes.json"
DECIDED_LABELS = ("confirmed", "confirmed with cost", "refuted", "not replicated")
UNITS = {"retention": "points of retention", "core_forgotten": "pp of the fragile core",
         "noncore_forgotten": "pp of the other known questions"}
PREDICTED = {"less_forgetting": "less forgetting", "more_forgetting": "more forgetting",
             "no_difference": "no difference"}


def _mean(values: Any) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _num(x: float | None, fmt: str = "{:+.2f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _cell(text: str, limit: int = 0) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if not limit or len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _arm_for(hashes: Sequence[str], arms: Sequence[explore.Arm]) -> explore.Arm | None:
    want = set(hashes)
    return next((a for a in arms if want & set(a.hashes_by_seed.values())), None)


def _describe(arm: explore.Arm | None, h: explore.Hypothesis | None = None) -> str:
    """An arm in words: its techniques with the parameters that were set, and its learning rate."""
    if arm is None:
        return "not run"
    parts = []
    for t in arm.techniques:
        if t.name == "agent_code":
            title = (h.new_technique.title if h is not None and h.new_technique else "") or t.params.get("title", "")
            parts.append(f"agent-written: {title}" if title else "agent-written technique")
        else:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(t.params.items()) if not isinstance(v, (dict, list)))
            parts.append(f"{t.name}({shown})" if shown else t.name)
    return (" + ".join(parts) or "no mitigation") + f", lr {arm.lr if arm.lr is not None else 1e-4:g}"


def load_instance(folder: Path) -> dict[str, Any]:
    hyps = explore.load_hypotheses(folder / "hypotheses.jsonl")
    rows = a1.read_rows(folder / "rows.jsonl") if (folder / "rows.jsonl").exists() else []
    calls = [json.loads(line) for line in (folder / "llm_calls.jsonl").read_text().splitlines() if line.strip()] \
        if (folder / "llm_calls.jsonl").exists() else []
    return {"name": folder.name.removeprefix("explore-"), "folder": folder, "hyps": hyps, "rows": rows,
            "model_usd": sum(float(c.get("cost_usd") or 0) for c in calls), "model_calls": len(calls),
            "calls_logged": (folder / "llm_calls.jsonl").exists()}


def record(h: explore.Hypothesis, arms: Sequence[explore.Arm]) -> dict[str, Any]:
    """One hypothesis with what its two arms measured, looked up by the spec hashes it stored."""
    test, control = _arm_for(h.test_spec_hashes, arms), _arm_for(h.control_spec_hashes, arms)
    if control is None and not h.control_spec_hashes:  # legacy record: no stored hashes to go by
        control = next((a for a in arms if a.name == h.control), None)

    def read(arm: explore.Arm | None, hashes: Sequence[str]) -> dict[str, float | None]:
        if arm is None:
            return {"legal": None, "core": None, "retention": None}
        seeds = [s for s, x in arm.hashes_by_seed.items() if x in set(hashes)] or list(arm.hashes_by_seed)
        return {"legal": _mean(arm.legal_ratio_by_seed[s] for s in seeds if s in arm.legal_ratio_by_seed),
                "core": _mean(arm.core_forgotten_by_seed[s] for s in seeds if s in arm.core_forgotten_by_seed),
                "retention": _mean(arm.retention_by_seed[s] for s in seeds if s in arm.retention_by_seed)}

    return {"h": h, "test": test, "control": control, "t": read(test, h.test_spec_hashes),
            "c": read(control, h.control_spec_hashes)}


def status(h: explore.Hypothesis) -> str:
    """The stored verdict, with the stricter label a confirmed result gets when it learned legal
    worse than its control by more than the margin (the --skeptic rule, applied to every record)."""
    if h.status == "confirmed" and h.acquisition_ok is False:
        return "confirmed with cost"
    return h.status.replace("_", " ")


def cells(inst: str, r: dict[str, Any]) -> list[str]:
    """One hypothesis as the nine table cells, plain text. The page renders the same cells."""
    h: explore.Hypothesis = r["h"]
    measured = "n/a" if h.effect_points is None else f"{h.effect_points:+.1f} (needs {h.threshold_points:.1f})"
    return [f"{inst} {h.id}", _cell(h.question, 150), _cell(_describe(r["test"], h), 110),
            _cell(_describe(r["control"]), 90),
            f"{PREDICTED[h.prediction.direction]}, {h.prediction.size_points:g} {UNITS[h.outcome]}", measured,
            f"{_num(r['t']['core'], '{:.0f}%')} / {_num(r['c']['core'], '{:.0f}%')}",
            f"{_num(r['t']['legal'], '{:.3f}')} / {_num(r['c']['legal'], '{:.3f}')}", status(h)]


COLUMNS = ["Hypothesis", "Question", "Tested", "Against", "Predicted before the run", "Measured effect",
           "Core lost, test / control", "Legal ratio, test / control", "Verdict"]


def _row(inst: str, r: dict[str, Any]) -> str:
    c = cells(inst, r)
    return "| " + " | ".join(c[:-1]) + f" | **{c[-1]}** |"


def totals(i: dict[str, Any]) -> dict[str, Any]:
    """The counts for one instance, from its records and rows."""
    c = Counter(status(h) for h in i["hyps"])
    own = [h for h in i["hyps"] if h.new_technique]
    full = [r for r in i["rows"] if not r.spec.name.endswith("-probe")]
    fin = sum(r.status == "finished" for r in full)
    return {"n": len(i["hyps"]), "confirmed": c["confirmed"], "confirmed with cost": c["confirmed with cost"],
            "refuted": c["refuted"], "inconclusive": c["inconclusive"], "rejected": c["rejected"],
            "running": c["running"] + c["accepted"], "own": len(own),
            "own_run": sum(bool(h.test_spec_hashes) and h.status != "rejected" for h in own),
            "fin": fin, "bad": len(full) - fin,
            "usd": sum((r.cost.est_usd or 0) for r in i["rows"] if r.cost)}


def gather(folders: Sequence[Path] | None = None, themes_path: Path | None = None,
           repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Everything the summary and the page print: instances, per-hypothesis records, themes."""
    folders = list(folders) if folders is not None else sorted(
        p for p in (repo_root / "results/agent").glob("explore-*") if (p / "hypotheses.jsonl").exists())
    instances = [load_instance(f) for f in folders]
    row_files = [p for p in [repo_root / x for x in BASE_ROWS] + [f / "rows.jsonl" for f in folders] if p.exists()]
    arms = explore.read_arms(row_files, repo_root)
    themes_path = themes_path or repo_root / THEMES
    themes = json.loads(themes_path.read_text())["themes"] if themes_path.exists() else []
    recs = {(i["name"], h.id): record(h, arms) for i in instances for h in i["hyps"]}
    return {"instances": instances, "arms": arms, "themes": themes, "recs": recs,
            "sources": row_files + [f / "hypotheses.jsonl" for f in folders]}


HEAD = "| " + " | ".join(COLUMNS) + " |\n" + "|---" * len(COLUMNS) + "|"


def render(instances: Sequence[dict[str, Any]], arms: Sequence[explore.Arm], themes: Sequence[dict[str, Any]],
           sources: Sequence[str]) -> str:
    recs = {(i["name"], h.id): record(h, arms) for i in instances for h in i["hyps"]}
    out = ["# What the research agents found (exploratory)", "",
           ("Built by `python -m clharness.agent.explore_summary` from the files named at the bottom. "
           "Do not edit by hand."), "",
           ("**How to read it.** Each hypothesis was written by a model before its runs, with a predicted "
           "direction and size. Code decided what could run and code gave the verdict: one new configuration "
           "at two seeds against one existing control arm, on the selection half of the eval items. "
           "\"Measured effect\" is positive when the tested configuration forgot less than its control. "
           "\"Core lost\" is the share of the 58 fragile-core questions wrong at the end of the run. "
           "\"Legal ratio\" is held-out legal perplexity after legal training over the base model's: lower "
           "means it learned more. *Confirmed with cost* means the prediction held but the configuration "
           "learned legal worse than its control by more than 5%. Two seeds is thin: read every verdict as a "
           "lead, not a result. None of this is part of the pre-registered analysis."), "",
           "## Totals", "",
           ("| Instance | Hypotheses | Confirmed | Confirmed with cost | Refuted | Inconclusive | Refused by the gate | "
           "Still running | Agent-written techniques proposed / run | Runs finished / failed | Compute | Model calls |"),
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    total: Counter[str] = Counter()
    for i in instances:
        t = totals(i)
        out.append(f"| {i['name']} | {t['n']} | {t['confirmed']} | {t['confirmed with cost']} | {t['refuted']} "
                   f"| {t['inconclusive']} | {t['rejected']} | {t['running']} | {t['own']} / {t['own_run']} "
                   f"| {t['fin']} / {t['bad']} | ${t['usd']:.0f} | "
                   + (f"${i['model_usd']:.2f} ({i['model_calls']}) |" if i["calls_logged"] else "not committed |"))
        total.update(t)
        total.update({"musd": i["model_usd"], "calls": i["model_calls"]})
    out += [(f"| **all** | {total['n']} | {total['confirmed']} | {total['confirmed with cost']} | {total['refuted']} | "
            f"{total['inconclusive']} | {total['rejected']} | {total['running']} | "
            f"{total['own']} / {total['own_run']} | {total['fin']} / {total['bad']} | ${total['usd']:.0f} | "
            f"${total['musd']:.2f} ({total['calls']}), where committed |"), "",
            "Compute is the rows' own estimate (`cost.est_usd`), probes included. Runs counts full runs only."]

    used: set[tuple[str, str]] = set()
    if themes:
        out += ["", "## Findings by theme", "",
                ("The grouping and the sentences under each heading were written by a person "
                f"(`{THEMES}`); the tables under them are computed.")]
    for theme in themes:
        keys = [tuple(x.split(":", 1)) for x in theme["hypotheses"]]
        present = [k for k in keys if k in recs]
        if not present:
            continue
        out += ["", f"### {theme['title']}", "", theme["text"], "", HEAD]
        out += [_row(k[0], recs[k]) for k in present]
        used.update(present)

    rest = [k for k, r in recs.items() if k not in used and status(r["h"]) in DECIDED_LABELS]
    if rest:
        out += ["", "## Other decided hypotheses", "", HEAD] + [_row(k[0], recs[k]) for k in rest]

    out += ["", "## Techniques the agents wrote themselves", "",
            ("Each is one hook class carried in the run spec. It had to pass a syntax allow-list, a tiny-model "
            "training check on the laptop and a four-step run on a GPU before its two full runs."), "",
            "| Hypothesis | Technique | Checks passed | Outcome | Measured effect | Verdict |", "|---|---|---|---|---|---|"]
    for (inst, _), r in recs.items():
        h = r["h"]
        if h.new_technique:
            measured = "n/a" if h.effect_points is None else f"{h.effect_points:+.1f} (needs {h.threshold_points:.1f})"
            why = f": {_cell(h.reason, 120)}" if h.status == "rejected" else ""
            out.append(f"| {inst} {h.id} | {_cell(h.new_technique.title or h.question, 110)} | "
                       f"{len(h.checks)} of 3 | {UNITS[h.outcome]} | {measured} | {status(h)}{why} |")

    out += ["", "## Every hypothesis", "", HEAD] + [_row(k[0], r) for k, r in recs.items()]
    out += ["", "## Built from", ""] + [f"- `{s}`" for s in sources]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="*", type=Path, help="explore output folders (default: results/agent/explore-*)")
    parser.add_argument("--themes", type=Path, default=REPO_ROOT / THEMES)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/agent/explore_summary.md")
    args = parser.parse_args(argv)
    g = gather(args.folders or None, args.themes)
    instances, arms, themes = g["instances"], g["arms"], g["themes"]
    rel = [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else p.name for p in g["sources"]]
    args.out.write_text(render(instances, arms, themes, rel))
    print(f"wrote {args.out} ({sum(len(i['hyps']) for i in instances)} hypotheses, {len(instances)} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
