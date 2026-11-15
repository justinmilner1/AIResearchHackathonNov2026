"""The Exa evidence trail: one readable Markdown page from the agent's output files.

    clh exa-trail results/agent/explore-himani results/agent/loop-himani
    clh exa-trail DIR... [--out FILE]           # default out: results/agent/exa_trail.md

The Exa sponsor challenge asks two things of the entry: point to where Exa is called, and show
what the system found through Exa and how it used it. The calls are all in
`src/clharness/agent/search.py`:

  * `ExaSearch.search` — the Scout's queries (agents.py `LLMScout.gather`), the explore loop's
    `--search` argument, and the search the model itself chooses after a cycle
    (explore.py `run_next_search` -> `next_search.json` -> `sources.jsonl`)
  * `ExaSearch.answer` — the prior-work check on every accepted hypothesis, asked before any
    compute is spent (explore.py `ask_prior_work` -> `Hypothesis.prior_work`)

This module reads the committed evidence only — each explore directory's `hypotheses.jsonl`,
`decisions.jsonl`, `sources.jsonl`, `next_search.json` and `rows.jsonl`, and the Scout's
`candidates/*.json` result files — and writes the trail as Markdown. It makes no network calls
and no model calls, writes no timestamp of its own and sorts everything it can, so the same
input files give a byte-identical page. Everything quoted from the web is untrusted text: it is
HTML/Markdown-escaped, kept to a character limit, and never turned into a link other than the
URL the file recorded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import read_jsonl
from .explore import Hypothesis

# The two functions that touch Exa, with the places the agent calls them from. Static: the
# point of the page is that a reader can check this list against the code.
EXA_CALL_SITES = (
    ("src/clharness/agent/search.py", "ExaSearch.search",
     ("one web search with highlights; called by the Scout (agents.py LLMScout.gather, the "
      "queries in agents.SCOUT_QUERIES), by the explore loop's --search argument, and by "
      "run_next_search for the search the model chose itself")),
    ("src/clharness/agent/search.py", "ExaSearch.answer",
     ("one grounded answer with citations per accepted hypothesis, before compute is spent "
      "(explore.py ask_prior_work -> Hypothesis.prior_work); informational, never a gate")),
)

_SCOUT_KEYS = ("candidates", "claims", "sources")
_URL_OK = re.compile(r"^https?://[^\s()\[\]<>'\"]+$", re.IGNORECASE)
INLINE_LIMIT = 500
ANSWER_LIMIT = 1200
DESCRIPTION_LIMIT = 400


def _esc(text: Any, limit: int = INLINE_LIMIT) -> str:
    """Untrusted text for one line of Markdown: no raw `<`/`>` (no HTML), no `[`/`]` (a title
    cannot smuggle in a link target), truncated."""
    flat = " ".join(str(text).split())
    s = flat[:limit] + ("…" if len(flat) > limit else "")
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace("[", "&#91;").replace("]", "&#93;"))


def _link(url: Any) -> str:
    """A link to the recorded URL, or the URL as plain text when it is not plain http(s)."""
    url = str(url).strip()
    return f"<{url}>" if _URL_OK.match(url) else f"`{_esc(url)}`"


def _paper(source: dict[str, Any]) -> str:
    """One cited source line: title, date and URL."""
    title = _esc(source.get("title") or "(no title)", 200)
    published = source.get("published")
    when = f" ({_esc(str(published)[:10])})" if published else ""
    return f"{title}{when} — {_link(source.get('url', ''))}"


# --------------------------------------------------------------------------------------------
# Reading: explore directories and Scout result files
# --------------------------------------------------------------------------------------------


def _hypotheses(path: Path) -> list[dict[str, Any]]:
    """Hypotheses in file order. A line written by another version of the model (extra fields)
    is kept as raw JSON rather than dropped."""
    out = []
    for record in read_jsonl(path).records:
        try:
            out.append(Hypothesis.model_validate(record).model_dump(mode="json"))
        except ValidationError:
            out.append(record)
    return out


def _rows_by_hash(path: Path) -> dict[str, dict[str, Any]]:
    """spec hash -> {name, seed, status} of the explore dir's own rows, for the run links."""
    out = {}
    for record in read_jsonl(path).records:
        spec_hash, spec, status = record.get("spec_hash"), record.get("spec") or {}, record.get("status")
        if spec_hash:
            out[spec_hash] = {"name": spec.get("name", ""), "seed": spec.get("seed"),
                              "status": status}
    return out


def _scout_file(path: Path) -> dict[str, Any] | None:
    """A Scout result JSON, or None. Tolerant: checked by shape, not by model version."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and any(k in data for k in _SCOUT_KEYS):
        return data
    return None


def _is_explore_dir(path: Path) -> bool:
    return (path / "hypotheses.jsonl").exists()


def _scout_files_in(path: Path) -> list[Path]:
    """Scout result JSONs in a directory: `candidates/*.json`, or `*.json` of the directory
    itself when the directory is named `candidates`. One level only."""
    found: list[Path] = []
    cands = path / "candidates"
    if cands.is_dir():
        found += sorted(cands.glob("*.json"))
    if path.name == "candidates" and path.is_dir():
        found += sorted(p for p in path.glob("*.json") if p.parent == path)
    return found


def discover(paths: Sequence[str | Path]) -> tuple[list[Path], list[Path]]:
    """(explore dirs, scout result files) found under the arguments. A passed directory that
    holds neither directly is searched one level down (`results/agent` finds `explore-*` and
    `*/candidates`)."""
    explore_dirs: list[Path] = []
    scout_files: list[Path] = []

    def add_dir(d: Path) -> None:
        if _is_explore_dir(d) and d not in explore_dirs:
            explore_dirs.append(d)
        for f in _scout_files_in(d):
            if f not in scout_files:
                scout_files.append(f)

    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path not in scout_files:
                scout_files.append(path)
            continue
        if not path.is_dir():
            continue
        add_dir(path)
        for sub in sorted(path.iterdir()):
            if sub.is_dir():
                add_dir(sub)
    return explore_dirs, scout_files


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _prior_work_lines(pw: dict[str, Any] | None) -> list[str]:
    """The Exa `answer` call for one hypothesis, or why there is none."""
    if not isinstance(pw, dict):
        return ["- **Prior work (Exa `answer`, before launch):** not asked"]
    if pw.get("error"):
        return [(f"- **Prior work (Exa `answer`, before launch):** asked, Exa error: "
                 f"{_esc(pw['error'], 300)}")]
    lines = [f"- **Prior work (Exa `answer`, before launch):** {_esc(pw.get('answer', ''), ANSWER_LIMIT)}"]
    for c in pw.get("citations") or []:
        lines.append(f"  - {_paper(c)}")
    return lines


def _hypothesis_lines(h: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[str]:
    status = _esc(h.get("status", ""))
    lines = [f"### {_esc(h.get('id', '?'))} — {status}",
             "",
             f"- **Claim:** {_esc(h.get('claim', ''))}"]
    origin = _esc(h.get("origin", ""))
    source = _esc(h.get("source") or "", 600)
    lines.append(f"- **Origin:** {origin}" + (f" — {source}" if source else ""))
    own = h.get("new_technique")
    if isinstance(own, dict):
        checks = ", ".join(_esc(c, 60) for c in h.get("checks") or []) or "none recorded"
        lines.append(f"- **Agent-written technique:** {_esc(own.get('title', ''), 120)} "
                     f"(checks passed: {checks})")
    gate = "accepted" if h.get("test_spec_hashes") else status
    reason = _esc(h.get("reason", ""), 600)
    lines.append(f"- **Gate:** {gate}" + (f" — {reason}" if reason and status == "rejected" else ""))
    lines += _prior_work_lines(h.get("prior_work"))
    if status in ("confirmed", "confirmed_with_cost", "refuted", "inconclusive", "not_replicated"):
        verdict = f"**{status}**"
        effect, threshold = h.get("effect_points"), h.get("threshold_points")
        if effect is not None and threshold is not None:
            verdict += f" — effect {effect:+.2f} points, threshold {threshold:.2f}"
        verdict += (f" ({reason})" if reason else "")
        lines.append(f"- **Verdict:** {verdict}")
    elif status == "running":
        lines.append("- **Verdict:** running — rows not all in yet")
    runs = []
    for spec_hash in h.get("test_spec_hashes") or []:
        row = rows.get(spec_hash)
        label = row["name"] if row else "no row yet"
        detail = f" {row['status']}" if row else ""
        runs.append(f"`{spec_hash[:12]}` {label}{detail}")
    for spec_hash in h.get("control_spec_hashes") or []:
        row = rows.get(spec_hash)
        label = f"control: {row['name']}" if row else "control row"
        detail = f" {row['status']}" if row else ""
        runs.append(f"`{spec_hash[:12]}` {label}{detail}")
    if runs:
        lines.append(f"- **Runs:** {'; '.join(runs)}")
    lines.append("")
    return lines


def _searches(dir_path: Path, hypotheses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The searches the model chose for the next cycle: next_search.json and the
    explore/next_search decision-log entries (the ask), joined with the explore/search
    entries and the sources.jsonl lines (the run). One entry per ask; a run the query
    matches fills in the URLs found."""
    asked: list[dict[str, Any]] = []
    pending = dir_path / "next_search.json"
    if pending.exists():  # the live request: the file knows whether it was run
        try:
            req = json.loads(pending.read_text(encoding="utf-8"))
        except ValueError:
            req = {}
        if req.get("query"):
            asked.append({"query": req["query"], "asked_by": req.get("asked_by"),
                          "after": req.get("after") or [], "used": bool(req.get("used")),
                          "sources": []})
    ran: list[dict[str, Any]] = []
    for record in read_jsonl(dir_path / "decisions.jsonl").records:
        key, data = str(record.get("key", "")), record.get("data") or {}
        if key.startswith("explore/next_search") and data.get("query"):
            if all(s["query"] != data["query"] for s in asked):
                asked.append({"query": data["query"], "asked_by": data.get("asked_by"),
                              "after": data.get("after") or [], "used": False, "sources": []})
        elif key.startswith("explore/search") and data.get("query"):
            urls = [str(u) for u in data.get("urls") or []]
            ran.append({"query": data["query"], "asked_by": data.get("asked_by"),
                        "sources": [{"url": u} for u in urls]})
    searches = list(asked)
    unmatched_runs = list(ran)
    for search in searches:  # a run joins the ask it answered (first unmatched with the query)
        for i, run in enumerate(unmatched_runs):
            if run["query"] == search["query"]:
                search["used"], search["sources"] = True, run["sources"]
                unmatched_runs.pop(i)
                break
    for run in unmatched_runs:  # ran without a recorded ask (older log shape)
        searches.append({"query": run["query"], "asked_by": run.get("asked_by"), "after": [],
                         "used": True, "sources": run["sources"]})
    by_query: dict[str, list[dict[str, Any]]] = {}
    for record in read_jsonl(dir_path / "sources.jsonl").records:
        by_query.setdefault(str(record.get("query", "")), []).append(record)
    for search in searches:  # the fetched sources carry titles the decision log's URLs do not
        richer = by_query.pop(search["query"], [])
        if richer:
            search["sources"] = richer
    for query, records in sorted(by_query.items()):  # a --search argument run: no decision entry
        searches.append({"query": query, "asked_by": "--search argument", "after": [],
                         "used": True, "sources": records})
    # Which hypotheses cite a URL one of these searches returned (URL substring of `source`).
    for search in searches:
        found_urls = {str(s.get("url")) for s in search["sources"]}
        search["cited_by"] = [str(h.get("id")) for h in hypotheses
                              if any(u and u in str(h.get("source") or "") for u in found_urls)]
    return searches


def _search_lines(searches: Sequence[dict[str, Any]]) -> list[str]:
    if not searches:
        return []
    lines = ["## Searches the agent chose itself", ""]
    for search in searches:
        after = ", ".join(_esc(a, 20) for a in search["after"]) or "not recorded"
        asked_by = _esc(search.get("asked_by") or "model")
        state = "run through Exa" if search["used"] else "not run yet"
        lines.append(f"- **{_esc(search['query'], 300)}** — asked by {asked_by}, after {after}; {state}")
        for s in search["sources"]:
            lines.append(f"  - {_paper(s)}")
        if search.get("cited_by"):
            lines.append(f"  - cited by hypothesis {', '.join(_esc(i, 20) for i in search['cited_by'])}")
    lines.append("")
    return lines


def _scout_lines(files: Sequence[Path]) -> tuple[list[str], dict[str, int]]:
    lines: list[str] = []
    stats = {"searches": 0, "sources": 0, "candidates": 0}
    for path in files:
        data = _scout_file(path)
        if data is None:
            lines.append(f"### `{path}`", "", "not a Scout result file — skipped", "")
            continue
        stats["searches"] += int(data.get("searches") or 0)
        by_id = {s.get("id"): s for s in data.get("sources") or []}
        stats["sources"] += len(by_id)
        lines.append(f"### `{path.name}`")
        lines.append("")
        candidates = data.get("candidates") or []
        stats["candidates"] += len(candidates)
        if data.get("skipped_reason"):
            lines.append(f"Scout skipped: {_esc(data['skipped_reason'], 300)}")
            lines.append("")
        for cand in candidates:
            lines.append(f"- **{_esc(cand.get('name', ''), 120)}** — "
                         f"{_esc(cand.get('description', ''), DESCRIPTION_LIMIT)}")
            for sid in cand.get("source_ids") or []:
                source = by_id.get(sid)
                if source is not None:
                    lines.append(f"  - {_paper(source)}")
        for card in data.get("claims") or []:
            lines.append(f"- claim card `{_esc(card.get('id', ''), 20)}`: {_esc(card.get('claim', ''), 300)}")
            for sid in card.get("source_ids") or []:
                source = by_id.get(sid)
                if source is not None:
                    lines.append(f"  - {_paper(source)}")
        lines.append("")
    return lines, stats


def build_trail(paths: Sequence[str | Path]) -> str:
    """The Markdown page for the given directories/files. Deterministic for the same files."""
    explore_dirs, scout_files = discover(paths)

    per_dir: list[tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]] = []
    totals = {"answers": 0, "decision_searches": 0, "source_records": 0, "urls": set()}
    for d in explore_dirs:
        hypotheses = _hypotheses(d / "hypotheses.jsonl")
        # Controls live in the shared row files: results/*.jsonl, then results/agent/rows.jsonl;
        # the explore dir's own rows file wins where several name a spec.
        rows: dict[str, dict[str, Any]] = {}
        for f in sorted(d.parent.parent.glob("*.jsonl")) if d.parent.parent.is_dir() else []:
            rows.update(_rows_by_hash(f))
        rows = {**rows, **_rows_by_hash(d.parent / "rows.jsonl"),
                **_rows_by_hash(d / "rows.jsonl")}
        searches = _searches(d, hypotheses)
        per_dir.append((d, hypotheses, rows, searches))
        totals["answers"] += sum(1 for h in hypotheses if h.get("prior_work") is not None)
        totals["decision_searches"] += sum(1 for s in searches if s["used"])
        for search in searches:
            for s in search["sources"]:
                totals["source_records"] += 1
                if s.get("url"):
                    totals["urls"].add(str(s["url"]))
        for h in hypotheses:
            for c in (h.get("prior_work") or {}).get("citations") or []:
                totals["source_records"] += 1
                if c.get("url"):
                    totals["urls"].add(str(c["url"]))

    scout_lines, scout_stats = _scout_lines(scout_files)
    totals["source_records"] += scout_stats["sources"]

    lines: list[str] = ["# Exa evidence trail", ""]
    lines.append("Where Exa is called (the only module that holds the key):")
    for path, func, what in EXA_CALL_SITES:
        lines.append(f"- `{path}` `{func}` — {what}")
    calls = scout_stats["searches"] + totals["decision_searches"] + totals["answers"]
    lines += [
        "",
        (f"Exa calls on record: **{calls}** "
         f"({scout_stats['searches']} Scout searches, {totals['decision_searches']} explore "
         f"searches, {totals['answers']} prior-work answers). Source records: "
         f"{totals['source_records']}, unique URLs: {len(totals['urls'])}."),
        "",
        ("Every title, snippet and answer below is text fetched from the web: it is quoted as "
         "data, escaped, and truncated. The only links are the URLs the files recorded."),
        "",
        "## Scout: technique candidates found through Exa search",
        "",
    ]
    lines += scout_lines or ["No Scout result files found.", ""]
    for d, hypotheses, rows, searches in per_dir:
        lines += [f"## Hypotheses — `{d}`", ""]
        if not hypotheses:
            lines.append("None on file.")
            lines.append("")
        for h in hypotheses:
            lines += _hypothesis_lines(h, rows)
        lines += _search_lines(searches)
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clh exa-trail", description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="*", type=Path,
                        help="explore output dirs, Conductor/loop dirs (their candidates/ are "
                             "read), candidates dirs, or Scout result JSON files. Default: "
                             "results/agent")
    parser.add_argument("--out", type=Path, default=Path("results/agent/exa_trail.md"),
                        help="where the Markdown goes (default: results/agent/exa_trail.md); "
                             "'-' writes to stdout")
    args = parser.parse_args(argv)
    dirs = args.dirs or [Path("results/agent")]
    page = build_trail(dirs)
    if str(args.out) == "-":
        sys.stdout.write(page)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
