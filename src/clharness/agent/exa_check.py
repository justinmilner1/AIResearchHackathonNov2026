"""The explore loop's prior-work check, run live on one claim: the Exa demo command.

    clh exa-check "replaying 5% of earlier tokens forgets less than a lower learning rate"
    clh exa-check "<claim>" --json          # the PriorWork record as JSON

This is the same call the explore loop makes on every hypothesis the gate accepts, before
any compute is spent (explore.py `ask_prior_work` -> `ExaSearch.answer`): one Exa `answer`
call, a short grounded answer with its citations. The loop stores the record on the
hypothesis as `prior_work`; here it is printed so a person can watch the check happen.

Everything printed came back from the web and goes to a terminal: control characters and
ANSI escape sequences (which could rewrite the screen) are removed, whitespace is collapsed,
text is truncated, and a URL is shown only when it is plain http(s). Brackets and `<`/`>` are
left readable, since a terminal does not render them (the Markdown trail in trail.py escapes
them instead). The key is
read from `EXA_API_KEY` inside search.py and is never printed. A failing search prints the
error and exits 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

from .explore import Citation, PriorWork, ask_prior_work

# Untrusted-text rules for a terminal: strip ANSI escape sequences and other control
# characters, collapse whitespace, truncate; a URL is shown only when it is plain http(s).
_URL_OK = re.compile(r"^https?://[^\s()\[\]<>'\"]+$", re.IGNORECASE)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
INLINE_LIMIT = 500
ANSWER_LIMIT = 1200


def _esc(text: Any, limit: int = INLINE_LIMIT) -> str:
    clean = _CONTROL.sub("", _ANSI.sub("", str(text)))
    flat = " ".join(clean.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _link(url: Any) -> str:
    url = str(url).strip()
    return url if _URL_OK.match(url) else f"(unsafe link omitted: {_esc(url, 80)})"


def _citation(c: Citation) -> str:
    """One citation: title, date and URL."""
    title = _esc(c.title or "(no title)", 200)
    when = f" ({_esc(str(c.published)[:10])})" if c.published else ""
    return f"{title}{when} — {_link(c.url)}"


def render(pw: PriorWork, claim: str) -> str:
    """The prior-work check as text: the question sent, the answer, numbered citations."""
    lines = [f"claim: {_esc(claim)}",
             "",
             f"question sent to Exa (ExaSearch.answer): {_esc(pw.question)}",
             "",
             f"answer: {_esc(pw.answer, ANSWER_LIMIT)}",
             ""]
    if pw.citations:
        lines.append("citations:")
        lines += [f"  {i}. {_citation(c)}" for i, c in enumerate(pw.citations, 1)]
    else:
        lines.append("citations: none")
    return "\n".join(lines)


def run(claim: str,
        answer: Callable[[str], tuple[str, Sequence[Any]]] | None = None,
        json_out: bool = False) -> int:
    """Ask Exa whether the claim was already tested. `answer` is injectable for tests
    (`search.FakeSearch().answer`); the default is the real `ExaSearch().answer`."""
    if answer is None:
        from .search import ExaSearch

        answer = ExaSearch().answer
    pw = ask_prior_work(claim, answer)
    if pw.error:
        print(f"error: {pw.error}", file=sys.stderr)
        return 1
    if json_out:
        print(pw.model_dump_json(indent=2))
    else:
        print(render(pw, claim))
    return 0


def main(argv: list[str] | None = None,
         answer: Callable[[str], tuple[str, Sequence[Any]]] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clh exa-check", description=__doc__.splitlines()[0])
    parser.add_argument("claim", help="the hypothesis claim to check against published work")
    parser.add_argument("--json", action="store_true",
                        help="print the PriorWork record as JSON instead of text")
    args = parser.parse_args(argv)
    return run(args.claim, answer=answer, json_out=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
