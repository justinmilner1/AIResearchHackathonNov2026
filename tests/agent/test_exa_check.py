"""clh exa-check: the explore loop's prior-work Exa call run live on one claim. FakeSearch
stands in for the API; no network."""

import json

from clharness.agent import exa_check
from clharness.agent import explore as ex
from clharness.agent.contracts import Source
from clharness.agent.search import FakeSearch

CLAIM = "replaying 5% of earlier tokens forgets less than a lower learning rate"


def _sources() -> list[Source]:
    now = "2026-11-15T00:00:00+00:00"
    return [
        Source(id="src-1", query="q", url="https://arxiv.org/abs/2608.00007",
               title="Replay \x1b[31m<img> [paper]\x07", snippet="", published="2026-08-01",
               retrieved_at=now),
        Source(id="src-2", query="q", url="https://example.org/lr-sweeps",
               title="Learning-rate sweeps", snippet="", published=None, retrieved_at=now),
    ]


def _fake() -> FakeSearch:
    """A FakeSearch whose `answer` returns citations for the claim's prior-work question.
    The question is whatever explore.ask_prior_work sends, so prime it once to record it."""
    fake = FakeSearch()
    ex.ask_prior_work(CLAIM, fake.answer)
    fake.results[fake.calls[-1][0]] = _sources()
    return fake


def test_prints_the_question_answer_and_numbered_citations(capsys):
    fake = _fake()
    assert exa_check.main([CLAIM], answer=fake.answer) == 0
    out = capsys.readouterr().out
    assert "question sent to Exa" in out and "Has this already been tested" in out
    assert CLAIM in out
    assert "fake answer: tested once, mixed results" in out
    assert "1. Replay" in out and "2. Learning-rate sweeps" in out
    assert "2026-08-01" in out
    assert "https://arxiv.org/abs/2608.00007" in out and "https://example.org/lr-sweeps" in out
    assert [call[1] for call in fake.calls] == ["answer", "answer"]


def test_untrusted_web_text_cannot_drive_the_terminal(capsys):
    """Web text goes to a terminal: escape sequences and control characters are removed,
    brackets stay readable (the citation markers [1][2] in an answer)."""
    assert exa_check.main([CLAIM], answer=_fake().answer) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert "Replay <img> [paper]" in out
    assert "&#91;" not in out and "&lt;" not in out


def test_a_failing_search_prints_the_error_and_exits_1(capsys):
    assert exa_check.main([CLAIM], answer=FakeSearch(fail=True).answer) == 1
    err = capsys.readouterr().err
    assert "fake search failure" in err


def test_json_round_trips_into_prior_work(capsys):
    fake = _fake()
    assert exa_check.main([CLAIM, "--json"], answer=fake.answer) == 0
    pw = ex.PriorWork.model_validate(json.loads(capsys.readouterr().out))
    assert pw.question.startswith("Has this already been tested") and CLAIM in pw.question
    assert pw.answer == "fake answer: tested once, mixed results"
    assert [c.url for c in pw.citations] == ["https://arxiv.org/abs/2608.00007",
                                           "https://example.org/lr-sweeps"]
    assert pw.citations[0].published == "2026-08-01"
    assert pw.error is None
