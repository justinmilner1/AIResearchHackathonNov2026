"""The Exa evidence trail: committed agent files in, one readable Markdown page out."""

import json
from pathlib import Path

from clharness.agent import explore as ex
from clharness.agent import trail
from clharness.cli import main as cli_main


def _hyp(hid: str = "h001", **kw) -> ex.Hypothesis:
    base = {"id": hid, "created_at": "2026-11-15T00:00:00+00:00", "author": "model",
            "origin": "surprise", "question": "q?",
            "claim": "replay at a low rate forgets less",
            "prediction": {"direction": "less_forgetting", "size_points": 0.5},
            "control": "no-mitigation@lr3e-05", "decision_rule": "the rule", "status": "accepted",
            "reason": "", "test_spec_hashes": ["ab12cd34" + "0" * 56],
            "control_spec_hashes": ["ff00ee11" + "0" * 56]}
    return ex.Hypothesis(**{**base, **kw})


def _explore_dir(tmp_path: Path, hypotheses=()) -> Path:
    out = tmp_path / "explore-x"
    out.mkdir()
    (out / "hypotheses.jsonl").write_text("".join(h.model_dump_json() + "\n" for h in hypotheses))
    return out


def _scout_file(tmp_path: Path) -> Path:
    cands = tmp_path / "loop-x" / "candidates"
    cands.mkdir(parents=True)
    data = {"candidates": [{"name": "Spaced Repetition", "description": "review old examples",
                            "source_ids": ["src-1"]}],
            "sources": [{"id": "src-1", "query": "q", "url": "https://arxiv.org/abs/2608.00001",
                         "title": "Spaced review <b>paper</b>", "snippet": "x",
                         "published": "2026-08-18", "retrieved_at": "2026-11-15T00:00:00+00:00"},
                        {"id": "src-2", "query": "q", "url": "https://arxiv.org/abs/2608.00002",
                         "title": "Uncited", "snippet": "y", "published": "2026-08-19",
                         "retrieved_at": "2026-11-15T00:00:00+00:00"}],
            "searches": 3}
    path = cands / "2026-11-15-0-claims-1-candidates.json"
    path.write_text(json.dumps(data))
    return tmp_path / "loop-x"


def test_prior_work_citations_appear_and_untrusted_text_is_escaped(tmp_path):
    pw = ex.PriorWork(
        question="already tested?", answer="Mostly <script>alert(1)</script> not.",
        citations=[ex.Citation(title="Paper A <img>", url="https://arxiv.org/abs/2608.00007",
                               published="2026-08-01"),
                   ex.Citation(title="Paper B", url="javascript:alert(1)")])
    out = _explore_dir(tmp_path, [_hyp(prior_work=pw)])
    page = trail.build_trail([out])
    assert "https://arxiv.org/abs/2608.00007" in page and "2026-08-01" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<img>" not in page and "&lt;img&gt;" in page
    assert "](" not in page  # no injected markdown links: the javascript: URL is shown as text
    assert "javascript:" in page  # but the recorded URL is still visible


def test_no_prior_work_says_not_asked(tmp_path):
    page = trail.build_trail([_explore_dir(tmp_path, [_hyp()])])
    assert "not asked" in page


def test_a_hypothesis_citing_a_chosen_search_url_is_linked_to_it(tmp_path):
    url = "https://arxiv.org/abs/2608.17530"
    h = _hyp("h002", origin="literature", source=f"Paper {url}")
    out = _explore_dir(tmp_path, [h])
    (out / "decisions.jsonl").write_text(
        json.dumps({"key": "explore/next_search#0", "data": {"query": "spaced replay",
                                                             "after": ["h001"], "asked_by": "model"}}) + "\n"
        + json.dumps({"key": "explore/search#1", "data": {"query": "spaced replay",
                                                          "urls": [url]}}) + "\n")
    (out / "sources.jsonl").write_text(json.dumps(
        {"query": "spaced replay", "url": url, "title": "Spaced replay paper",
         "snippet": "s", "published": "2026-08-18", "retrieved_at": "t", "id": "src-9"}) + "\n")
    page = trail.build_trail([out])
    assert "spaced replay" in page and "after h001" in page
    assert "Spaced replay paper" in page
    assert "cited by hypothesis h002" in page


def test_scout_candidates_render_their_papers(tmp_path):
    loop = _scout_file(tmp_path)
    page = trail.build_trail([loop])
    assert "Spaced Repetition" in page and "review old examples" in page
    assert "Spaced review" in page and "2026-08-18" in page
    assert "https://arxiv.org/abs/2608.00001" in page
    assert "2608.00002" not in page  # only the papers the candidate cites
    assert "<b>" not in page and "&lt;b&gt;" in page


def test_results_agent_discovers_explore_subdirs(tmp_path):
    _explore_dir(tmp_path, [_hyp()])
    page = trail.build_trail([tmp_path])
    assert "replay at a low rate forgets less" in page
    assert "h001" in page and "ab12cd34" in page


def test_output_is_byte_identical_on_rerun(tmp_path):
    out = _explore_dir(tmp_path, [_hyp()])
    loop = _scout_file(tmp_path)
    first = trail.build_trail([out, loop])
    assert trail.build_trail([out, loop]) == first


def test_cli_writes_the_default_page(tmp_path):
    out = _explore_dir(tmp_path, [_hyp()])
    target = tmp_path / "trail.md"
    assert cli_main(["exa-trail", str(out), "--out", str(target)]) == 0
    assert target.read_text() == trail.build_trail([out])


def test_a_costly_or_unreplicated_verdict_keeps_its_evidence_line():
    for status in ("confirmed_with_cost", "not_replicated"):
        h = _hyp(status=status, effect_points=29.31, threshold_points=12.74,
                 reason="effect +29.31; learns legal worse").model_dump(mode="json")
        lines = trail._hypothesis_lines(h, {})
        verdict = [x for x in lines if x.startswith("- **Verdict:**")]
        assert verdict and "+29.31" in verdict[0] and "learns legal worse" in verdict[0], status
