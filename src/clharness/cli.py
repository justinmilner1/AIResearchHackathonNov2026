"""Command-line entry point.

  clh env                                   detected environment
  clh build-data --set smoke|real [--domain D]   pull sources, tokenize, write manifests/<set>.json
  clh build-eval-manifest CONFIG            draw + freeze lm-eval item lists for CONFIG's suite
  clh verify CONFIG                         check data + eval manifests against disk / hub
  clh sae-baseline CONFIG                   SAE reconstruction on the base model (drift baseline)
  clh train CONFIG [--name N]               run the continual fine-tuning loop
  clh compare-runs A B                      repeatability diff of two runs' metrics.jsonl
  clh analyze RUN...                        per-run analysis.json + text report (no GPU)
  clh validate-row FILE...                  check result-row JSONL files against the format
  clh spec-hash FILE...                     idempotency key of run spec JSON files
  clh fake-rows [--out F] [--seed N]        invented result rows with planted effects
  clh export-schema [--out DIR]             JSON Schemas of the run spec and result row
  clh run SPEC... [--modal] [--out F]       run specs, append one result row each
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from clharness.config import REPO_ROOT


def _eval_manifest_path(cfg) -> Path:
    return REPO_ROOT / "manifests" / f"eval-{cfg.eval.suite}.json"


def cmd_env(_args) -> int:
    from clharness.env import describe_env

    for k, v in describe_env().items():
        print(f"{k}: {v}")
    return 0


def cmd_build_data(args) -> int:
    from clharness.data.build import build_smoke_set
    from clharness.model import load_tokenizer

    tok = load_tokenizer(args.model_id)
    t0 = time.time()
    if args.set == "real":
        from clharness.data.build import REAL_BUDGETS, build_real_domain
        from clharness.data.sources import REAL_SOURCES

        domains = args.domain or list(REAL_SOURCES)
        for d in domains:
            print(f"building real/{d} from {REAL_SOURCES[d].name} ...", flush=True)
            build_real_domain("real", d, REAL_SOURCES[d], REAL_BUDGETS[d], tok, seed=args.seed, seq_len=args.seq_len)
        print(f"manifests/real.json updated in {time.time() - t0:.0f}s")
        return 0
    if args.set != "smoke":
        print("unknown set (smoke | real)", file=sys.stderr)
        return 2
    m = build_smoke_set(tok, seed=args.seed, seq_len=args.seq_len)
    for d, e in m["domains"].items():
        print(d, {s: (v["n_docs"], v["n_tokens"]) for s, v in e["splits"].items()})
    print(f"wrote manifests/smoke.json in {time.time() - t0:.0f}s")
    return 0


def cmd_build_eval_manifest(args) -> int:
    from clharness.config import load_run_config
    from clharness.evals.lm_eval_runner import build_eval_manifest

    from clharness.config import load_baseline_config

    try:
        cfg = load_run_config(args.config)
    except Exception:  # noqa: BLE001 — eval-only configs (model + eval) are accepted too
        cfg = load_baseline_config(args.config)
    m = build_eval_manifest(cfg.eval, cfg.eval.suite)
    for t, e in m["tasks"].items():
        print(f"{t}: {e['n_items']}/{e['n_total_docs']} items, tier {e['tier']}")
    print(f"wrote {_eval_manifest_path(cfg).relative_to(REPO_ROOT)}")
    return 0


def cmd_verify(args) -> int:
    from clharness.config import load_run_config
    from clharness.data.manifest import ManifestMismatch, read_manifest, verify_manifest
    from clharness.evals.lm_eval_runner import verify_eval_manifest

    cfg = load_run_config(args.config)
    ok = True
    try:
        verify_manifest(read_manifest(cfg.data.manifest))
        print(f"data manifest ok: {cfg.data.manifest}")
    except ManifestMismatch as e:
        ok = False
        print(e)
    p = _eval_manifest_path(cfg)
    if p.exists():
        problems = verify_eval_manifest(cfg.eval, read_manifest(p))
        if problems:
            ok = False
            print("eval manifest problems:\n  " + "\n  ".join(problems))
        else:
            print(f"eval manifest ok: {p.relative_to(REPO_ROOT)}")
    elif cfg.eval.lm_eval_tasks:
        ok = False
        print(f"missing eval manifest {p.relative_to(REPO_ROOT)}; run: clh build-eval-manifest {args.config}")
    return 0 if ok else 1


def cmd_sae_baseline(args) -> int:
    """Reconstruction metrics of the fixed SAEs on the *base* model over the probe set."""
    import torch

    from clharness.activations import ResidualCapture
    from clharness.config import load_run_config
    from clharness.data.loader import fixed_chunks, load_tokens
    from clharness.data.manifest import read_manifest
    from clharness.env import pick_device, seed_everything
    from clharness.metrics.sae_stats import FeatureStatsAccumulator
    from clharness.model import load_model
    from clharness.sae import SAESpec, load_sae

    from clharness.train.loop import autocast_ctx

    cfg = load_run_config(args.config)
    seed_everything(cfg.train.seed)
    dev = pick_device(cfg.train.device)
    weight_dtype = {"bf16-pure": "bfloat16", "bf16-mixed": "float32", "fp32": "float32"}[cfg.train.precision]
    lm = load_model(cfg.model.model_id, device=dev, dtype=weight_dtype)
    ac = autocast_ctx(dev, cfg.train.precision)
    man = read_manifest(cfg.data.manifest)
    layers = cfg.model.capture_layers
    saes = {}
    for sc in cfg.model.saes:  # primary SAE per layer (first listed)
        saes.setdefault(sc.layer, load_sae(SAESpec(sc.release, sc.sae_id, sc.layer), dev))
    report = {"model": cfg.model.model_id, "set": man["set"], "device": dev, "dtype": str(lm.dtype),
              "exclude_bos": cfg.model.exclude_bos_from_stats, "layers": {}}
    per_domain: dict[str, dict] = {}
    with torch.no_grad(), ResidualCapture(lm.model, layers) as cap:
        for dom, entry in man["domains"].items():
            if cfg.data.probe_set not in entry["splits"]:
                continue
            chunks = fixed_chunks(
                load_tokens(entry["splits"][cfg.data.probe_set]["tokens"]["path"]),
                cfg.data.seq_len, cfg.data.probe_max_seqs_per_domain,
            )
            accs = {L: FeatureStatsAccumulator(saes[L], L) for L in layers}
            for i in range(0, chunks.shape[0], cfg.train.probe_batch_size):
                ids = chunks[i : i + cfg.train.probe_batch_size].to(dev)
                with ac():
                    lm.model(input_ids=ids, use_cache=False)
                mask = torch.ones_like(ids, dtype=torch.bool)
                if cfg.model.exclude_bos_from_stats:
                    mask &= ids != lm.tokenizer.bos_token_id
                for L in layers:
                    accs[L].update(cap.acts[L], mask)
                cap.clear()
            per_domain[dom] = {f"L{L}": accs[L].finalize().summary() for L in layers}
    report["per_domain"] = per_domain
    for L in layers:
        s = {k: sum(d[f"L{L}"][k] for d in per_domain.values()) / len(per_domain)
             for k in ("fvu", "variance_explained", "mse", "mean_l0", "cos", "x_norm", "frac_alive")}
        report["layers"][f"L{L}"] = {"sae_id": next(x.sae_id for x in cfg.model.saes if x.layer == L), **s}
        print(f"L{L}: " + ", ".join(f"{k}={v:.4g}" for k, v in s.items()))
    out = REPO_ROOT / "results" / "baselines" / f"sae_drift_base_{man['set']}_{dev.split(':')[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


def cmd_baseline_evals(args) -> int:
    from clharness.config import load_baseline_config
    from clharness.env import pick_device, seed_everything
    from clharness.evals.baseline import run_baseline, summarize
    from clharness.model import load_model

    cfg = load_baseline_config(args.config)
    seed_everything(0)
    dev = pick_device(cfg.device)
    lm = load_model(cfg.model.model_id, device=dev, dtype=cfg.model.dtype)
    t0 = time.time()
    res = run_baseline(cfg, dev, lm.model, lm.tokenizer, limit_items=args.limit_items)
    table = summarize(res)
    print("\n" + table)
    if not args.limit_items:
        md = REPO_ROOT / "results" / "baselines" / f"evals_{cfg.name}_{dev.split(':')[0]}.md"
        md.write_text(f"# Base-model eval baseline: {cfg.name}\n\nmodel: {cfg.model.model_id}, device: {dev}\n\n{table}\n")
        print(f"wrote {md.relative_to(REPO_ROOT)}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


def cmd_compare_runs(args) -> int:
    """Repeatability check: compare every logged scalar between two runs step by step."""
    import math

    def load(d: Path) -> dict[tuple[int, str], float]:
        out = {}
        for line in open(Path(d) / "metrics.jsonl"):
            r = json.loads(line)
            for k, v in r.items():
                if k != "step" and isinstance(v, (int, float)) and "wall" not in k and "time" not in k:
                    out[(r["step"], k)] = float(v)
        return out

    a, b = load(Path(args.run_a)), load(Path(args.run_b))
    keys = sorted(set(a) & set(b))
    missing = len(set(a) ^ set(b))
    exact = sum(1 for k in keys if a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])))
    worst = sorted(keys, key=lambda k: -abs(a[k] - b[k]) / (abs(a[k]) + 1e-12))[:10]
    print(f"{len(keys)} shared scalars, {exact} bitwise equal ({100*exact/max(1,len(keys)):.1f}%), {missing} unmatched keys")
    for k in worst:
        rel = abs(a[k] - b[k]) / (abs(a[k]) + 1e-12)
        if rel > 0:
            print(f"  step {k[0]:>6} {k[1]:45s} {a[k]:.6g} vs {b[k]:.6g}  rel {rel:.2e}")
    return 0 if exact == len(keys) else 1


def cmd_analyze(args) -> int:
    from clharness.analyze import analyze_run, report

    for run in args.runs:
        a = analyze_run(run, post_adapt_ckpt=args.post_adapt_ckpt)
        print(report(a))
        print(f"wrote results/runs/{run}/analysis.json\n")
    return 0


def cmd_items(args) -> int:
    from clharness.items import build_items, summary

    for run in args.runs:
        t = build_items(run, with_text=not args.no_text, persist=args.persist)
        print(summary(t))
        print(f"wrote results/runs/{run}/items.json\n")
    return 0


def cmd_report(args) -> int:
    from clharness.report import build_report

    for run in args.runs:
        out = build_report(run)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


def cmd_generate(args) -> int:
    from clharness.generate import generate_panel

    path = generate_panel(args.run, args.ckpts, max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


def cmd_demo(args) -> int:
    from clharness.demo import build_demo

    out = build_demo(args.run)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


def cmd_features(args) -> int:
    """Domain-resolved SAE feature analysis (see clharness.features)."""
    from clharness.features import domain_features, domain_pass

    if args.do_pass:
        path = domain_pass(args.run, args.ckpts, max_seqs=args.max_seqs, device=args.device)
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    d = domain_features(args.run)
    for L, v in d["layers"].items():
        print(f"L{L}: alive {v['n_alive']}, sets " + ", ".join(f"{k}:{n}" for k, n in v["set_sizes"].items()))
        for c, st in v["stage"].items():
            print(f"  {st['label']:16s} " + " | ".join(f"{d_}: med {x['median_ratio']:.2f} dec {x['frac_decayed'] * 100:.0f}% grow {x['frac_grown'] * 100:.0f}%" for d_, x in st["sets"].items()))
    print(f"wrote results/runs/{args.run}/domain_features.json")
    return 0


def cmd_reeval(args) -> int:
    """Re-run the multiple-choice evals on saved checkpoints of a finished run, writing
    choices_<task>.npy / gold_<task>.npy next to the existing items_<task>.npy. For runs made
    before chosen answers were logged (degenerate-answer alarm). Mac-friendly: one checkpoint
    at a time, weights from model.bf16.pt or model.pt; checkpoint 0 = the base model."""
    import numpy as np
    import torch

    from clharness.config import RunConfig
    from clharness.env import pick_device, seed_everything
    from clharness.evals.lm_eval_runner import run_lm_eval
    from clharness.model import load_model

    run_dir = REPO_ROOT / "results" / "runs" / args.run
    cfg = RunConfig.model_validate(json.load(open(run_dir / "config.resolved.json")))
    manifest = json.load(open(REPO_ROOT / "manifests" / f"eval-{cfg.eval.suite}.json"))
    tasks = [t for t in cfg.eval.lm_eval_tasks if ((t.task in args.tasks) if args.tasks else t.tier == 1)]
    seed_everything(cfg.train.seed)
    dev = pick_device(args.device or "auto")
    lm = load_model(cfg.model.model_id, device=dev, dtype=cfg.model.dtype)
    base_sd = {k: v.detach().clone() for k, v in lm.model.state_dict().items()} if 0 in args.ckpts else None
    for i in args.ckpts:
        ck = run_dir / "checkpoints" / f"ckpt_{i:04d}"
        if i == 0:
            lm.model.load_state_dict(base_sd)
        else:
            wf = ck / "model.bf16.pt" if (ck / "model.bf16.pt").exists() else ck / "model.pt"
            lm.model.load_state_dict(torch.load(wf, map_location=dev), strict=True)
        t0 = time.time()
        fp = float(lm.model.state_dict()["model.layers.13.mlp.down_proj.weight"].float().sum())
        print(f"ckpt {i} weights loaded (layer-13 down_proj sum {fp:.4f})", flush=True)
        res = run_lm_eval(lm.model, lm.tokenizer, tasks, manifest, args.batch_size or cfg.eval.batch_size, dev,
                          cfg.eval.include_path, seed=cfg.train.seed, max_length=cfg.eval.max_length)
        for task, r in res.items():
            if "_items" not in r:
                continue
            new = np.asarray(r["_items"], dtype=np.float32)
            stored = np.load(ck / f"items_{task}.npy") if (ck / f"items_{task}.npy").exists() else None
            agree = float((new == stored).mean()) if stored is not None else float("nan")
            if args.write_items:  # boundary items files of runs before 2026-09-09 held the post-re-warm state
                if stored is not None and not (ck / f"items_{task}.postrecover.npy").exists():
                    (ck / f"items_{task}.npy").rename(ck / f"items_{task}.postrecover.npy")
                np.save(ck / f"items_{task}.npy", new)
            line = (f"ckpt {i:2d} {task:14s} acc {new.mean():.3f} (stored {stored.mean() if stored is not None else float('nan'):.3f}, "
                    f"item agreement {agree:.3f}")
            if "_choices" in r:
                ch, gd = np.asarray(r["_choices"], dtype=np.int16), np.asarray(r["_gold"], dtype=np.int16)
                np.save(ck / f"choices_{task}.npy", ch)
                np.save(ck / f"gold_{task}.npy", gd)
                share = float(np.bincount(ch[ch >= 0]).max() / (ch >= 0).sum())
                derived = float(((ch == gd) == (new > 0.5)).mean())
                line += f", choice==gold reproduces {derived:.3f}) answer_share_max {share:.2f}"
            else:
                line += ")"
            print(line, flush=True)
        print(f"ckpt {i} done in {time.time() - t0:.0f}s")
    return 0


def cmd_train(args) -> int:
    from clharness.config import load_run_config
    from clharness.data.manifest import read_manifest
    from clharness.train.loop import Trainer
    from clharness.train.measure import default_hooks

    cfg = load_run_config(args.config)
    if args.name:
        cfg = cfg.model_copy(update={"name": args.name})
    p = _eval_manifest_path(cfg)
    eval_manifest = read_manifest(p) if p.exists() else None
    if cfg.eval.lm_eval_tasks and eval_manifest is None:
        print(f"missing {p.relative_to(REPO_ROOT)}; run: clh build-eval-manifest {args.config}", file=sys.stderr)
        return 2
    Trainer(cfg, default_hooks(), eval_manifest).run()
    return 0


def cmd_validate_row(args) -> int:
    from clharness.spec import read_rows

    bad = 0
    for path in args.files:
        try:
            rows = read_rows(path)
        except ValueError as e:
            print(f"INVALID {e}", file=sys.stderr)
            bad += 1
            continue
        by_status: dict[str, int] = {}
        for r in rows:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        print(f"ok {path}: {len(rows)} rows {by_status}")
    return 1 if bad else 0


def cmd_spec_hash(args) -> int:
    from clharness.spec import load_spec

    for path in args.files:
        print(f"{load_spec(path).spec_hash()}  {path}")
    return 0


def cmd_fake_rows(args) -> int:
    from clharness.fake import PLANTED, fake_rows
    from clharness.spec import write_rows

    out = Path(args.out)
    rows = fake_rows(args.seed)
    write_rows(out, rows)
    planted = out.with_name("planted_effects.json")
    planted.write_text(json.dumps(PLANTED, indent=2) + "\n")
    print(f"wrote {len(rows)} fake rows to {out} and the planted effects to {planted}")
    return 0


def cmd_run(args) -> int:
    """Run specs and append one result row each. --modal sends them to cloud GPUs in parallel."""
    from clharness.spec import append_row, load_spec

    specs = [load_spec(p) for p in args.specs]  # validate everything before spending GPU time
    if args.modal:
        import subprocess

        app = REPO_ROOT / "src" / "clharness" / "executor" / "modal_app.py"
        cmd = ["modal", "run"] + (["--detach"] if args.detach else [])
        cmd += [f"{app}::main", "--specs", ",".join(str(Path(p).resolve()) for p in args.specs),
                "--out", str(Path(args.out).resolve()), "--gpu", args.gpu]
        return subprocess.call(cmd, cwd=REPO_ROOT)
    from clharness.executor.run import run_spec

    worst = 0
    for spec in specs:
        row = run_spec(spec, backend="local")
        append_row(args.out, row)
        print(f"{row.status} {spec.name} {row.spec_hash[:12]} {row.reason or ''}")
        worst = max(worst, 0 if row.status == "finished" else 1)
    return worst


def cmd_export_schema(args) -> int:
    from clharness.spec import json_schemas

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schemas().items():
        (out / name).write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out / name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clh", description="continual-learning harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    p = sub.add_parser("validate-row", help="check JSONL files of result rows against the format")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_validate_row)
    p = sub.add_parser("spec-hash", help="print the idempotency key of run spec JSON files")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_spec_hash)
    p = sub.add_parser("fake-rows", help="write invented result rows with planted effects")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "fake" / "rows.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_fake_rows)
    p = sub.add_parser("run", help="run spec JSON files; one result row is appended per spec")
    p.add_argument("specs", nargs="+")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "rows.jsonl"))
    p.add_argument("--modal", action="store_true", help="run on Modal GPUs, one per spec, in parallel")
    p.add_argument("--gpu", default="H100", help="Modal GPU type (with --modal)")
    p.add_argument("--detach", action="store_true", help="with --modal: keep running after this terminal closes")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("export-schema", help="write the run-spec and result-row JSON Schemas")
    p.add_argument("--out", default=str(REPO_ROOT / "configs" / "schema"))
    p.set_defaults(fn=cmd_export_schema)
    p = sub.add_parser("build-data")
    p.add_argument("--set", default="smoke", help="smoke | real")
    p.add_argument("--domain", action="append", help="real set: build only these domains (repeatable)")
    p.add_argument("--model-id", default="google/gemma-3-1b-pt")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--seq-len", type=int, default=256, help="smoke: 256; real: 1024")
    p.set_defaults(fn=cmd_build_data)
    for name, fn in [("build-eval-manifest", cmd_build_eval_manifest), ("verify", cmd_verify),
                     ("sae-baseline", cmd_sae_baseline), ("train", cmd_train)]:
        p = sub.add_parser(name)
        p.add_argument("config")
        if name == "train":
            p.add_argument("--name", default=None, help="override cfg.name (output dir + wandb run name)")
        p.set_defaults(fn=fn)
    p = sub.add_parser("analyze", help="summarise a run's output into analysis.json + a text report")
    p.add_argument("runs", nargs="+")
    p.add_argument("--post-adapt-ckpt", type=int, default=3, help="checkpoint used as the packed-ppl reference (T16)")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("items", help="per-item view: which protected-set questions were lost / recovered / gained, and when")
    p.add_argument("runs", nargs="+")
    p.add_argument("--no-text", action="store_true", help="skip loading question text from the datasets")
    p.add_argument("--persist", type=int, default=3, help="'lost' = wrong at each of the last N checkpoints")
    p.set_defaults(fn=cmd_items)
    p = sub.add_parser("report", help="self-contained HTML forgetting report for a run (needs analyze + items)")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("generate", help="greedy completions of a fixed prompt set from base + saved checkpoints (demo panel)")
    p.add_argument("run")
    p.add_argument("--ckpts", type=lambda s: [int(x) for x in s.split(",")], default=[0, 10, 21, 32, 43])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("demo", help="hackathon page: report sections in pitch order + generation panel")
    p.add_argument("run")
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("features", help="domain-resolved SAE feature analysis; --pass recomputes per-domain stats from saved weights")
    p.add_argument("run")
    p.add_argument("--pass", dest="do_pass", action="store_true")
    p.add_argument("--ckpts", type=lambda s: [int(x) for x in s.split(",")], default=[0, 10, 21, 32, 43])
    p.add_argument("--max-seqs", type=int, default=128)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_features)
    p = sub.add_parser("reeval", help="re-run tier-1 MC evals on saved checkpoints to record chosen answers")
    p.add_argument("run")
    p.add_argument("--ckpts", type=lambda s: [int(x) for x in s.split(",")], required=True, help="e.g. 0,10,21,32,43")
    p.add_argument("--tasks", type=lambda s: s.split(","), default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--write-items", action="store_true", help="also replace items_<task>.npy (old file kept as .postrecover.npy)")
    p.set_defaults(fn=cmd_reeval)
    p = sub.add_parser("compare-runs", help="repeatability: diff all logged scalars of two runs")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.set_defaults(fn=cmd_compare_runs)
    p = sub.add_parser("baseline-evals", help="score every candidate eval on the base model")
    p.add_argument("config")
    p.add_argument("--limit-items", type=int, default=None, help="dry run: cap items per task")
    p.set_defaults(fn=cmd_baseline_evals)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
