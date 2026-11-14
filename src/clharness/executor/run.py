"""Run one spec and return one result row. The same code runs locally and on a cloud GPU.

    spec -> RunConfig (frozen bench + the spec's overrides) -> Trainer -> ResultRow

`run_spec` never raises for a problem with the run itself: an exception becomes a `failed` row
with the reason, and a run that finishes but fails a sanity check becomes an `invalid` row. A
crash of the machine (preemption, lost container) is left to the caller's retry policy.
"""

from __future__ import annotations

import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clharness.config import REPO_ROOT, OptimCfg, RunConfig, load_yaml_with_includes
from clharness.spec import Cost, ResultRow, RunSpec

# Rough on-demand price per GPU-hour, for the cost estimate in the row and the budget guard.
GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "A10G": 1.10}


def run_name(spec: RunSpec) -> str:
    """Directory name of a run: the spec hash, so the same experiment always lands in one place."""
    return spec.spec_hash()[:16]


def config_from_spec(spec: RunSpec, output_dir: str | None = None) -> RunConfig:
    path = REPO_ROOT / "configs" / "bench" / f"{spec.bench}.yaml"
    if not path.exists():
        raise ValueError(f"unknown bench {spec.bench!r}: {path.relative_to(REPO_ROOT)} does not exist")
    raw = load_yaml_with_includes(path)
    unsupported = set(spec.optim) - set(OptimCfg.model_fields)
    if unsupported:
        raise ValueError(f"optim override(s) {sorted(unsupported)} are not supported by the harness yet")
    raw["name"] = run_name(spec)
    raw["train"]["seed"] = spec.seed
    raw["data"]["domains"] = [d.model_dump() for d in spec.stream]
    raw["optim"].update(spec.optim)
    if output_dir is not None:
        raw["train"]["output_dir"] = output_dir
    return RunConfig.model_validate(raw)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_spec(
    spec: RunSpec,
    backend: str = "local",
    gpu_type: str | None = None,
    git_sha: str | None = None,
    output_dir: str | None = None,
    attempt: int = 1,
) -> ResultRow:
    from clharness.wandb_log import git_commit

    common: dict[str, Any] = {
        "spec": spec, "spec_hash": spec.spec_hash(), "git_sha": git_sha or git_commit() or "unknown",
        "backend": backend, "attempt": attempt, "started_at": _now(),
    }
    try:
        measured, cost, run_dir = _train_and_measure(spec, gpu_type, output_dir)
    except Exception as e:  # noqa: BLE001 - every failure of the run itself is reported as a row
        reason = f"{type(e).__name__}: {e}".strip()[:600]
        return ResultRow(status="failed", reason=reason, finished_at=_now(), **common)
    invalid = measured.pop("invalid_reason")
    measured.pop("gradient_tokens")
    return ResultRow(
        status="invalid" if invalid else "finished", reason=invalid, finished_at=_now(),
        cost=cost, items_path=str(run_dir / "checkpoints"), **measured, **common,
    )


def _train_and_measure(spec: RunSpec, gpu_type: str | None, output_dir: str | None):
    import torch

    from clharness.data.manifest import read_manifest
    from clharness.executor.rows import measure_run
    from clharness.techniques import build_hooks
    from clharness.train.loop import Trainer
    from clharness.train.measure import default_hooks

    cfg = config_from_spec(spec, output_dir)
    technique_hooks = build_hooks(spec.techniques)
    run_dir = (REPO_ROOT / cfg.train.output_dir / cfg.name).resolve()
    if run_dir.exists():  # a retry or a rerun: never mix two attempts' files
        shutil.rmtree(run_dir)
    eval_manifest = read_manifest(REPO_ROOT / "manifests" / f"eval-{cfg.eval.suite}.json")

    t0 = time.time()
    trainer = Trainer(cfg, default_hooks(cfg) + technique_hooks, eval_manifest)
    trainer.run()
    wall = time.time() - t0

    measured = measure_run(Path(trainer.out_dir), [d.name for d in spec.stream])
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if gpu_type is None:
        gpu_type = torch.cuda.get_device_name(0) if trainer.device == "cuda" else trainer.device
    price = GPU_USD_PER_HOUR.get(gpu_type)
    cost = Cost(
        gradient_tokens=measured["gradient_tokens"],
        wall_clock_s=round(wall, 1),
        gpu_type=gpu_type,
        update_bytes=trainable * 4,  # the trained parameters as an fp32 delta
        trainable_params=trainable,
        est_usd=round(wall / 3600 * price, 2) if price else None,
    )
    return measured, cost, Path(trainer.out_dir)
