"""Modal executor: run specs on cloud GPUs, one GPU per spec, in parallel.

Setup, once per Modal workspace (see README, "Running on Modal"):
    modal secret create huggingface HF_TOKEN=<token with access to Gemma and StarCoder data>
    modal run src/clharness/executor/modal_app.py::data       # build the 6.7 GB data set once

Run specs (normally through `clh run --modal SPEC...`):
    modal run src/clharness/executor/modal_app.py::main --specs a.json,b.json

Where things live on Modal:
    volume clh-data  -> <repo>/data    the tokenized corpora, checked against manifests/real.json
    volume clh-work  -> /vol           hf/ model and dataset cache, runs/ run directories,
                                       rows/<spec_hash>.json one result row per finished spec

This file is imported by the local `modal` client, which may not have the harness installed.
So it imports only the standard library and modal at the top; harness imports are inside the
remote functions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

REMOTE_REPO = "/root/repo"
WORK = "/vol"
# CLH_MODAL_GPU=none defines the app with no GPU function resources, so the data build can run
# on a workspace that is not yet allowed to request GPUs.
DEFAULT_GPU = os.environ.get("CLH_MODAL_GPU", "H100")
RUN_TIMEOUT_S = 75 * 60

app = modal.App("clharness")
data_vol = modal.Volume.from_name("clh-data", create_if_missing=True)
work_vol = modal.Volume.from_name("clh-work", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")  # must define HF_TOKEN

if modal.is_local():
    _repo = Path(__file__).resolve().parents[3]
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_sync(str(_repo), extra_options="--no-install-project")  # dependencies from uv.lock
        .env({
            "PYTHONPATH": f"{REMOTE_REPO}/src",
            "HF_HOME": f"{WORK}/hf",
            "WANDB_MODE": "offline",
            "TOKENIZERS_PARALLELISM": "false",
        })
        .add_local_dir(_repo / "src", f"{REMOTE_REPO}/src", ignore=["**/__pycache__"])
        .add_local_dir(_repo / "configs", f"{REMOTE_REPO}/configs")
        .add_local_dir(_repo / "manifests", f"{REMOTE_REPO}/manifests")
    )
else:  # inside the container the image already exists; nothing to describe
    image = modal.Image.debian_slim(python_version="3.12")

_common = {
    "image": image,
    "volumes": {f"{REMOTE_REPO}/data": data_vol, WORK: work_vol},
    "secrets": [hf_secret],
}


@app.function(cpu=8, memory=32768, timeout=3 * 3600, **_common)
def build_data(force: bool = False) -> str:
    """Build the real data set on the data volume, unless it is already there and intact."""
    import shutil

    from clharness.data.manifest import ManifestMismatch, read_manifest, verify_manifest

    committed = Path(REMOTE_REPO) / "manifests" / "real.json"
    keep = Path("/tmp/real.committed.json")
    shutil.copy(committed, keep)  # build-data rewrites manifests/real.json
    if not force:
        try:
            verify_manifest(read_manifest(keep))
            return "data volume already matches manifests/real.json"
        except ManifestMismatch as e:
            print(f"data volume incomplete ({len(e.args[0]) if e.args else '?'} problems); building")
    subprocess.run(
        [sys.executable, "-m", "clharness.cli", "build-data", "--set", "real", "--seq-len", "1024"],
        check=True, cwd=REMOTE_REPO,
    )
    data_vol.commit()
    try:
        verify_manifest(read_manifest(keep))
    except ManifestMismatch as e:
        shutil.copy(committed, f"{WORK}/real.rebuilt.json")
        work_vol.commit()
        return f"BUILT, BUT IT DIFFERS FROM manifests/real.json (rebuilt manifest saved to {WORK}/real.rebuilt.json):\n{e}"
    return "built; data volume matches manifests/real.json"


@app.function(cpu=4, memory=16384, timeout=3600, **_common)
def warm_cache() -> str:
    """Download the model and the eval data sets into the shared cache, so parallel runs
    read them instead of all downloading at once."""
    from lm_eval.tasks import TaskManager, get_task_dict

    from clharness.config import REPO_ROOT, load_run_config
    from clharness.model import load_model

    cfg = load_run_config(REPO_ROOT / "configs" / "bench" / "fast.yaml")
    load_model(cfg.model.model_id, device="cpu", dtype="float32", attn_implementation=None)
    tm = TaskManager(include_path=str(REPO_ROOT / cfg.eval.include_path))
    names = [t.task for t in cfg.eval.lm_eval_tasks]
    get_task_dict(names, tm)
    work_vol.commit()
    return f"cached {cfg.model.model_id} and eval tasks {names}"


@app.function(
    gpu=None if DEFAULT_GPU == "none" else DEFAULT_GPU, timeout=RUN_TIMEOUT_S,
    # Retries cover machine failures only (preemption, lost container). A run that raises
    # returns a `failed` row instead, and a bad result returns an `invalid` row: neither retries.
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
    **_common,
)
def run(spec_json: str, git_sha: str, gpu_type: str = DEFAULT_GPU) -> str:
    """Run one spec. Returns the result row as a JSON line and saves it on the work volume."""
    from clharness.executor.run import run_spec
    from clharness.spec import RunSpec, row_to_line

    spec = RunSpec.model_validate_json(spec_json)
    row = run_spec(spec, backend="modal", gpu_type=gpu_type, git_sha=git_sha, output_dir=f"{WORK}/runs")
    line = row_to_line(row)
    rows_dir = Path(WORK) / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    (rows_dir / f"{row.spec_hash}.json").write_text(line)
    work_vol.commit()
    return line


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).parent).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@app.local_entrypoint()
def data(force: bool = False):
    print(build_data.remote(force=force))
    print(warm_cache.remote())


@app.local_entrypoint()
def main(specs: str, out: str = "results/rows.jsonl", gpu: str = DEFAULT_GPU):
    """Run comma-separated spec files in parallel and append their rows to `out`."""
    texts = [Path(p).read_text() for p in specs.split(",") if p]
    sha = _git_sha()
    fn = run if gpu == DEFAULT_GPU else run.with_options(gpu=gpu)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"launching {len(texts)} run(s) on {gpu} at commit {sha[:10]}")
    results = fn.starmap([(t, sha, gpu) for t in texts], order_outputs=False, return_exceptions=True)
    bad = 0
    for res in results:
        if isinstance(res, Exception):  # the machine failed more often than the retry limit
            bad += 1
            print(f"LOST {type(res).__name__}: {res}")
            continue
        with open(out_path, "a") as f:
            f.write(res)
        row = json.loads(res)
        print(f"{row['status']} {row['spec']['name']} {row['spec_hash'][:12]} {row.get('reason') or ''}")
        bad += row["status"] != "finished"
    print(f"{len(texts) - bad}/{len(texts)} finished; rows appended to {out_path}")
