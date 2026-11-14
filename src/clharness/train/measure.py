"""Measurement hooks: evals and representation tracking at every checkpoint.

EvalHook   – tier-1 at every checkpoint (perplexity slices + loglikelihood MC tasks),
             tier-2 (generation tasks) when ckpt.is_tier2.
ProbeHook  – forward the probe set, capture residuals at the configured layers, and compute
             per-SAE-feature aggregates + reconstruction/drift metrics + linear CKA vs the base
             model. Per-feature arrays go to <ckpt>/features_layer{L}.npz; scalars are logged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clharness.activations import ResidualCapture
from clharness.data.loader import fixed_chunks, load_tokens
from clharness.evals.lm_eval_runner import run_lm_eval
from clharness.evals.perplexity import docs_from_jsonl, perplexity_on_chunks, perplexity_on_docs
from clharness.metrics.cka import LinearCKAAccumulator
from clharness.metrics.sae_stats import FeatureStatsAccumulator
from clharness.metrics.similarity import RobustSimilarity, Standardizer
from clharness.sae import SAESpec, load_sae
from clharness.train.hooks import BaseHook, CheckpointInfo, RunContext
from clharness.train.recover import RecoverabilityHook
from clharness.train.signals import SignalLogHook


def _fit_linear_probe(x: torch.Tensor, y: torch.Tensor, n_classes: int, steps: int = 300) -> torch.nn.Linear:
    """Multinomial logistic regression on standardised features (CPU, fp32, L-BFGS). Deterministic."""
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.LBFGS(probe.parameters(), lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(x), y) + 1e-3 * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return probe


def _probe_acc(probe: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((probe(x).argmax(1) == y).float().mean())


class EvalHook(BaseHook):
    name = "eval"

    def on_run_start(self, ctx: RunContext):
        cfg = ctx.cfg
        self.slices: dict[str, torch.Tensor] = {}
        self.doc_slices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for s in cfg.eval.perplexity_slices:
            entry = ctx.data_manifest["domains"][s.name]["splits"]["heldout"]
            toks = load_tokens(entry["tokens"]["path"])
            self.slices[s.name] = fixed_chunks(toks, cfg.data.seq_len, s.max_seqs)
            # per-document variant (T16): same number of sequences, one document each
            from clharness.config import REPO_ROOT as _R
            self.doc_slices[s.name] = docs_from_jsonl(_R / entry["jsonl"]["path"], ctx.tokenizer, cfg.data.seq_len, s.max_seqs or 256)
        self.tier1 = [t for t in cfg.eval.lm_eval_tasks if t.tier == 1]
        self.tier2 = [t for t in cfg.eval.lm_eval_tasks if t.tier == 2]
        if (self.tier1 or self.tier2) and ctx.eval_manifest is None:
            raise ValueError("lm-eval tasks configured but no eval manifest given")
        self.base_items: dict[str, np.ndarray] = {}  # task -> per-item correctness at checkpoint 0
        self.n_choices: dict[str, float] = {}
        for x in cfg.eval.lm_eval_tasks:
            cu = (ctx.eval_manifest or {}).get("tasks", {}).get(x.task, {}).get("chance_uniform")
            self.n_choices[x.task] = (1.0 / cu) if cu else 0.0  # 0 => no chance adjustment (generation tasks)
        return {"n_perplexity_slices": len(self.slices), "n_tier1": len(self.tier1), "n_tier2": len(self.tier2)}

    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        cfg = ctx.cfg
        out: dict[str, float] = {}
        t0 = time.time()
        with ctx.ac():
            for name, chunks in self.slices.items():
                r = perplexity_on_chunks(ctx.model, chunks, cfg.eval.batch_size, ctx.device, name)
                out[f"ppl/{name}"] = r.ppl
                out[f"nll/{name}"] = r.nll
                docs, dmask = self.doc_slices[name]
                rd = perplexity_on_docs(ctx.model, docs, dmask, cfg.eval.batch_size, ctx.device, name)
                out[f"ppl_doc/{name}"] = rd.ppl  # packing-independent (T16)
                out[f"nll_doc/{name}"] = rd.nll
                out[f"ppl_doc/{name}_tokens"] = float(rd.n_tokens)
        out["ppl/wall_s"] = time.time() - t0
        tasks = list(self.tier1) + (list(self.tier2) if ckpt.is_tier2 else [])
        if tasks:
            t0 = time.time()
            with ctx.ac():
                res = run_lm_eval(
                    ctx.model, ctx.tokenizer, tasks, ctx.eval_manifest, cfg.eval.batch_size, ctx.device,
                    cfg.eval.include_path, seed=cfg.train.seed, max_length=cfg.eval.max_length,
                )
            for task, metrics in res.items():
                items = metrics.pop("_items", None)
                metrics.pop("_item_metric", None)
                choices, gold = metrics.pop("_choices", None), metrics.pop("_gold", None)
                if choices is not None and ckpt.path:  # which option was picked: degenerate-answer alarm
                    np.save(Path(ckpt.path) / f"choices_{task}.npy", np.asarray(choices, dtype=np.int16))
                    np.save(Path(ckpt.path) / f"gold_{task}.npy", np.asarray(gold, dtype=np.int16))
                    ch = np.asarray(choices)
                    ch = ch[ch >= 0]
                    if ch.size:
                        out[f"lm_eval/{task}/answer_share_max"] = float(np.bincount(ch).max() / ch.size)
                for k, v in metrics.items():
                    if "_stderr" in k or k == "sample_len":
                        continue
                    out[f"lm_eval/{task}/{k.replace(',none', '').replace(',', '/')}"] = v
                if items is not None:
                    cur = np.asarray(items, dtype=np.float32)
                    np.save(Path(ckpt.path) / f"items_{task}.npy", cur) if ckpt.path else None
                    if task not in self.base_items:
                        self.base_items[task] = cur
                    base = self.base_items[task]
                    ok = ~(np.isnan(cur) | np.isnan(base))
                    n = int(ok.sum()) or 1
                    f10 = float(((base > 0.5) & (cur < 0.5) & ok).sum()) / n  # forgot
                    f01 = float(((base < 0.5) & (cur > 0.5) & ok).sum()) / n  # backward transfer
                    # chance adjustment, Harmon et al. 2025 (arXiv 2510.17776):
                    #   F_chance = (1 - acc_pre)(1 - acc_post) / (k - 1);  F_true = max(F - F_chance, 0)
                    # and the same expression for backward transfer. k = 0 => generation task, no adjustment.
                    k = self.n_choices.get(task, 0.0)
                    a_pre = float((base[ok] > 0.5).mean()); a_post = float((cur[ok] > 0.5).mean())
                    chance = (1.0 - a_pre) * (1.0 - a_post) / (k - 1.0) if k > 1 else 0.0
                    out[f"lm_eval/{task}/flip_1to0"] = f10
                    out[f"lm_eval/{task}/flip_0to1"] = f01
                    out[f"lm_eval/{task}/flip_1to0_adj"] = max(f10 - chance, 0.0)
                    out[f"lm_eval/{task}/flip_0to1_adj"] = max(f01 - chance, 0.0)
            out["lm_eval/wall_s"] = time.time() - t0
        ctx.state["eval_last"] = (ckpt.index, dict(out))  # reused by RecoverabilityHook as its "before"
        return out


class ProbeHook(BaseHook):
    name = "probe"

    def on_run_start(self, ctx: RunContext):
        cfg = ctx.cfg
        self.layers = list(cfg.model.capture_layers)
        # SAEs keyed by a short tag: the first SAE at a layer is "L{layer}", extra ones "L{layer}-{sae_id}"
        self.sae_tags: dict[str, tuple[int, object]] = {}
        seen_layers: set[int] = set()
        for sc in cfg.model.saes:
            sae = load_sae(SAESpec(sc.release, sc.sae_id, sc.layer), ctx.device)
            tag = f"L{sc.layer}" if sc.layer not in seen_layers else f"L{sc.layer}-{sc.sae_id}"
            seen_layers.add(sc.layer)
            self.sae_tags[tag] = (sc.layer, sae)
        self.saes = {L: sae for tag, (L, sae) in self.sae_tags.items() if "-" not in tag}  # primary per layer
        self.base_fvu: dict[str, float] = {}
        self.fvu_invalid_delta = cfg.model.sae_fvu_invalid_delta
        # probe set: fixed chunks from every domain's probe split (incl. general), in manifest order
        chunks = []
        self.probe_domains: list[tuple[str, int]] = []
        for dom, entry in ctx.data_manifest["domains"].items():
            if cfg.data.probe_set not in entry["splits"]:
                continue
            c = fixed_chunks(
                load_tokens(entry["splits"][cfg.data.probe_set]["tokens"]["path"]),
                cfg.data.seq_len, cfg.data.probe_max_seqs_per_domain,
            )
            chunks.append(c)
            self.probe_domains.append((dom, int(c.shape[0])))
        self.probe = torch.cat(chunks)
        self.bos_id = ctx.tokenizer.bos_token_id
        self.exclude_bos = cfg.model.exclude_bos_from_stats
        self.base_acts: dict[int, torch.Tensor] | None = None  # filled at checkpoint 0
        self.cka_max_tokens = cfg.train.cka_max_tokens
        self.std: dict[int, Standardizer] = {}  # base-model per-dim mean/std, frozen after ckpt 0
        # readout probes: domain id per probe sequence; fixed half/half split for fit / test
        self.readout = cfg.train.readout_probe
        self.seq_domain = torch.cat([torch.full((n,), i, dtype=torch.long) for i, (_d, n) in enumerate(self.probe_domains)])
        g = torch.Generator().manual_seed(cfg.train.seed)
        perm = torch.randperm(self.probe.shape[0], generator=g)
        self.fit_idx, self.test_idx = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        self.base_probe: dict[int, torch.nn.Linear] = {}  # fitted on the base model's activations
        self.k_massive = cfg.train.massive_dims_k
        self.knn_tokens = cfg.train.mknn_tokens
        self.knn_k = cfg.train.mknn_k
        return {"n_probe_seqs": float(self.probe.shape[0]), "n_probe_tokens": float(self.probe.numel())}

    def _mask(self, ids: torch.Tensor) -> torch.Tensor:
        m = torch.ones_like(ids, dtype=torch.bool)
        if self.exclude_bos:
            m &= ids != self.bos_id
        return m

    @torch.no_grad()
    def on_checkpoint(self, ctx: RunContext, ckpt: CheckpointInfo):
        cfg = ctx.cfg
        model = ctx.model
        was_training = model.training
        model.eval()
        accs = {tag: FeatureStatsAccumulator(sae, L) for tag, (L, sae) in self.sae_tags.items()}
        ckas = {L: LinearCKAAccumulator(self.saes[L].cfg.d_in, self.saes[L].cfg.d_in) for L in self.layers}
        collect_base = self.base_acts is None
        robust = {} if collect_base else {
            L: RobustSimilarity(self.std[L], self.k_massive, self.knn_tokens, self.knn_k) for L in self.layers
        }
        massive_mag = {L: torch.zeros(self.k_massive, dtype=torch.float64) for L in self.layers}
        massive_n = 0
        pooled: dict[int, list[torch.Tensor]] = {L: [] for L in self.layers}  # per-sequence mean activations
        base_buf: dict[int, list[torch.Tensor]] = {L: [] for L in self.layers}
        cka_tokens_seen = 0
        t0 = time.time()
        with ResidualCapture(model, self.layers) as cap:
            for i in range(0, self.probe.shape[0], cfg.train.probe_batch_size):
                ids = self.probe[i : i + cfg.train.probe_batch_size].to(ctx.device)
                with ctx.ac():
                    model(input_ids=ids, use_cache=False)
                mask = self._mask(ids)
                n_new = min(int(mask.sum()), max(0, self.cka_max_tokens - cka_tokens_seen))
                for tag, (L, _sae) in self.sae_tags.items():
                    accs[tag].update(cap.acts[L], mask)
                for L in self.layers:
                    acts = cap.acts[L]
                    if self.readout:
                        m3 = mask.unsqueeze(-1).to(acts.dtype)
                        pooled[L].append(((acts * m3).sum(1) / m3.sum(1).clamp(min=1)).float().cpu())
                    flat_all = acts[mask]
                    if not collect_base:
                        # T3: magnitude of the base model's top-k massive dimensions, this checkpoint
                        massive_mag[L] += flat_all[:, self.std[L].massive_dims(self.k_massive)].abs().float().cpu().double().sum(0)
                    if n_new > 0:
                        flat = flat_all[:n_new]  # [n_new, d], same token order every checkpoint
                        if collect_base:
                            # bf16, not fp16: Gemma residuals exceed fp16's range (|x| > 65504 -> inf)
                            base_buf[L].append(flat.to(torch.bfloat16).cpu())
                            self.std.setdefault(L, Standardizer(flat.shape[1])).update(flat)
                        else:
                            ref = self.base_acts[L][cka_tokens_seen : cka_tokens_seen + n_new]
                            ckas[L].update(flat.float(), ref.float())
                            robust[L].update(flat.float(), ref.float())
                if not collect_base:
                    massive_n += int(mask.sum())
                cka_tokens_seen += n_new
                cap.clear()
        if collect_base:
            self.base_acts = {L: torch.cat(base_buf[L]) for L in self.layers}
            base_path = Path(ctx.output_dir) / "base_probe_acts.pt"
            torch.save({L: t for L, t in self.base_acts.items()}, base_path)
            for L in self.layers:
                self.std[L].finalize()

        out: dict[str, float] = {"wall_s_forward": time.time() - t0, "cka_tokens": float(cka_tokens_seen)}
        ckpt_dir = Path(ckpt.path) if ckpt.path else Path(ctx.output_dir)
        for tag, (L, _sae) in self.sae_tags.items():
            fs = accs[tag].finalize()
            for k, v in fs.summary().items():
                out[f"{tag}/{k}"] = v
            # pre-registered validity flag: FVU rise over the base checkpoint beyond the threshold
            if collect_base:
                self.base_fvu[tag] = fs.recon.fvu
            out[f"{tag}/fvu_delta"] = fs.recon.fvu - self.base_fvu[tag]
            out[f"{tag}/valid"] = 1.0 if out[f"{tag}/fvu_delta"] <= self.fvu_invalid_delta else 0.0
            np.savez_compressed(
                ckpt_dir / f"features_{tag}.npz",
                freq=fs.freq.numpy(), mean_mag=fs.mean_mag.numpy(), mean_act=fs.mean_act.numpy(),
                mean_pre=fs.mean_pre.numpy(), near_thr=fs.near_thr.numpy(),
                n_tokens=np.int64(fs.n_tokens),
            )
        if self.readout:
            for L in self.layers:
                feats = torch.cat(pooled[L])
                if collect_base:
                    self.std_pool = getattr(self, "std_pool", {})
                    self.std_pool[L] = (feats[self.fit_idx].mean(0), feats[self.fit_idx].std(0).clamp(min=1e-6))
                mu, sd = self.std_pool[L]
                z = (feats - mu) / sd
                y = self.seq_domain
                n_cls = len(self.probe_domains)
                if collect_base:
                    self.base_probe[L] = _fit_linear_probe(z[self.fit_idx], y[self.fit_idx], n_cls)
                # (a) base readout applied to current activations: drops => readout misalignment
                out[f"L{L}/readout_base_probe_acc"] = _probe_acc(self.base_probe[L], z[self.test_idx], y[self.test_idx])
                # (b) fresh probe on current activations: drops => the information itself faded
                fresh = _fit_linear_probe(z[self.fit_idx], y[self.fit_idx], n_cls)
                out[f"L{L}/readout_fresh_probe_acc"] = _probe_acc(fresh, z[self.test_idx], y[self.test_idx])
        for L in self.layers:
            out[f"L{L}/cka_vs_base"] = 1.0 if collect_base else ckas[L].compute()
            dims = self.std[L].massive_dims(self.k_massive)
            for j, dim in enumerate(dims.tolist()):
                base_mag = float(self.std[L].mean_abs[dim])
                cur_mag = base_mag if collect_base else float(massive_mag[L][j] / max(massive_n, 1))
                out[f"L{L}/massive{j}_dim"] = float(dim)
                out[f"L{L}/massive{j}_mag_rel"] = cur_mag / max(base_mag, 1e-12)  # 1.0 at ckpt 0
            if collect_base:
                out[f"L{L}/cka_std"] = 1.0
                out[f"L{L}/cka_nomassive"] = 1.0
                out[f"L{L}/mknn"] = 1.0
            else:
                for k, v in robust[L].compute().items():
                    out[f"L{L}/{k}"] = v
        if was_training:
            model.train()
        return out


def default_hooks(cfg: Any = None) -> list[BaseHook]:
    """The measurement hooks. Given a config, leaves out the ones it switches off: the probe
    hook when no layers are captured, the recoverability probe at recover_tokens 0, and the
    signal log when signal_log is false. With no config, returns all four (the pilot setup)."""
    ev = EvalHook()
    if cfg is None:
        return [ev, ProbeHook(), RecoverabilityHook(ev), SignalLogHook()]
    hooks: list[BaseHook] = [ev]
    if cfg.model.capture_layers:
        hooks.append(ProbeHook())
    if cfg.train.recover_tokens > 0:
        hooks.append(RecoverabilityHook(ev))
    if cfg.train.signal_log:
        hooks.append(SignalLogHook())
    return hooks
