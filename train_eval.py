"""
Minimal train/eval harness for PathwayGNN.

This module adds:
1. Dataset builder from JSONL samples: (request context, edge targets, optional physics residual)
2. Grouped split by subject/cohort key to reduce leakage
3. Metrics per edge type + aggregate loss reporting
4. Optional physics residual loss term wiring via hidden_state_loss(..., physics_residual=...)

Expected JSONL sample schema (one sample per line):
{
  "group_id": "subject_or_cohort_id",
  "request": {
    "wild_type_fluxes": {"RXN_A": 0.12, ...},
    "perturbed_fluxes": {"RXN_A": 0.09, ...},
    "symptoms": ["fatigue", ...],
    "genotype": {"rs4680": "AG", ...}
  },
  "targets": {
    "modulates": [..],
    "regulates": [..],
    "signaling": [..],
    "bridges": [..],
    "transports_to": [..]
  },
  "physics_residual": [..]   # optional precomputed stoichiometric imbalance vector
}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dgl
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, ModelConfig
from .graph import GraphMeta, build_graph
from .loss import hidden_state_loss
from .model import WHY_TODAY_LATENT_NAMES, PathwayGNN
from .observation_features import OBSERVATION_FEATURE_DIM, build_observation_feature_vector


logger = logging.getLogger(__name__)

# Must match PathwayGNN.context_proj input width in model.py.
_CONTEXT_DIM = 8
_WHY_TODAY_LATENT_DIM = 12


@dataclass
class TrainConfig:
    out_weights: Path
    dataset_jsonl: Path | None = None
    dataset_cache: Path | None = None
    val_cache: Path | None = None
    init_weights: Path | None = None
    val_jsonl: Path | None = None
    seed: int = 42
    train_fraction: float = 0.8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    log_every: int = 1000
    batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    grad_accum_steps: int = 1
    amp: bool = True
    val_every: int = 1
    val_max_batches: int = 0
    max_seconds: float = 0.0
    throughput_csv: Path | None = None
    w_modulates: float = 1.0
    w_regulates: float = 1.6
    w_signaling: float = 1.6
    w_bridges: float = 1.6
    w_transports_to: float = 1.087
    normalize_edge_loss_by_size: bool = False
    head_dominance_warn_ratio: float = 0.75
    throughput_heartbeat_seconds: float = 10.0
    lambda_physics: float = 10.0
    lambda_smooth: float = 0.01
    lambda_confidence: float = 0.1
    lambda_global_latent: float = 2.0
    lambda_teacher_distill: float = 1.0
    lambda_posterior_gain: float = 0.5
    lambda_coupling: float = 0.2
    component_dominance_warn_ratio: float = 0.75
    adaptive_head_balance: bool = True
    head_balance_momentum: float = 0.7
    head_balance_floor: float = 0.3
    head_balance_ceiling: float = 3.0
    hard_head_weight_floor: float = 1.6
    easy_head_weight_ceiling: float = 1.1
    smooth_max_ratio_vs_pred: float = 0.30
    conf_max_ratio_vs_pred: float = 0.25
    hard_head_grad_min: float = 0.05
    hard_head_stall_logs: int = 5
    min_head_support_for_balance: float = 0.001
    low_support_easy_head_decay: float = 0.90
    grad_conflict_control: bool = True
    grad_conflict_cosine_threshold: float = -0.05
    grad_conflict_easy_scale: float = 0.85
    cache_out: Path | None = None
    cache_out_val: Path | None = None
    build_cache_only: bool = False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if bool(row.get("exclude_from_training", False)):
            continue
        filtered.append(row)
    return filtered


def _grouped_split(
    rows: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gid = row.get("group_id", "unknown")
        groups.setdefault(gid, []).append(row)
    # Leakage-safe split by group + time:
    # assign each group a representative time and split chronologically.
    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda gid: _group_time_key(groups.get(gid, [])))
    split = int(len(group_ids) * train_fraction)
    train_ids = set(group_ids[:split])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for gid, items in groups.items():
        if gid in train_ids:
            train_rows.extend(items)
        else:
            val_rows.extend(items)
    return train_rows, val_rows


@dataclass(frozen=True)
class ContextSpec:
    rxn_indices: torch.Tensor
    rxn_values: torch.Tensor
    global_ctx: float
    context_vec: torch.Tensor


@dataclass(frozen=True)
class PreparedSample:
    context: ContextSpec
    targets: dict[str, torch.Tensor]
    target_masks: dict[str, torch.Tensor]
    physics_residual: float | None
    sample_weight: float


@dataclass(frozen=True)
class PreparedTensorPack:
    rxn_ctx: torch.Tensor
    global_ctx: torch.Tensor
    context_vec: torch.Tensor
    targets: dict[str, torch.Tensor]
    target_masks: dict[str, torch.Tensor]
    teacher_targets: dict[str, torch.Tensor]
    teacher_target_masks: dict[str, torch.Tensor]
    physics_vals: torch.Tensor
    physics_mask: torch.Tensor
    physics_vec: torch.Tensor
    physics_vec_mask: torch.Tensor
    posterior_gain: torch.Tensor
    posterior_gain_mask: torch.Tensor
    objective_features: torch.Tensor
    observation_features: torch.Tensor
    sample_weight: torch.Tensor
    prev_latent: torch.Tensor

    @property
    def num_rows(self) -> int:
        return int(self.context_vec.shape[0])


class IndexDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = int(max(0, n))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return int(index)


class GraphBatchAssembler:
    def __init__(self, base_graph, node_counts: dict[str, int], device: torch.device) -> None:
        self.base_graph = base_graph
        self.node_counts = node_counts
        self.device = device
        self._batched_templates: dict[int, Any] = {}

    def _template(self, bsz: int):
        g = self._batched_templates.get(bsz)
        if g is None:
            g = dgl.batch([self.base_graph] * bsz).to(self.device)
            for ntype in g.ntypes:
                g.nodes[ntype].data["ctx"] = torch.zeros(
                    (g.num_nodes(ntype), _CONTEXT_DIM), dtype=torch.float32, device=self.device
                )
                g.nodes[ntype].data["obs"] = torch.zeros(
                    (g.num_nodes(ntype), OBSERVATION_FEATURE_DIM), dtype=torch.float32, device=self.device
                )
                g.nodes[ntype].data["prev_latent"] = torch.zeros(
                    (g.num_nodes(ntype), 3), dtype=torch.float32, device=self.device
                )
            self._batched_templates[bsz] = g
        return g

    def make_batch(
        self,
        pack: PreparedTensorPack,
        idx: torch.Tensor,
    ) -> tuple[Any, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        idx = idx.to(dtype=torch.long, device="cpu")
        bsz = int(idx.numel())
        bg = self._template(bsz)
        for ntype in bg.ntypes:
            bg.nodes[ntype].data["ctx"].zero_()
            bg.nodes[ntype].data["obs"].zero_()
            bg.nodes[ntype].data["prev_latent"].zero_()

        if self.node_counts.get("reaction", 0) > 0:
            per_graph = int(self.node_counts["reaction"])
            rxn_vals = pack.rxn_ctx.index_select(0, idx)
            rxn_vals = rxn_vals.to(self.device, non_blocking=True)
            bg.nodes["reaction"].data["ctx"].view(bsz, per_graph, _CONTEXT_DIM)[:, :, 0] = rxn_vals

        context_vec = pack.context_vec.index_select(0, idx).to(self.device, non_blocking=True)
        global_ctx = pack.global_ctx.index_select(0, idx).to(self.device, non_blocking=True)
        observation_features = pack.observation_features.index_select(0, idx).to(self.device, non_blocking=True)
        if self.node_counts.get("reaction", 0) > 0 and _CONTEXT_DIM > 1:
            per_graph = int(self.node_counts["reaction"])
            bg.nodes["reaction"].data["ctx"].view(bsz, per_graph, _CONTEXT_DIM)[:, :, 1:] = context_vec[:, None, 1:]
        if self.node_counts.get("enzyme", 0) > 0:
            per_graph = int(self.node_counts["enzyme"])
            bg.nodes["enzyme"].data["ctx"].view(bsz, per_graph, _CONTEXT_DIM)[:, :, :] = context_vec[:, None, :]
        if self.node_counts.get("metabolite", 0) > 0:
            per_graph = int(self.node_counts["metabolite"])
            met_ctx = bg.nodes["metabolite"].data["ctx"].view(bsz, per_graph, _CONTEXT_DIM)
            met_ctx[:, :, :] = 0.5 * context_vec[:, None, :]
            met_ctx[:, :, 0] = 0.5 * global_ctx.unsqueeze(1)
        for ntype in bg.ntypes:
            if self.node_counts.get(ntype, 0) <= 0:
                continue
            per_graph = int(self.node_counts[ntype])
            bg.nodes[ntype].data["obs"].view(bsz, per_graph, OBSERVATION_FEATURE_DIM)[:, :, :] = observation_features.unsqueeze(1)
        prev_latent = pack.prev_latent.index_select(0, idx).to(self.device, non_blocking=True)
        for ntype in bg.ntypes:
            if self.node_counts.get(ntype, 0) <= 0:
                continue
            per_graph = int(self.node_counts[ntype])
            bg.nodes[ntype].data["prev_latent"].view(bsz, per_graph, 3)[:, :, :] = prev_latent.unsqueeze(1)

        targets: dict[str, torch.Tensor] = {}
        target_masks: dict[str, torch.Tensor] = {}
        teacher_targets: dict[str, torch.Tensor] = {}
        teacher_target_masks: dict[str, torch.Tensor] = {}
        graph_level_keys = {
            "global_bridge_state",
            "global_signaling_state",
            "global_crosstalk_state",
            "why_today_latents",
            "teacher_why_today_latents",
        }
        for k, v in pack.targets.items():
            bt = v.index_select(0, idx).to(self.device, non_blocking=True)
            targets[k] = bt if k in graph_level_keys else bt.reshape(-1)
        for k, v in pack.target_masks.items():
            bm = v.index_select(0, idx).to(self.device, non_blocking=True)
            target_masks[k] = bm.to(dtype=torch.bool) if k in graph_level_keys else bm.reshape(-1).to(dtype=torch.bool)
        for k, v in pack.teacher_targets.items():
            bt = v.index_select(0, idx).to(self.device, non_blocking=True)
            teacher_targets[k] = bt if k in graph_level_keys else bt.reshape(-1)
        for k, v in pack.teacher_target_masks.items():
            bm = v.index_select(0, idx).to(self.device, non_blocking=True)
            teacher_target_masks[k] = bm.to(dtype=torch.bool) if k in graph_level_keys else bm.reshape(-1).to(dtype=torch.bool)
        if teacher_targets:
            targets.update(teacher_targets)
            target_masks.update(teacher_target_masks)

        physics_vals = pack.physics_vals.index_select(0, idx).to(self.device, non_blocking=True)
        physics_mask = pack.physics_mask.index_select(0, idx).to(self.device, non_blocking=True)
        physics_vec = pack.physics_vec.index_select(0, idx).to(self.device, non_blocking=True)
        physics_vec_mask = pack.physics_vec_mask.index_select(0, idx).to(self.device, non_blocking=True)
        posterior_gain = pack.posterior_gain.index_select(0, idx).to(self.device, non_blocking=True)
        posterior_gain_mask = pack.posterior_gain_mask.index_select(0, idx).to(self.device, non_blocking=True)
        sample_weight = pack.sample_weight.index_select(0, idx).to(self.device, non_blocking=True)
        return (
            bg,
            targets,
            target_masks,
            physics_vals,
            physics_mask,
            physics_vec,
            physics_vec_mask,
            posterior_gain,
            posterior_gain_mask,
            sample_weight,
            bsz,
        )


class PreparedRowsDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        meta: GraphMeta,
        target_sizes: dict[str, int],
    ) -> None:
        self._samples: list[PreparedSample] = []
        for row in rows:
            targets, target_masks = _to_targets_and_masks(row, target_sizes)
            self._samples.append(
                PreparedSample(
                    context=_prepare_context_spec(meta, row.get("request", {})),
                    targets=targets,
                    target_masks=target_masks,
                    physics_residual=_to_physics(row),
                    sample_weight=max(0.0, min(5.0, float(row.get("sample_weight", 1.0)))),
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PreparedSample:
        return self._samples[index]


class PreparedCollator:
    def __init__(self, base_graph, node_counts: dict[str, int]) -> None:
        self.base_graph = base_graph
        self.node_counts = node_counts
        self._batched_templates: dict[int, Any] = {}

    def _batched_graph_template(self, bsz: int):
        g = self._batched_templates.get(bsz)
        if g is None:
            g = dgl.batch([self.base_graph] * bsz)
            for ntype in g.ntypes:
                g.nodes[ntype].data["ctx"] = torch.zeros(
                    (g.num_nodes(ntype), _CONTEXT_DIM), dtype=torch.float32
                )
            self._batched_templates[bsz] = g
        return g

    def __call__(self, samples: list[PreparedSample]):
        bsz = len(samples)
        bg = self._batched_graph_template(bsz)
        for ntype in bg.ntypes:
            bg.nodes[ntype].data["ctx"].zero_()

        if self.node_counts.get("reaction", 0) > 0:
            rxn_ctx = bg.nodes["reaction"].data["ctx"]
            per_graph = int(self.node_counts["reaction"])
            for i, sample in enumerate(samples):
                idx = sample.context.rxn_indices
                off = i * per_graph
                if _CONTEXT_DIM > 1:
                    rxn_ctx[off:off + per_graph, 1:] = sample.context.context_vec[1:].unsqueeze(0)
                if idx.numel() == 0:
                    continue
                rxn_ctx[idx + off, 0] = sample.context.rxn_values

        if self.node_counts.get("enzyme", 0) > 0:
            enz_ctx = bg.nodes["enzyme"].data["ctx"]
            per_graph = int(self.node_counts["enzyme"])
            for i, sample in enumerate(samples):
                s = i * per_graph
                e = s + per_graph
                enz_ctx[s:e, :] = sample.context.context_vec.unsqueeze(0)

        if self.node_counts.get("metabolite", 0) > 0:
            met_ctx = bg.nodes["metabolite"].data["ctx"]
            per_graph = int(self.node_counts["metabolite"])
            for i, sample in enumerate(samples):
                s = i * per_graph
                e = s + per_graph
                met_ctx[s:e, :] = 0.5 * sample.context.context_vec.unsqueeze(0)

        targets, target_masks = _concat_targets(samples)

        physics_vals = torch.zeros(bsz, dtype=torch.float32)
        physics_mask = torch.zeros(bsz, dtype=torch.bool)
        sample_weights = torch.zeros(bsz, dtype=torch.float32)
        for i, sample in enumerate(samples):
            sample_weights[i] = float(sample.sample_weight)
            if sample.physics_residual is not None:
                physics_vals[i] = float(sample.physics_residual)
                physics_mask[i] = True

        return bg, targets, target_masks, physics_vals, physics_mask, sample_weights, bsz


class SingleProcessBatchLoader:
    def __init__(
        self,
        dataset: PreparedRowsDataset,
        batch_size: int,
        collate_fn: PreparedCollator,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.collate_fn = collate_fn
        self.shuffle = shuffle

    def __len__(self) -> int:
        n = len(self.dataset)
        q, r = divmod(n, self.batch_size)
        return q + (1 if r else 0)

    def __iter__(self):
        idx = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            chunk = idx[i:i + self.batch_size]
            samples = [self.dataset[j] for j in chunk]
            yield self.collate_fn(samples)


def _prepare_context_spec(
    meta: GraphMeta,
    request: dict[str, Any],
) -> ContextSpec:
    wild_type = request.get("wild_type_fluxes", {}) or {}
    perturbed = request.get("perturbed_fluxes", {}) or {}
    genotype = request.get("genotype", {}) or {}
    symptoms = request.get("symptoms", []) or []
    symptom_scores_0_1 = request.get("symptom_scores_0_1", {}) or {}
    lifestyle = request.get("lifestyle", {}) or {}
    lifestyle_generic = request.get("lifestyle_generic", {}) or lifestyle
    lifestyle_gene_modifiers = request.get("lifestyle_gene_modifiers", []) or []
    demographic_modifiers = request.get("demographic_modifiers", []) or []
    obj_rel = request.get("objective_channel_reliability", {}) or {}
    demographics = request.get("demographics", {}) or {}

    rxn_indices: list[int] = []
    rxn_values: list[float] = []
    # Iterate only observed flux keys to avoid O(|all reactions|) per-row scan.
    flux_keys = set(wild_type.keys()) & set(perturbed.keys()) if isinstance(wild_type, dict) and isinstance(perturbed, dict) else set()
    for rid in flux_keys:
        idx = meta.rxn_map.get(str(rid))
        if idx is None:
            continue
        wt = wild_type.get(rid)
        pt = perturbed.get(rid)
        if wt is None or pt is None:
            continue
        try:
            wt_f = float(wt)
            pt_f = float(pt)
        except (TypeError, ValueError):
            continue
        rel = (pt_f - wt_f) / (abs(wt_f) + 1e-6)
        rxn_indices.append(int(idx))
        rxn_values.append(float(torch.tanh(torch.tensor(max(-3.0, min(3.0, rel)))).item()))

    geno_count = len(genotype) if isinstance(genotype, dict) else 0
    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0
    symptom_score = 0.0
    if isinstance(symptom_scores_0_1, dict) and symptom_scores_0_1:
        sym_vals: list[float] = []
        for v in symptom_scores_0_1.values():
            try:
                sym_vals.append(max(0.0, min(1.0, float(v))))
            except Exception:
                continue
        if sym_vals:
            symptom_score = sum(sym_vals) / len(sym_vals)
    elif symptom_count > 0:
        symptom_score = min(1.0, symptom_count / 10.0)

    lifestyle_score = 0.0
    if isinstance(lifestyle_generic, dict) and lifestyle_generic:
        vals: list[float] = []
        for v in lifestyle_generic.values():
            try:
                vals.append(max(0.0, min(1.0, float(v))))
            except Exception:
                continue
        if vals:
            lifestyle_score = sum(vals) / len(vals)

    gene_mod_score = 0.0
    if isinstance(lifestyle_gene_modifiers, list) and lifestyle_gene_modifiers:
        grade_w = {"A": 1.0, "B": 0.8, "C": 0.55, "D": 0.3}
        mod_terms: list[float] = []
        for m in lifestyle_gene_modifiers:
            if not isinstance(m, dict):
                continue
            try:
                exposure = max(0.0, min(1.0, float(m.get("exposure_value_0_1", 0.5))))
                strength = max(0.0, min(2.0, float(m.get("effect_strength_prior", 0.7))))
                reliability = max(0.0, min(1.0, float(m.get("answer_reliability", 0.7))))
            except Exception:
                continue
            direction = str(m.get("effect_direction", "mixed"))
            sign = 1.0 if direction == "increase" else -1.0 if direction == "decrease" else 0.0
            evidence = grade_w.get(str(m.get("evidence_grade", "D")), 0.3)
            mod_terms.append(sign * (exposure - 0.5) * strength * evidence * reliability)
        if mod_terms:
            gene_mod_score = sum(mod_terms) / len(mod_terms)

    age_years = None
    for key in ("age_years", "age", "age_year"):
        if key in request:
            age_years = request.get(key)
            break
    if age_years is None and isinstance(demographics, dict):
        for key in ("age_years", "age"):
            if key in demographics:
                age_years = demographics.get(key)
                break
    age_score = 0.0
    if age_years is not None:
        try:
            age_val = float(age_years)
            age_score = max(-1.0, min(1.0, (age_val - 45.0) / 35.0))
        except Exception:
            age_score = 0.0

    sex_raw = (
        request.get("sex_at_birth")
        or request.get("sex")
        or (demographics.get("sex_at_birth") if isinstance(demographics, dict) else None)
        or (demographics.get("sex") if isinstance(demographics, dict) else None)
        or "unknown"
    )
    sex_s = str(sex_raw).strip().lower()
    if sex_s in {"female", "f", "woman"}:
        sex_score = 1.0
    elif sex_s in {"male", "m", "man"}:
        sex_score = -1.0
    else:
        sex_score = 0.0

    demog_modifier_score = 0.0
    if isinstance(demographic_modifiers, list) and demographic_modifiers:
        vals: list[float] = []
        for m in demographic_modifiers:
            if not isinstance(m, dict):
                continue
            try:
                mult = float(m.get("multiplier", 1.0))
            except Exception:
                continue
            vals.append(max(-1.0, min(1.0, mult - 1.0)))
        if vals:
            demog_modifier_score = sum(vals) / len(vals)

    # Fusion policy:
    # - generic lifestyle is weak likelihood
    # - gene-conditioned modifiers are stronger priors (when evidence grade is high)
    # - objective channels (labs/wearables) downweight lifestyle contribution on conflict.
    labs_present = bool(obj_rel.get("labs_present", False))
    wear_present = bool(obj_rel.get("wearables_present", False))
    try:
        labs_rel = max(0.0, min(1.0, float(obj_rel.get("labs_reliability", 0.0))))
    except Exception:
        labs_rel = 0.0
    try:
        wear_rel = max(0.0, min(1.0, float(obj_rel.get("wearables_reliability", 0.0))))
    except Exception:
        wear_rel = 0.0
    objective_strength = max(labs_rel if labs_present else 0.0, wear_rel if wear_present else 0.0)
    lifestyle_conflict_downweight = 1.0 - 0.5 * objective_strength

    # Add objective channel features as stronger evidence channel in v2+.
    objective_delta_score = 0.0
    obj_feats = request.get("objective_channel_features", {}) or {}
    if isinstance(obj_feats, dict) and obj_feats:
        vals: list[float] = []
        for v in obj_feats.values():
            try:
                vals.append(max(-1.0, min(1.0, float(v))))
            except Exception:
                continue
        if vals:
            objective_delta_score = sum(vals) / len(vals)

    global_ctx = float(
        torch.tanh(
            torch.tensor(
                0.04 * geno_count
                + 0.10 * symptom_score
                + (0.04 * lifestyle_score + 0.12 * gene_mod_score) * lifestyle_conflict_downweight
                + 0.08 * objective_delta_score
                + 0.07 * age_score
                + 0.05 * sex_score
                + 0.10 * demog_modifier_score
            )
        )
    )
    context_vec = torch.tensor(
        [
            global_ctx,  # 0: fused global context
            float(symptom_score),  # 1: symptom burden
            float(lifestyle_score),  # 2: lifestyle burden
            float(objective_delta_score),  # 3: objective channel delta
            float(gene_mod_score),  # 4: lifestyle-gene modifier score
            float(age_score),  # 5: age context
            float(sex_score),  # 6: sex-at-birth context
            float(max(0.0, min(1.0, geno_count / 8.0))),  # 7: genotype burden
        ],
        dtype=torch.float32,
    )
    return ContextSpec(
        rxn_indices=torch.tensor(rxn_indices, dtype=torch.long),
        rxn_values=torch.tensor(rxn_values, dtype=torch.float32),
        global_ctx=global_ctx,
        context_vec=context_vec,
    )


def _to_physics(sample: dict[str, Any]) -> float | None:
    physics = sample.get("physics_residual")
    if physics is None:
        return None
    if isinstance(physics, (int, float)):
        return float(physics)
    if isinstance(physics, list) and physics:
        vals: list[float] = []
        for v in physics:
            try:
                vals.append(float(v))
            except Exception:
                continue
        if vals:
            return float(sum(x * x for x in vals) / len(vals))
    return None


def _to_physics_vector(sample: dict[str, Any]) -> list[float]:
    vec = sample.get("physics_residual_vector")
    if not isinstance(vec, list) or not vec:
        return []
    out: list[float] = []
    for v in vec:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == fv and abs(fv) < 1e9:
            out.append(fv)
    return out


def _to_posterior_gain(sample: dict[str, Any]) -> float | None:
    try:
        post = float(sample.get("physics_residual"))
        baseline = float(sample.get("baseline_physics_residual"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(post) or not math.isfinite(baseline):
        return None
    scale = max(abs(baseline), 1e-6)
    gain = (baseline - post) / scale
    return float(max(0.0, min(1.0, gain)))


def _to_prev_latent(sample: dict[str, Any]) -> tuple[float, float, float]:
    req = sample.get("request", {}) if isinstance(sample.get("request"), dict) else {}
    # Preferred explicit request fields for temporal rollout.
    keys = (
        req.get("prev_global_bridge_state"),
        req.get("prev_global_signaling_state"),
        req.get("prev_global_crosstalk_state"),
    )
    out: list[float] = []
    for v in keys:
        try:
            out.append(float(v))
        except Exception:
            out.append(0.0)
    # Fallback: previous posterior object if present.
    if all(abs(x) < 1e-12 for x in out):
        prev = sample.get("prev_posterior", {})
        if isinstance(prev, dict):
            try:
                out = [
                    float(prev.get("global_bridge_state", 0.0)),
                    float(prev.get("global_signaling_state", 0.0)),
                    float(prev.get("global_crosstalk_state", 0.0)),
                ]
            except Exception:
                out = [0.0, 0.0, 0.0]
    return (
        max(-1.0, min(1.0, out[0])),
        max(-1.0, min(1.0, out[1])),
        max(-1.0, min(1.0, out[2])),
    )


def _dense_from_sparse(
    spec: dict[str, Any],
    fallback_size: int,
    semantics: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Prefer model-derived fallback size to guarantee compatibility.
    size = max(0, int(fallback_size))
    out = torch.zeros(size, dtype=torch.float32)
    mask = torch.ones(size, dtype=torch.bool) if semantics == "sparse_nonzero_from_dense" else torch.zeros(size, dtype=torch.bool)

    idxs = spec.get("indices", [])
    vals = spec.get("values", [])
    if not isinstance(idxs, list) or not isinstance(vals, list):
        return out, mask
    n = min(len(idxs), len(vals))
    for i in range(n):
        try:
            idx = int(idxs[i])
            if idx < 0 or idx >= size:
                continue
            out[idx] = float(vals[i])
            mask[idx] = True
        except Exception:
            continue
    return out, mask


def _to_targets_and_masks(
    sample: dict[str, Any],
    target_sizes: dict[str, int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def _why_today_dense_from_object(value: Any, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(max(0, int(size)), dtype=torch.float32)
        mask = torch.zeros(max(0, int(size)), dtype=torch.bool)
        if not isinstance(value, dict):
            return out, mask
        for idx, name in enumerate(WHY_TODAY_LATENT_NAMES[: out.numel()]):
            raw = value.get(name)
            score = raw
            if isinstance(raw, dict):
                score = raw.get("score", raw.get("value", raw.get("posterior", 0.0)))
            try:
                out[idx] = float(score)
                mask[idx] = True
            except Exception:
                continue
        return out, mask

    def _dense_target(
        value: Any,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(max(0, int(size)), dtype=torch.float32)
        mask = torch.zeros(max(0, int(size)), dtype=torch.bool)
        if not isinstance(value, list):
            return out, mask
        n = min(len(value), out.numel())
        for i in range(n):
            try:
                out[i] = float(value[i])
                mask[i] = True
            except Exception:
                continue
        return out, mask

    def _latent_val(key: str) -> float:
        try:
            return float(latent.get(key, 0.0))
        except Exception:
            return 0.0

    def _teacher_why_today_dense() -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM)), dtype=torch.float32)
        mask = torch.zeros(int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM)), dtype=torch.bool)
        distillation = sample.get("distillation", {})
        if not isinstance(distillation, dict):
            return out, mask
        latent_name = str(distillation.get("latent") or "").strip()
        if not latent_name or latent_name not in WHY_TODAY_LATENT_NAMES:
            return out, mask
        idx = WHY_TODAY_LATENT_NAMES.index(latent_name)
        if idx >= out.numel():
            return out, mask
        try:
            out[idx] = float(distillation.get("teacher_pred"))
            mask[idx] = True
        except Exception:
            return torch.zeros_like(out), torch.zeros_like(mask)
        return out, mask

    out: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    dense = sample.get("targets")
    sparse = sample.get("targets_sparse")
    sparse_semantics = str(sample.get("targets_sparse_semantics") or "partial_observed")
    if sparse_semantics not in {"partial_observed", "sparse_nonzero_from_dense"}:
        sparse_semantics = "partial_observed"
    keys = ("modulates", "regulates", "signaling", "bridges", "transports_to")
    global_keys = ("global_bridge_state", "global_signaling_state", "global_crosstalk_state")
    latent = sample.get("latent_targets", {})
    if not isinstance(latent, dict):
        latent = {}
    why_today_latents = None

    if isinstance(dense, dict):
        for key in keys:
            vals = dense.get(key, [])
            t = torch.tensor(vals, dtype=torch.float32)
            want = int(target_sizes.get(key, int(t.numel())))
            if t.numel() > want:
                t = t[:want]
            elif t.numel() < want:
                pad = torch.zeros(want - t.numel(), dtype=torch.float32)
                t = torch.cat([t, pad], dim=0)
            out[key] = t
            masks[key] = torch.ones(want, dtype=torch.bool)
        for gk in global_keys:
            out[gk] = torch.tensor([_latent_val(gk)], dtype=torch.float32)
            masks[gk] = torch.tensor([gk in latent], dtype=torch.bool)
        why_today_latents, why_today_mask = _dense_target(
            dense.get("why_today_latents", []),
            target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
        )
        if not bool(why_today_mask.any()):
            why_today_latents, why_today_mask = _why_today_dense_from_object(
                sample.get("why_today_latents", {}),
                target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
            )
        out["why_today_latents"] = why_today_latents
        masks["why_today_latents"] = why_today_mask
        teacher_latents, teacher_mask = _teacher_why_today_dense()
        out["teacher_why_today_latents"] = teacher_latents
        masks["teacher_why_today_latents"] = teacher_mask
        return out, masks

    if isinstance(sparse, dict):
        for key in keys:
            spec = sparse.get(key, {})
            if isinstance(spec, dict):
                out[key], masks[key] = _dense_from_sparse(spec, target_sizes.get(key, 0), sparse_semantics)
            else:
                out[key] = torch.zeros(target_sizes.get(key, 0), dtype=torch.float32)
                masks[key] = torch.zeros(target_sizes.get(key, 0), dtype=torch.bool)
        for gk in global_keys:
            out[gk] = torch.tensor([_latent_val(gk)], dtype=torch.float32)
            masks[gk] = torch.tensor([gk in latent], dtype=torch.bool)
        why_today_latents, why_today_mask = _dense_target(
            sample.get("why_today_latents", []),
            target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
        )
        if not bool(why_today_mask.any()):
            why_today_latents, why_today_mask = _why_today_dense_from_object(
                sample.get("why_today_latents", {}),
                target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
            )
        out["why_today_latents"] = why_today_latents
        masks["why_today_latents"] = why_today_mask
        teacher_latents, teacher_mask = _teacher_why_today_dense()
        out["teacher_why_today_latents"] = teacher_latents
        masks["teacher_why_today_latents"] = teacher_mask
        return out, masks

    for key in keys:
        out[key] = torch.zeros(target_sizes.get(key, 0), dtype=torch.float32)
        masks[key] = torch.zeros(target_sizes.get(key, 0), dtype=torch.bool)
    for gk in global_keys:
        out[gk] = torch.tensor([_latent_val(gk)], dtype=torch.float32)
        masks[gk] = torch.tensor([gk in latent], dtype=torch.bool)
    why_today_latents, why_today_mask = _dense_target(
        sample.get("why_today_latents", []),
        target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
    )
    if not bool(why_today_mask.any()):
        why_today_latents, why_today_mask = _why_today_dense_from_object(
            sample.get("why_today_latents", {}),
            target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM),
        )
    out["why_today_latents"] = why_today_latents
    masks["why_today_latents"] = why_today_mask
    teacher_latents, teacher_mask = _teacher_why_today_dense()
    out["teacher_why_today_latents"] = teacher_latents
    masks["teacher_why_today_latents"] = teacher_mask
    return out, masks


def _edge_mse(outputs: dict[str, Any], targets: dict[str, torch.Tensor], etype: str) -> float:
    pred = outputs.get(f"{etype}_hidden")
    target = targets.get(etype)
    if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    n = min(pred.numel(), target.numel())
    return float(torch.mean((pred[:n] - target[:n]) ** 2).item())


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _group_time_key(rows: list[dict[str, Any]]) -> float:
    best = float("-inf")
    for row in rows:
        ts = (
            (((row.get("window_post") or {}).get("t_end")))
            or (((row.get("window_post") or {}).get("t_start")))
            or (((row.get("intervention_event") or {}).get("t_applied")))
        )
        if not isinstance(ts, str) or not ts.strip():
            continue
        raw = ts.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            v = float(__import__("datetime").datetime.fromisoformat(raw).timestamp())
            if v > best:
                best = v
        except Exception:
            continue
    return best if best != float("-inf") else 0.0


def _detect_gpu_util() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        line = out.strip().splitlines()[0].strip()
        return float(line)
    except Exception:
        return None


def _head_grad_norms(model: PathwayGNN) -> dict[str, float]:
    heads = {
        "modulates": model.modulates_head,
        "regulates": model.regulates_head,
        "signaling": model.signaling_head,
        "bridges": model.bridges_head,
        "transports_to": model.transports_to_head,
    }
    out: dict[str, float] = {}
    for name, module in heads.items():
        sq = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            sq += float(torch.sum(p.grad.detach() ** 2).item())
        out[name] = float(math.sqrt(max(0.0, sq)))
    return out


def _flat_grad(module: torch.nn.Module) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    for p in module.parameters():
        if p.grad is None:
            continue
        parts.append(p.grad.detach().reshape(-1))
    if not parts:
        return None
    return torch.cat(parts, dim=0)


def _apply_grad_conflict_damping(
    model: PathwayGNN,
    hard_heads: tuple[str, ...],
    easy_heads: tuple[str, ...],
    cosine_threshold: float,
    easy_scale: float,
) -> dict[str, float]:
    """PCGrad-lite: damp easy-head gradients when anti-aligned with hard-head mean.

    This is a lightweight conflict handler that avoids expensive multi-backward
    passes while still reducing destructive easy-vs-hard gradient interference.
    """
    modules = {
        "modulates": model.modulates_head,
        "regulates": model.regulates_head,
        "signaling": model.signaling_head,
        "bridges": model.bridges_head,
        "transports_to": model.transports_to_head,
    }
    hard_vecs: list[torch.Tensor] = []
    for h in hard_heads:
        gv = _flat_grad(modules[h])
        if gv is not None and gv.numel() > 0:
            hard_vecs.append(gv)
    if not hard_vecs:
        return {"checks": 0.0, "conflicts": 0.0}
    hard_mean = hard_vecs[0]
    for v in hard_vecs[1:]:
        if v.shape == hard_mean.shape:
            hard_mean = hard_mean + v
    hard_mean = hard_mean / float(max(1, len(hard_vecs)))
    hnorm = float(torch.norm(hard_mean).item())
    if hnorm <= 1e-12:
        return {"checks": 0.0, "conflicts": 0.0}

    checks = 0
    conflicts = 0
    for e in easy_heads:
        ev = _flat_grad(modules[e])
        if ev is None or ev.numel() == 0 or ev.shape != hard_mean.shape:
            continue
        en = float(torch.norm(ev).item())
        if en <= 1e-12:
            continue
        cos = float(torch.dot(ev, hard_mean).item() / (en * hnorm + 1e-12))
        checks += 1
        if cos < float(cosine_threshold):
            conflicts += 1
            for p in modules[e].parameters():
                if p.grad is not None:
                    p.grad.mul_(float(easy_scale))
    return {"checks": float(checks), "conflicts": float(conflicts)}


def _concat_targets(rows: list[PreparedSample]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    keys = (
        "modulates",
        "regulates",
        "signaling",
        "bridges",
        "transports_to",
        "global_bridge_state",
        "global_signaling_state",
        "global_crosstalk_state",
        "why_today_latents",
        "teacher_why_today_latents",
    )
    chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    mask_chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    for row in rows:
        t = row.targets
        m = row.target_masks
        for k in keys:
            chunks[k].append(t.get(k, torch.zeros(0, dtype=torch.float32)))
            mask_chunks[k].append(m.get(k, torch.zeros(0, dtype=torch.bool)))
    targets = {k: (torch.cat(v, dim=0) if v else torch.zeros(0, dtype=torch.float32)) for k, v in chunks.items()}
    target_masks = {k: (torch.cat(v, dim=0) if v else torch.zeros(0, dtype=torch.bool)) for k, v in mask_chunks.items()}
    return targets, target_masks


def _rows_to_tensor_pack(
    rows: list[dict[str, Any]],
    meta: GraphMeta,
    target_sizes: dict[str, int],
) -> PreparedTensorPack:
    n = len(rows)
    r = int(len(meta.rxn_map))
    rxn_ctx = torch.zeros((n, r), dtype=torch.float32)
    global_ctx = torch.zeros(n, dtype=torch.float32)
    context_vec = torch.zeros((n, _CONTEXT_DIM), dtype=torch.float32)
    targets = {
        "modulates": torch.zeros((n, int(target_sizes.get("modulates", 0))), dtype=torch.float32),
        "regulates": torch.zeros((n, int(target_sizes.get("regulates", 0))), dtype=torch.float32),
        "signaling": torch.zeros((n, int(target_sizes.get("signaling", 0))), dtype=torch.float32),
        "bridges": torch.zeros((n, int(target_sizes.get("bridges", 0))), dtype=torch.float32),
        "transports_to": torch.zeros((n, int(target_sizes.get("transports_to", 0))), dtype=torch.float32),
        "global_bridge_state": torch.zeros((n, 1), dtype=torch.float32),
        "global_signaling_state": torch.zeros((n, 1), dtype=torch.float32),
        "global_crosstalk_state": torch.zeros((n, 1), dtype=torch.float32),
        "why_today_latents": torch.zeros((n, int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM))), dtype=torch.float32),
    }
    target_masks = {
        "modulates": torch.zeros((n, int(target_sizes.get("modulates", 0))), dtype=torch.bool),
        "regulates": torch.zeros((n, int(target_sizes.get("regulates", 0))), dtype=torch.bool),
        "signaling": torch.zeros((n, int(target_sizes.get("signaling", 0))), dtype=torch.bool),
        "bridges": torch.zeros((n, int(target_sizes.get("bridges", 0))), dtype=torch.bool),
        "transports_to": torch.zeros((n, int(target_sizes.get("transports_to", 0))), dtype=torch.bool),
        "global_bridge_state": torch.zeros((n, 1), dtype=torch.bool),
        "global_signaling_state": torch.zeros((n, 1), dtype=torch.bool),
        "global_crosstalk_state": torch.zeros((n, 1), dtype=torch.bool),
        "why_today_latents": torch.zeros((n, int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM))), dtype=torch.bool),
    }
    teacher_targets = {
        "teacher_why_today_latents": torch.zeros((n, int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM))), dtype=torch.float32),
    }
    teacher_target_masks = {
        "teacher_why_today_latents": torch.zeros((n, int(target_sizes.get("why_today_latents", _WHY_TODAY_LATENT_DIM))), dtype=torch.bool),
    }
    physics_vals = torch.zeros(n, dtype=torch.float32)
    physics_mask = torch.zeros(n, dtype=torch.bool)
    physics_rows: list[list[float]] = []
    max_vec_len = 0
    posterior_gain = torch.zeros(n, dtype=torch.float32)
    posterior_gain_mask = torch.zeros(n, dtype=torch.bool)
    objective_features = torch.zeros((n, 6), dtype=torch.float32)
    observation_features = torch.zeros((n, OBSERVATION_FEATURE_DIM), dtype=torch.float32)
    sample_weight = torch.ones(n, dtype=torch.float32)

    for i, row in enumerate(rows):
        c = _prepare_context_spec(meta, row.get("request", {}))
        if c.rxn_indices.numel() > 0:
            rxn_ctx[i, c.rxn_indices] = c.rxn_values
        global_ctx[i] = float(c.global_ctx)
        context_vec[i, :] = c.context_vec
        t, tm = _to_targets_and_masks(row, target_sizes)
        for k, tv in t.items():
            if k in targets and targets[k].numel() > 0:
                targets[k][i, : tv.numel()] = tv
        for k, mv in tm.items():
            if k in target_masks and target_masks[k].numel() > 0:
                target_masks[k][i, : mv.numel()] = mv
            elif k in teacher_target_masks and teacher_target_masks[k].numel() > 0:
                teacher_target_masks[k][i, : mv.numel()] = mv
        for k, tv in t.items():
            if k in teacher_targets and teacher_targets[k].numel() > 0:
                teacher_targets[k][i, : tv.numel()] = tv
        p = _to_physics(row)
        if p is not None:
            physics_vals[i] = float(p)
            physics_mask[i] = True
        pv = _to_physics_vector(row)
        physics_rows.append(pv)
        max_vec_len = max(max_vec_len, len(pv))
        pg = _to_posterior_gain(row)
        if pg is not None:
            posterior_gain[i] = float(pg)
            posterior_gain_mask[i] = True
        req = row.get("request", {}) if isinstance(row.get("request"), dict) else {}
        obj = req.get("objective_channel_reliability", {}) if isinstance(req.get("objective_channel_reliability"), dict) else {}
        objective_features[i, 0] = 1.0 if bool(obj.get("labs_present", False)) else 0.0
        objective_features[i, 1] = 1.0 if bool(obj.get("wearables_present", False)) else 0.0
        try:
            objective_features[i, 2] = max(0.0, min(1.0, float(obj.get("labs_reliability", 0.0))))
        except Exception:
            pass
        try:
            objective_features[i, 3] = max(0.0, min(1.0, float(obj.get("wearables_reliability", 0.0))))
        except Exception:
            pass
        feats = req.get("objective_channel_features", {}) if isinstance(req.get("objective_channel_features"), dict) else {}
        try:
            objective_features[i, 4] = float(feats.get("labs_delta_score", 0.0))
            objective_features[i, 5] = float(feats.get("wearables_delta_score", 0.0))
        except Exception:
            pass
        sample_weight[i] = max(0.0, min(5.0, float(row.get("sample_weight", 1.0))))
        observation_features[i, :] = torch.tensor(build_observation_feature_vector(req), dtype=torch.float32)

    physics_vec = torch.zeros((n, max_vec_len), dtype=torch.float32)
    physics_vec_mask = torch.zeros(n, dtype=torch.bool)
    if max_vec_len > 0:
        for i, pv in enumerate(physics_rows):
            if not pv:
                continue
            k = min(max_vec_len, len(pv))
            physics_vec[i, :k] = torch.tensor(pv[:k], dtype=torch.float32)
            physics_vec_mask[i] = True
    prev_latent = torch.zeros((n, 3), dtype=torch.float32)
    for i, row in enumerate(rows):
        pv = _to_prev_latent(row)
        prev_latent[i, 0] = pv[0]
        prev_latent[i, 1] = pv[1]
        prev_latent[i, 2] = pv[2]

    return PreparedTensorPack(
        rxn_ctx=rxn_ctx,
        global_ctx=global_ctx,
        context_vec=context_vec,
        targets=targets,
        target_masks=target_masks,
        teacher_targets=teacher_targets,
        teacher_target_masks=teacher_target_masks,
        physics_vals=physics_vals,
        physics_mask=physics_mask,
        physics_vec=physics_vec,
        physics_vec_mask=physics_vec_mask,
        posterior_gain=posterior_gain,
        posterior_gain_mask=posterior_gain_mask,
        objective_features=objective_features,
        observation_features=observation_features,
        sample_weight=sample_weight,
        prev_latent=prev_latent,
    )


def _save_pack(path: Path, pack: PreparedTensorPack) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "rxn_ctx": pack.rxn_ctx,
            "global_ctx": pack.global_ctx,
            "context_vec": pack.context_vec,
            "targets": pack.targets,
            "target_masks": pack.target_masks,
            "teacher_targets": pack.teacher_targets,
            "teacher_target_masks": pack.teacher_target_masks,
            "physics_vals": pack.physics_vals,
            "physics_mask": pack.physics_mask,
            "physics_vec": pack.physics_vec,
            "physics_vec_mask": pack.physics_vec_mask,
            "posterior_gain": pack.posterior_gain,
            "posterior_gain_mask": pack.posterior_gain_mask,
            "objective_features": pack.objective_features,
            "observation_features": pack.observation_features,
            "sample_weight": pack.sample_weight,
            "prev_latent": pack.prev_latent,
        },
        path,
    )


def _load_pack(path: Path) -> PreparedTensorPack:
    blob = torch.load(path, map_location="cpu")
    n_rows = int(blob["global_ctx"].shape[0]) if "global_ctx" in blob else int(blob["context_vec"].shape[0])
    context_vec = blob.get("context_vec")
    if context_vec is None:
        # Legacy cache compatibility: only scalar global context existed.
        gc = blob["global_ctx"].reshape(-1, 1).to(dtype=torch.float32)
        context_vec = torch.zeros((n_rows, _CONTEXT_DIM), dtype=torch.float32)
        context_vec[:, 0:1] = gc
    targets = blob["targets"]
    target_masks = blob.get("target_masks")
    if not isinstance(target_masks, dict):
        target_masks = {}
        for k, v in targets.items():
            target_masks[k] = torch.ones_like(v, dtype=torch.bool)
    teacher_targets = blob.get("teacher_targets")
    if not isinstance(teacher_targets, dict):
        teacher_targets = {
            "teacher_why_today_latents": torch.zeros_like(
                targets.get("why_today_latents", torch.zeros((n_rows, _WHY_TODAY_LATENT_DIM), dtype=torch.float32))
            )
        }
    teacher_target_masks = blob.get("teacher_target_masks")
    if not isinstance(teacher_target_masks, dict):
        teacher_target_masks = {
            "teacher_why_today_latents": torch.zeros_like(
                teacher_targets.get("teacher_why_today_latents", torch.zeros((n_rows, _WHY_TODAY_LATENT_DIM), dtype=torch.float32)),
                dtype=torch.bool,
            )
        }
    return PreparedTensorPack(
        rxn_ctx=blob["rxn_ctx"],
        global_ctx=blob["global_ctx"],
        context_vec=context_vec,
        targets=targets,
        target_masks=target_masks,
        teacher_targets=teacher_targets,
        teacher_target_masks=teacher_target_masks,
        physics_vals=blob["physics_vals"],
        physics_mask=blob["physics_mask"],
        physics_vec=blob.get("physics_vec", torch.zeros((n_rows, 0), dtype=torch.float32)),
        physics_vec_mask=blob.get("physics_vec_mask", torch.zeros(n_rows, dtype=torch.bool)),
        posterior_gain=blob.get("posterior_gain", torch.zeros(n_rows, dtype=torch.float32)),
        posterior_gain_mask=blob.get("posterior_gain_mask", torch.zeros(n_rows, dtype=torch.bool)),
        objective_features=blob.get("objective_features", torch.zeros((n_rows, 6), dtype=torch.float32)),
        observation_features=blob.get("observation_features", torch.zeros((n_rows, OBSERVATION_FEATURE_DIM), dtype=torch.float32)),
        sample_weight=blob["sample_weight"],
        prev_latent=blob.get("prev_latent", torch.zeros((n_rows, 3), dtype=torch.float32)),
    )


def _head_support_from_pack(pack: PreparedTensorPack) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, t in pack.target_masks.items():
        if t.numel() <= 0:
            out[k] = 0.0
            continue
        labeled = torch.count_nonzero(t.to(dtype=torch.bool)).item()
        out[k] = float(labeled) / float(max(1, t.numel()))
    return out


def _make_index_loader(
    n_rows: int,
    cfg: TrainConfig,
    *,
    shuffle: bool,
):
    # DataLoader multiprocessing/prefetch overlaps host-side batch preparation
    # with model compute when num_workers > 0 and prefetch_factor is set.
    # Source: PyTorch DataLoader docs + tuning guide.
    # https://docs.pytorch.org/docs/stable/data.html
    # https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    #
    # We intentionally keep graph assembly in the main process and only fetch
    # row indices in workers to avoid Python/DGL object IPC overhead.
    ds = IndexDataset(n_rows)
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(cfg.batch_size)),
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": max(0, int(cfg.num_workers)),
        "pin_memory": bool(cfg.pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
    return DataLoader(ds, **kwargs)


def train_and_eval(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    if cfg.dataset_cache is None:
        if cfg.dataset_jsonl is None:
            raise ValueError("dataset_jsonl is required when dataset_cache is not provided")
        rows = _load_jsonl(cfg.dataset_jsonl)
        if cfg.val_jsonl is not None:
            train_rows = rows
            val_rows = _load_jsonl(cfg.val_jsonl)
        else:
            train_rows, val_rows = _grouped_split(rows, cfg.train_fraction, cfg.seed)

    graph, meta = build_graph(DataConfig())
    model_cfg = ModelConfig(
        lambda_physics=float(cfg.lambda_physics),
        lambda_smooth=float(cfg.lambda_smooth),
        lambda_confidence=float(cfg.lambda_confidence),
        lambda_global_latent=float(cfg.lambda_global_latent),
        lambda_teacher_distill=float(cfg.lambda_teacher_distill),
        lambda_posterior_gain=float(cfg.lambda_posterior_gain),
        lambda_coupling=float(cfg.lambda_coupling),
    )
    model = PathwayGNN(model_cfg).to(device)
    if cfg.init_weights is not None:
        state = torch.load(cfg.init_weights, map_location="cpu")
        incompatible_keys = model.load_state_dict(state, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning(
                "Loaded init weights %s with missing keys: %s",
                cfg.init_weights,
                ", ".join(sorted(incompatible_keys.missing_keys)),
            )
        if incompatible_keys.unexpected_keys:
            logger.warning(
                "Loaded init weights %s with unexpected keys: %s",
                cfg.init_weights,
                ", ".join(sorted(incompatible_keys.unexpected_keys)),
            )
    with torch.no_grad():
        bootstrap_outputs = model(graph.to(device))
    target_sizes = {
        "modulates": int(bootstrap_outputs.get("modulates_hidden", torch.zeros(0)).numel()),
        "regulates": int(bootstrap_outputs.get("regulates_hidden", torch.zeros(0)).numel()),
        "signaling": int(bootstrap_outputs.get("signaling_hidden", torch.zeros(0)).numel()),
        "bridges": int(bootstrap_outputs.get("bridges_hidden", torch.zeros(0)).numel()),
        "transports_to": int(bootstrap_outputs.get("transports_to_hidden", torch.zeros(0)).numel()),
        "global_bridge_state": int(bootstrap_outputs.get("global_bridge_state", torch.zeros(1)).numel()),
        "global_signaling_state": int(bootstrap_outputs.get("global_signaling_state", torch.zeros(1)).numel()),
        "global_crosstalk_state": int(bootstrap_outputs.get("global_crosstalk_state", torch.zeros(1)).numel()),
        "why_today_latents": int(bootstrap_outputs.get("why_today_latents", torch.zeros(_WHY_TODAY_LATENT_DIM)).numel()),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    else:
        # Torch 2.2 uses torch.cuda.amp; keep compatibility with cloud DLAMI images.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)
    node_counts = {nt: int(graph.num_nodes(nt)) for nt in graph.ntypes}
    assembler = GraphBatchAssembler(base_graph=graph, node_counts=node_counts, device=device)

    prep_start = time.time()
    if cfg.dataset_cache is not None:
        train_pack = _load_pack(cfg.dataset_cache)
    else:
        train_pack = _rows_to_tensor_pack(train_rows, meta, target_sizes)
        if cfg.cache_out is not None:
            _save_pack(cfg.cache_out, train_pack)
    if cfg.val_cache is not None:
        val_pack = _load_pack(cfg.val_cache)
    else:
        val_pack = _rows_to_tensor_pack(val_rows, meta, target_sizes) if val_rows else None
        if val_pack is not None and cfg.cache_out_val is not None:
            _save_pack(cfg.cache_out_val, val_pack)
    train_head_support = _head_support_from_pack(train_pack)
    val_head_support = _head_support_from_pack(val_pack) if val_pack is not None else {}
    prep_elapsed = time.time() - prep_start
    print(
        (
            f"prepared_rows train={train_pack.num_rows} val={(val_pack.num_rows if val_pack is not None else 0)} "
            f"in={_fmt_seconds(prep_elapsed)} "
            f"batch_size={cfg.batch_size} workers={cfg.num_workers} "
            f"prefetch_factor={cfg.prefetch_factor if cfg.num_workers > 0 else 0}"
        ),
        flush=True,
    )
    if cfg.build_cache_only:
        print("build_cache_only=true", flush=True)
        return
    print(f"train_head_support_density={train_head_support}", flush=True)
    if val_head_support:
        print(f"val_head_support_density={val_head_support}", flush=True)

    train_loader = _make_index_loader(train_pack.num_rows, cfg, shuffle=True)
    val_loader = _make_index_loader(val_pack.num_rows, cfg, shuffle=False) if val_pack is not None else None

    edge_loss_weights = {
        "modulates": float(cfg.w_modulates),
        "regulates": float(cfg.w_regulates),
        "signaling": float(cfg.w_signaling),
        "bridges": float(cfg.w_bridges),
        "transports_to": float(cfg.w_transports_to),
    }
    hard_heads = ("signaling", "bridges", "transports_to")
    easy_heads = ("modulates", "regulates")
    global_heads = ("global_bridge_state", "global_signaling_state", "global_crosstalk_state")
    if cfg.normalize_edge_loss_by_size:
        for k in list(edge_loss_weights.keys()):
            edge_loss_weights[k] = edge_loss_weights[k] / max(1.0, float(target_sizes.get(k, 1)))

    throughput_writer = None
    throughput_fh = None
    if cfg.throughput_csv is not None:
        cfg.throughput_csv.parent.mkdir(parents=True, exist_ok=True)
        throughput_fh = cfg.throughput_csv.open("w", newline="", encoding="utf-8")
        throughput_writer = csv.writer(throughput_fh)
        throughput_writer.writerow([
            "stage",
            "epoch",
            "rows_seen",
            "rows_total",
            "pct",
            "rows_per_s",
            "elapsed_s",
            "eta_s",
            "loss_w",
            "loss_u",
            "physics_rows_used",
            "comp_pred_w",
            "comp_gain_w",
            "comp_physics_w",
            "comp_smooth_w",
            "comp_conf_w",
            "gpu_util",
        ])

    run_start = time.time()
    stop_early = False

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_unweighted = 0.0
        train_head_w = {k: 0.0 for k in ("modulates", "regulates", "signaling", "bridges", "transports_to")}
        train_global_w = {k: 0.0 for k in global_heads}
        train_head_grad = {k: 0.0 for k in ("modulates", "regulates", "signaling", "bridges", "transports_to")}
        train_head_grad_samples = 0
        train_components = {"pred": 0.0, "posterior_gain": 0.0, "physics": 0.0, "smooth": 0.0, "confidence": 0.0}
        hard_head_stall_count = 0
        epoch_start = time.time()
        total_train = int(train_pack.num_rows)
        seen_train = 0
        seen_phys_rows = 0.0
        next_train_log = max(1, int(cfg.log_every))
        last_train_csv_emit = epoch_start
        # Mini-batching is used to reduce variance and improve hardware utilization
        # versus single-sample SGD in large-scale training.
        # Source: Bottou et al., "Stochastic Gradient Descent Tricks" (2012)
        # https://leon.bottou.org/publications/pdf/tricks-2012.pdf
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, idx in enumerate(train_loader, start=1):
            (
                bg,
                targets,
                target_masks,
                physics_t,
                physics_mask,
                physics_vec_t,
                physics_vec_mask,
                posterior_gain_t,
                posterior_gain_mask,
                sample_weights,
                batch_rows,
            ) = assembler.make_batch(train_pack, idx)
            with autocast_ctx():
                outputs = model(bg)
                loss, breakdown = hidden_state_loss(
                    outputs,
                    targets,
                    target_masks=target_masks,
                    cfg=model_cfg,
                    physics_residual=physics_t,
                    physics_residual_vector=physics_vec_t,
                    physics_vector_mask=physics_vec_mask,
                    posterior_gain_target=posterior_gain_t,
                    posterior_gain_mask=posterior_gain_mask,
                    sample_weights=sample_weights,
                    physics_mask=physics_mask,
                    edge_loss_weights=edge_loss_weights,
                    edge_size_normalize=False,
                    smooth_max_ratio_vs_pred=float(cfg.smooth_max_ratio_vs_pred),
                    conf_max_ratio_vs_pred=float(cfg.conf_max_ratio_vs_pred),
                )
                loss_unweighted = torch.tensor(float(breakdown.get("total_unweighted", float(loss.item()))))
                loss_for_step = loss / float(max(1, int(cfg.grad_accum_steps)))
            scaler.scale(loss_for_step).backward()
            grad_now = _head_grad_norms(model)
            for k in train_head_grad:
                train_head_grad[k] += float(grad_now.get(k, 0.0))
            train_head_grad_samples += 1
            if (batch_idx % max(1, int(cfg.grad_accum_steps)) == 0) or (seen_train + batch_rows >= total_train):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            seen_train += batch_rows
            seen_phys_rows += float(breakdown.get("physics_rows_used", 0.0))
            train_loss += float(loss.item()) * batch_rows
            train_loss_unweighted += float(loss_unweighted.item()) * batch_rows
            for k in train_head_w:
                train_head_w[k] += float(breakdown.get(f"pred_{k}", 0.0)) * batch_rows
            for k in train_global_w:
                train_global_w[k] += float(breakdown.get(f"pred_{k}", 0.0)) * batch_rows
            train_components["pred"] += float(breakdown.get("pred", 0.0)) * batch_rows
            train_components["posterior_gain"] += float(breakdown.get("posterior_gain", 0.0)) * batch_rows
            train_components["physics"] += float(breakdown.get("physics", 0.0)) * batch_rows
            train_components["smooth"] += float(breakdown.get("smooth", 0.0)) * batch_rows
            train_components["confidence"] += float(breakdown.get("confidence", 0.0)) * batch_rows

            if seen_train >= next_train_log or seen_train >= total_train:
                elapsed = time.time() - epoch_start
                rate = seen_train / max(1e-6, elapsed)
                remaining = (total_train - seen_train) / max(1e-6, rate)
                pct = 100.0 * (seen_train / max(1, total_train))
                avg_loss = train_loss / max(1, seen_train)
                avg_loss_unweighted = train_loss_unweighted / max(1, seen_train)
                avg_head_grad = {
                    k: (v / max(1, train_head_grad_samples))
                    for k, v in train_head_grad.items()
                }
                print(
                    (
                        f"[train] epoch={epoch+1}/{cfg.num_epochs} "
                        f"step={seen_train}/{max(1,total_train)} ({pct:.2f}%) "
                        f"avg_loss_w={avg_loss:.6f} "
                        f"avg_loss_u={avg_loss_unweighted:.6f} "
                        f"gain_w={(train_components['posterior_gain'] / max(1, seen_train)):.6f} "
                        f"physics_rows_used={int(seen_phys_rows)}/{max(1,seen_train)} "
                        f"head_grad_norm={avg_head_grad} "
                        f"rate={rate:.2f} rows/s "
                        f"elapsed={_fmt_seconds(elapsed)} "
                        f"eta={_fmt_seconds(remaining)}"
                    ),
                    flush=True,
                )
                hard_mean_grad = sum(float(avg_head_grad.get(h, 0.0)) for h in hard_heads) / float(len(hard_heads))
                if hard_mean_grad < float(cfg.hard_head_grad_min):
                    hard_head_stall_count += 1
                else:
                    hard_head_stall_count = 0
                if hard_head_stall_count >= int(max(1, cfg.hard_head_stall_logs)):
                    print(
                        f"abort_hard_head_stall=true hard_mean_grad={hard_mean_grad:.6f} "
                        f"threshold={cfg.hard_head_grad_min:.6f} stall_logs={hard_head_stall_count}",
                        flush=True,
                    )
                    stop_early = True
                    break
                if throughput_writer is not None:
                    throughput_writer.writerow([
                        "train",
                        epoch + 1,
                        seen_train,
                        total_train,
                        f"{pct:.4f}",
                        f"{rate:.6f}",
                        f"{elapsed:.6f}",
                        f"{remaining:.6f}",
                        f"{avg_loss:.8f}",
                        f"{avg_loss_unweighted:.8f}",
                        int(seen_phys_rows),
                        f"{(train_components['pred'] / max(1, seen_train)):.8f}",
                        f"{(train_components['posterior_gain'] / max(1, seen_train)):.8f}",
                        f"{(train_components['physics'] / max(1, seen_train)):.8f}",
                        f"{(train_components['smooth'] / max(1, seen_train)):.8f}",
                        f"{(train_components['confidence'] / max(1, seen_train)):.8f}",
                        _detect_gpu_util(),
                    ])
                    throughput_fh.flush()
                    last_train_csv_emit = time.time()
                while seen_train >= next_train_log:
                    next_train_log += max(1, int(cfg.log_every))
            if throughput_writer is not None:
                now = time.time()
                if (now - last_train_csv_emit) >= max(1.0, float(cfg.throughput_heartbeat_seconds)):
                    elapsed = now - epoch_start
                    rate = seen_train / max(1e-6, elapsed)
                    remaining = (total_train - seen_train) / max(1e-6, rate)
                    pct = 100.0 * (seen_train / max(1, total_train))
                    avg_loss = train_loss / max(1, seen_train)
                    avg_loss_unweighted = train_loss_unweighted / max(1, seen_train)
                    throughput_writer.writerow([
                        "train",
                        epoch + 1,
                        seen_train,
                        total_train,
                        f"{pct:.4f}",
                        f"{rate:.6f}",
                        f"{elapsed:.6f}",
                        f"{remaining:.6f}",
                        f"{avg_loss:.8f}",
                        f"{avg_loss_unweighted:.8f}",
                        int(seen_phys_rows),
                        f"{(train_components['pred'] / max(1, seen_train)):.8f}",
                        f"{(train_components['posterior_gain'] / max(1, seen_train)):.8f}",
                        f"{(train_components['physics'] / max(1, seen_train)):.8f}",
                        f"{(train_components['smooth'] / max(1, seen_train)):.8f}",
                        f"{(train_components['confidence'] / max(1, seen_train)):.8f}",
                        _detect_gpu_util(),
                    ])
                    throughput_fh.flush()
                    last_train_csv_emit = now
            if cfg.max_seconds > 0 and (time.time() - run_start) >= cfg.max_seconds:
                stop_early = True
                break

        model.eval()
        should_run_val = (
            (val_pack is not None and val_pack.num_rows > 0)
            and (
                ((epoch + 1) % max(1, int(cfg.val_every)) == 0)
                or (epoch + 1 == cfg.num_epochs)
            )
        )
        if not should_run_val:
            train_loss /= max(1, seen_train)
            train_loss_unweighted /= max(1, seen_train)
            print(
                f"epoch={epoch+1} train_loss_w={train_loss:.6f} "
                f"train_loss_u={train_loss_unweighted:.6f} "
                f"train_gain_w={(train_components['posterior_gain'] / max(1, seen_train)):.6f} "
                f"val_skipped=true edge_mse={{}}",
                flush=True,
            )
        else:
            val_loss = 0.0
            val_loss_unweighted = 0.0
            val_head_w = {k: 0.0 for k in ("modulates", "regulates", "signaling", "bridges", "transports_to")}
            val_global_w = {k: 0.0 for k in global_heads}
            val_components = {"pred": 0.0, "posterior_gain": 0.0, "physics": 0.0, "smooth": 0.0, "confidence": 0.0}
            val_metrics = {k: [] for k in ("modulates", "regulates", "signaling", "bridges", "transports_to")}
            val_start = time.time()
            total_val = int(val_pack.num_rows if val_pack is not None else 0)
            seen_val = 0
            seen_val_phys_rows = 0.0
            next_val_log = max(1, int(cfg.log_every))
            last_val_csv_emit = val_start
            with torch.no_grad():
                for batch_idx, idx in enumerate(val_loader or [], start=1):
                    (
                        bg,
                        targets,
                        target_masks,
                        physics_t,
                        physics_mask,
                        physics_vec_t,
                        physics_vec_mask,
                        posterior_gain_t,
                        posterior_gain_mask,
                        sample_weights,
                        batch_rows,
                    ) = assembler.make_batch(val_pack, idx)
                    with autocast_ctx():
                        outputs = model(bg)
                        loss, breakdown = hidden_state_loss(
                            outputs,
                            targets,
                            target_masks=target_masks,
                            cfg=model_cfg,
                            physics_residual=physics_t,
                            physics_residual_vector=physics_vec_t,
                            physics_vector_mask=physics_vec_mask,
                            posterior_gain_target=posterior_gain_t,
                            posterior_gain_mask=posterior_gain_mask,
                            sample_weights=sample_weights,
                            physics_mask=physics_mask,
                            edge_loss_weights=edge_loss_weights,
                            edge_size_normalize=False,
                            smooth_max_ratio_vs_pred=float(cfg.smooth_max_ratio_vs_pred),
                            conf_max_ratio_vs_pred=float(cfg.conf_max_ratio_vs_pred),
                        )
                    loss_unweighted = torch.tensor(float(breakdown.get("total_unweighted", float(loss.item()))))

                    seen_val += batch_rows
                    seen_val_phys_rows += float(breakdown.get("physics_rows_used", 0.0))
                    val_loss += float(loss.item()) * batch_rows
                    val_loss_unweighted += float(loss_unweighted.item()) * batch_rows
                    for k in val_head_w:
                        val_head_w[k] += float(breakdown.get(f"pred_{k}", 0.0)) * batch_rows
                    for k in val_global_w:
                        val_global_w[k] += float(breakdown.get(f"pred_{k}", 0.0)) * batch_rows
                    val_components["pred"] += float(breakdown.get("pred", 0.0)) * batch_rows
                    val_components["posterior_gain"] += float(breakdown.get("posterior_gain", 0.0)) * batch_rows
                    val_components["physics"] += float(breakdown.get("physics", 0.0)) * batch_rows
                    val_components["smooth"] += float(breakdown.get("smooth", 0.0)) * batch_rows
                    val_components["confidence"] += float(breakdown.get("confidence", 0.0)) * batch_rows
                    for et in val_metrics:
                        val_metrics[et].append(_edge_mse(outputs, targets, et))

                    if seen_val >= next_val_log or seen_val >= total_val:
                        elapsed = time.time() - val_start
                        rate = seen_val / max(1e-6, elapsed)
                        remaining = (total_val - seen_val) / max(1e-6, rate)
                        pct = 100.0 * (seen_val / max(1, total_val))
                        avg_loss = val_loss / max(1, seen_val)
                        avg_loss_unweighted = val_loss_unweighted / max(1, seen_val)
                        print(
                            (
                                f"[val] epoch={epoch+1}/{cfg.num_epochs} "
                                f"step={seen_val}/{max(1,total_val)} ({pct:.2f}%) "
                                f"avg_loss_w={avg_loss:.6f} "
                                f"avg_loss_u={avg_loss_unweighted:.6f} "
                                f"gain_w={(val_components['posterior_gain'] / max(1, seen_val)):.6f} "
                                f"physics_rows_used={int(seen_val_phys_rows)}/{max(1,seen_val)} "
                                f"rate={rate:.2f} rows/s "
                                f"elapsed={_fmt_seconds(elapsed)} "
                                f"eta={_fmt_seconds(remaining)}"
                            ),
                            flush=True,
                        )
                        if throughput_writer is not None:
                            throughput_writer.writerow([
                                "val",
                                epoch + 1,
                                seen_val,
                                total_val,
                                f"{pct:.4f}",
                                f"{rate:.6f}",
                                f"{elapsed:.6f}",
                                f"{remaining:.6f}",
                                f"{avg_loss:.8f}",
                                f"{avg_loss_unweighted:.8f}",
                                int(seen_val_phys_rows),
                                f"{(val_components['pred'] / max(1, seen_val)):.8f}",
                                f"{(val_components['posterior_gain'] / max(1, seen_val)):.8f}",
                                f"{(val_components['physics'] / max(1, seen_val)):.8f}",
                                f"{(val_components['smooth'] / max(1, seen_val)):.8f}",
                                f"{(val_components['confidence'] / max(1, seen_val)):.8f}",
                                _detect_gpu_util(),
                            ])
                            throughput_fh.flush()
                            last_val_csv_emit = time.time()
                        while seen_val >= next_val_log:
                            next_val_log += max(1, int(cfg.log_every))
                    if throughput_writer is not None:
                        now = time.time()
                        if (now - last_val_csv_emit) >= max(1.0, float(cfg.throughput_heartbeat_seconds)):
                            elapsed = now - val_start
                            rate = seen_val / max(1e-6, elapsed)
                            remaining = (total_val - seen_val) / max(1e-6, rate)
                            pct = 100.0 * (seen_val / max(1, total_val))
                            avg_loss = val_loss / max(1, seen_val)
                            avg_loss_unweighted = val_loss_unweighted / max(1, seen_val)
                            throughput_writer.writerow([
                                "val",
                                epoch + 1,
                                seen_val,
                                total_val,
                                f"{pct:.4f}",
                                f"{rate:.6f}",
                                f"{elapsed:.6f}",
                                f"{remaining:.6f}",
                                f"{avg_loss:.8f}",
                                f"{avg_loss_unweighted:.8f}",
                                int(seen_val_phys_rows),
                                f"{(val_components['pred'] / max(1, seen_val)):.8f}",
                                f"{(val_components['posterior_gain'] / max(1, seen_val)):.8f}",
                                f"{(val_components['physics'] / max(1, seen_val)):.8f}",
                                f"{(val_components['smooth'] / max(1, seen_val)):.8f}",
                                f"{(val_components['confidence'] / max(1, seen_val)):.8f}",
                                _detect_gpu_util(),
                            ])
                            throughput_fh.flush()
                            last_val_csv_emit = now
                    if cfg.val_max_batches > 0 and batch_idx >= cfg.val_max_batches:
                        break
                    if cfg.max_seconds > 0 and (time.time() - run_start) >= cfg.max_seconds:
                        stop_early = True
                        break

            train_loss /= max(1, seen_train)
            train_loss_unweighted /= max(1, seen_train)
            val_loss /= max(1, seen_val)
            val_loss_unweighted /= max(1, seen_val)
            metric_summary = {}
            for k, vals in val_metrics.items():
                valid = [vv for vv in vals if vv == vv]
                metric_summary[k] = (sum(valid) / len(valid)) if valid else float("nan")
            train_head_avg = {k: (v / max(1, seen_train)) for k, v in train_head_w.items()}
            train_global_avg = {k: (v / max(1, seen_train)) for k, v in train_global_w.items()}
            val_head_avg = {k: (v / max(1, seen_val)) for k, v in val_head_w.items()}
            val_global_avg = {k: (v / max(1, seen_val)) for k, v in val_global_w.items()}
            train_comp_avg = {k: (v / max(1, seen_train)) for k, v in train_components.items()}
            val_comp_avg = {k: (v / max(1, seen_val)) for k, v in val_components.items()}
            for scope, heads in (("train", train_head_avg), ("val", val_head_avg)):
                total_head = sum(max(0.0, float(v)) for v in heads.values())
                if total_head > 0:
                    top = max(heads.items(), key=lambda kv: kv[1])
                    share = float(top[1]) / total_head
                    if share >= cfg.head_dominance_warn_ratio:
                        print(
                            f"warning={scope}_head_dominance head={top[0]} share={share:.3f} threshold={cfg.head_dominance_warn_ratio:.3f}",
                            flush=True,
                        )
            for scope, comps in (("train", train_comp_avg), ("val", val_comp_avg)):
                total_comp = sum(max(0.0, float(v)) for v in comps.values())
                if total_comp > 0:
                    top = max(comps.items(), key=lambda kv: kv[1])
                    share = float(top[1]) / total_comp
                    if share >= cfg.component_dominance_warn_ratio:
                        print(
                            f"warning={scope}_component_dominance component={top[0]} share={share:.3f} threshold={cfg.component_dominance_warn_ratio:.3f}",
                            flush=True,
                        )
            print(
                f"epoch={epoch+1} train_loss_w={train_loss:.6f} train_loss_u={train_loss_unweighted:.6f} "
                f"val_loss_w={val_loss:.6f} val_loss_u={val_loss_unweighted:.6f} "
                f"train_head_pred_w={train_head_avg} val_head_pred_w={val_head_avg} "
                f"train_global_pred_w={train_global_avg} val_global_pred_w={val_global_avg} "
                f"train_components_w={train_comp_avg} val_components_w={val_comp_avg} "
                f"val_rows_seen={seen_val} edge_mse={metric_summary}",
                flush=True,
            )
            if cfg.adaptive_head_balance:
                # Adaptive per-head balancing by inverse observed contribution.
                # Keeps sparse heads (e.g., signaling/bridges) from being dominated.
                current = dict(edge_loss_weights)
                head_vals = {k: max(1e-8, float(train_head_avg.get(k, 0.0))) for k in current}
                active_for_balance: dict[str, bool] = {}
                for k in current:
                    support = float(train_head_support.get(k, 0.0))
                    # Low-support easy heads are excluded from inverse-contribution balancing
                    # to avoid runaway weights from tiny denominators.
                    active_for_balance[k] = not (k in easy_heads and support < float(cfg.min_head_support_for_balance))
                active_vals = [v for k, v in head_vals.items() if active_for_balance.get(k, True)]
                if active_vals:
                    gm = 1.0
                    for v in active_vals:
                        gm *= max(1e-8, float(v))
                    gm = gm ** (1.0 / float(len(active_vals)))
                else:
                    gm = 1.0
                for k in current:
                    if active_for_balance.get(k, True):
                        target = gm / head_vals[k]
                        updated = cfg.head_balance_momentum * float(current[k]) + (1.0 - cfg.head_balance_momentum) * float(target)
                        edge_loss_weights[k] = float(max(cfg.head_balance_floor, min(cfg.head_balance_ceiling, updated)))
                    else:
                        # Decay low-support easy heads toward a modest weight so they don't dominate.
                        decayed = float(current[k]) * float(cfg.low_support_easy_head_decay)
                        edge_loss_weights[k] = float(max(cfg.head_balance_floor, min(cfg.easy_head_weight_ceiling, decayed)))
                # Additional policy for this project:
                # - hard heads get a minimum floor,
                # - easy heads get an upper cap,
                # then re-center to keep total scale stable.
                for k in hard_heads:
                    edge_loss_weights[k] = max(float(cfg.hard_head_weight_floor), float(edge_loss_weights[k]))
                for k in easy_heads:
                    edge_loss_weights[k] = min(float(cfg.easy_head_weight_ceiling), float(edge_loss_weights[k]))
                mean_w = sum(float(v) for v in edge_loss_weights.values()) / float(len(edge_loss_weights))
                if mean_w > 1e-8:
                    for k in edge_loss_weights:
                        edge_loss_weights[k] = float(edge_loss_weights[k]) / mean_w
                print(f"adaptive_head_weights={edge_loss_weights}", flush=True)
        if cfg.max_seconds > 0 and (time.time() - run_start) >= cfg.max_seconds:
            stop_early = True
        if stop_early:
            print(f"stopped_early=true max_seconds={cfg.max_seconds}", flush=True)
            break

    cfg.out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.out_weights)
    print(f"saved_weights={cfg.out_weights}")
    if throughput_fh is not None:
        throughput_fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval harness for PathwayGNN")
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--val-cache", type=Path, default=None)
    parser.add_argument("--out-weights", required=True, type=Path)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--throughput-csv", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--w-modulates", type=float, default=1.0)
    parser.add_argument("--w-regulates", type=float, default=1.0)
    parser.add_argument("--w-signaling", type=float, default=1.0)
    parser.add_argument("--w-bridges", type=float, default=1.0)
    parser.add_argument("--w-transports-to", type=float, default=1.0)
    parser.add_argument("--normalize-edge-loss-by-size", action="store_true")
    parser.add_argument("--head-dominance-warn-ratio", type=float, default=0.75)
    parser.add_argument("--throughput-heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--lambda-physics", type=float, default=10.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--lambda-confidence", type=float, default=0.1)
    parser.add_argument("--lambda-global-latent", type=float, default=2.0)
    parser.add_argument("--lambda-teacher-distill", type=float, default=1.0)
    parser.add_argument("--lambda-posterior-gain", type=float, default=0.5)
    parser.add_argument("--lambda-coupling", type=float, default=0.2)
    parser.add_argument("--component-dominance-warn-ratio", type=float, default=0.75)
    parser.add_argument("--no-adaptive-head-balance", action="store_true")
    parser.add_argument("--head-balance-momentum", type=float, default=0.7)
    parser.add_argument("--head-balance-floor", type=float, default=0.25)
    parser.add_argument("--head-balance-ceiling", type=float, default=4.0)
    parser.add_argument("--hard-head-weight-floor", type=float, default=1.0)
    parser.add_argument("--easy-head-weight-ceiling", type=float, default=1.5)
    parser.add_argument("--smooth-max-ratio-vs-pred", type=float, default=0.30)
    parser.add_argument("--conf-max-ratio-vs-pred", type=float, default=0.25)
    parser.add_argument("--hard-head-grad-min", type=float, default=0.05)
    parser.add_argument("--hard-head-stall-logs", type=int, default=5)
    parser.add_argument("--min-head-support-for-balance", type=float, default=0.001)
    parser.add_argument("--low-support-easy-head-decay", type=float, default=0.90)
    parser.add_argument("--cache-out", type=Path, default=None)
    parser.add_argument("--cache-out-val", type=Path, default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    args = parser.parse_args()

    if args.dataset_jsonl is None and args.dataset_cache is None:
        raise SystemExit("one of --dataset-jsonl or --dataset-cache is required")
    if args.build_cache_only and args.cache_out is None and args.dataset_cache is None:
        raise SystemExit("--build-cache-only requires --cache-out when --dataset-cache is not provided")

    train_and_eval(
        TrainConfig(
            dataset_jsonl=args.dataset_jsonl,
            dataset_cache=args.dataset_cache,
            val_cache=args.val_cache,
            out_weights=args.out_weights,
            init_weights=args.init_weights,
            val_jsonl=args.val_jsonl,
            train_fraction=args.train_fraction,
            num_epochs=args.epochs,
            seed=args.seed,
            log_every=args.log_every,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=(not args.no_persistent_workers),
            pin_memory=(not args.no_pin_memory),
            grad_accum_steps=max(1, int(args.grad_accum_steps)),
            amp=(not args.no_amp),
            val_every=args.val_every,
            val_max_batches=args.val_max_batches,
            max_seconds=args.max_seconds,
            throughput_csv=args.throughput_csv,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            w_modulates=args.w_modulates,
            w_regulates=args.w_regulates,
            w_signaling=args.w_signaling,
            w_bridges=args.w_bridges,
            w_transports_to=args.w_transports_to,
            normalize_edge_loss_by_size=args.normalize_edge_loss_by_size,
            head_dominance_warn_ratio=args.head_dominance_warn_ratio,
            throughput_heartbeat_seconds=args.throughput_heartbeat_seconds,
            lambda_physics=args.lambda_physics,
            lambda_smooth=args.lambda_smooth,
            lambda_confidence=args.lambda_confidence,
            lambda_global_latent=args.lambda_global_latent,
            lambda_teacher_distill=args.lambda_teacher_distill,
            lambda_posterior_gain=args.lambda_posterior_gain,
            lambda_coupling=args.lambda_coupling,
            component_dominance_warn_ratio=args.component_dominance_warn_ratio,
            adaptive_head_balance=(not args.no_adaptive_head_balance),
            head_balance_momentum=args.head_balance_momentum,
            head_balance_floor=args.head_balance_floor,
            head_balance_ceiling=args.head_balance_ceiling,
            hard_head_weight_floor=args.hard_head_weight_floor,
            easy_head_weight_ceiling=args.easy_head_weight_ceiling,
            smooth_max_ratio_vs_pred=args.smooth_max_ratio_vs_pred,
            conf_max_ratio_vs_pred=args.conf_max_ratio_vs_pred,
            hard_head_grad_min=args.hard_head_grad_min,
            hard_head_stall_logs=args.hard_head_stall_logs,
            min_head_support_for_balance=args.min_head_support_for_balance,
            low_support_easy_head_decay=args.low_support_easy_head_decay,
            cache_out=args.cache_out,
            cache_out_val=args.cache_out_val,
            build_cache_only=args.build_cache_only,
        )
    )


if __name__ == "__main__":
    main()
