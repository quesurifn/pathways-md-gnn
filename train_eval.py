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
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ModelConfig
from .graph import GraphMeta, build_graph
from .loss import hidden_state_loss
from .model import PathwayGNN


@dataclass
class TrainConfig:
    dataset_jsonl: Path
    out_weights: Path
    seed: int = 42
    train_fraction: float = 0.8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _grouped_split(
    rows: list[dict[str, Any]],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gid = row.get("group_id", "unknown")
        groups.setdefault(gid, []).append(row)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

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


def _apply_request_context(
    g,
    meta: GraphMeta,
    request: dict[str, Any],
):
    # Keep this in sync with calibrate._apply_request_context.
    g = g.clone()
    for ntype in g.ntypes:
        g.nodes[ntype].data["ctx"] = torch.zeros((g.num_nodes(ntype), 1), dtype=torch.float32)

    wild_type = request.get("wild_type_fluxes", {}) or {}
    perturbed = request.get("perturbed_fluxes", {}) or {}
    genotype = request.get("genotype", {}) or {}
    symptoms = request.get("symptoms", []) or []

    if g.num_nodes("reaction") > 0:
        rxn_ctx = g.nodes["reaction"].data["ctx"]
        for rid, idx in meta.rxn_map.items():
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
            rxn_ctx[idx, 0] = float(torch.tanh(torch.tensor(max(-3.0, min(3.0, rel)))))

    geno_count = len(genotype) if isinstance(genotype, dict) else 0
    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0
    global_ctx = float(torch.tanh(torch.tensor(0.05 * geno_count + 0.1 * symptom_count)))
    if g.num_nodes("enzyme") > 0:
        g.nodes["enzyme"].data["ctx"][:] = global_ctx
    if g.num_nodes("metabolite") > 0:
        g.nodes["metabolite"].data["ctx"][:] = 0.5 * global_ctx
    return g


def _to_targets(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw = sample.get("targets", {})
    out: dict[str, torch.Tensor] = {}
    for key in ("modulates", "regulates", "signaling", "bridges", "transports_to"):
        vals = raw.get(key, [])
        out[key] = torch.tensor(vals, dtype=torch.float32)
    return out


def _edge_mse(outputs: dict[str, Any], targets: dict[str, torch.Tensor], etype: str) -> float:
    pred = outputs.get(f"{etype}_hidden")
    target = targets.get(etype)
    if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    n = min(pred.numel(), target.numel())
    return float(torch.mean((pred[:n] - target[:n]) ** 2).item())


def train_and_eval(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)

    rows = _load_jsonl(cfg.dataset_jsonl)
    train_rows, val_rows = _grouped_split(rows, cfg.train_fraction, cfg.seed)

    graph, meta = build_graph(DataConfig())
    model_cfg = ModelConfig()
    model = PathwayGNN(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        for row in train_rows:
            g = _apply_request_context(graph, meta, row.get("request", {}))
            outputs = model(g)
            targets = _to_targets(row)
            physics = row.get("physics_residual")
            physics_t = None
            if physics is not None:
                physics_t = torch.tensor(physics, dtype=torch.float32)
            loss, _ = hidden_state_loss(outputs, targets, model_cfg, physics_residual=physics_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        val_metrics = {k: [] for k in ("modulates", "regulates", "signaling", "bridges", "transports_to")}
        with torch.no_grad():
            for row in val_rows:
                g = _apply_request_context(graph, meta, row.get("request", {}))
                outputs = model(g)
                targets = _to_targets(row)
                physics = row.get("physics_residual")
                physics_t = None
                if physics is not None:
                    physics_t = torch.tensor(physics, dtype=torch.float32)
                loss, _ = hidden_state_loss(outputs, targets, model_cfg, physics_residual=physics_t)
                val_loss += float(loss.item())
                for et in val_metrics:
                    val_metrics[et].append(_edge_mse(outputs, targets, et))

        train_loss /= max(1, len(train_rows))
        val_loss /= max(1, len(val_rows))
        metric_summary = {
            k: (sum(vv for vv in vals if vv == vv) / max(1, sum(1 for vv in vals if vv == vv)))
            for k, vals in val_metrics.items()
        }
        print(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f} edge_mse={metric_summary}")

    cfg.out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.out_weights)
    print(f"saved_weights={cfg.out_weights}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval harness for PathwayGNN")
    parser.add_argument("--dataset-jsonl", required=True, type=Path)
    parser.add_argument("--out-weights", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_eval(
        TrainConfig(
            dataset_jsonl=args.dataset_jsonl,
            out_weights=args.out_weights,
            num_epochs=args.epochs,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

