"""
calibrate — JSON in → JSON out entry point for the GNN Calibrator.

This is the single public API surface for the inference package.
The Rust engine calls this; the GNN returns hidden states + audit trace.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import torch

from .audit import AuditTrace, EdgeAttribution, AttentionSummary
from .config import DataConfig, ModelConfig
from .graph import build_graph, GraphMeta
from .model import PathwayGNN

logger = logging.getLogger(__name__)

# Module-level singletons (loaded once, reused across calls)
_graph = None
_meta: GraphMeta | None = None
_model = None


def _ensure_loaded(
    model_path: Path | None = None,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> None:
    """Lazy-load graph and model on first call."""
    global _graph, _meta, _model

    if _graph is None:
        logger.info("Building graph from seed data...")
        _graph, _meta = build_graph(data_cfg or DataConfig())

    if _model is None:
        model_cfg = model_cfg or ModelConfig()
        _model = PathwayGNN(model_cfg)
        if model_path and model_path.exists():
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            _model.load_state_dict(state)
            logger.info("Loaded model weights from %s", model_path)
        else:
            logger.warning("No trained weights — running with random initialisation.")
        _model.eval()


def calibrate(request: dict[str, Any]) -> dict[str, Any]:
    """Run a single calibration inference.

    Parameters
    ----------
    request : dict
        Expected keys:
            wild_type_fluxes : dict   — per-reaction fluxes from wild-type simulation
            perturbed_fluxes : dict   — per-reaction fluxes from SNP-perturbed simulation
            symptoms         : list   — user-reported symptoms (optional)
            genotype         : dict   — SNP variants (optional)
            model_path       : str    — path to trained weights (optional)

    Returns
    -------
    dict with keys:
        calibration_id     : str
        hidden_states      : dict  — per-edge-type hidden states (for audit)
            modulates  : list[dict] — {edge_idx, value, confidence}
            regulates  : list[dict]
            signaling  : list[dict]
            bridges    : list[dict]
        bridge_saturations : dict  — for Rust flux engine
            Maps bridge_id → {"value": f64, "confidence": f64}
            Matches Rust BridgeSaturation struct
        modulates_effects  : dict  — for Rust allosteric modulation
            Maps "metabolite_enzyme" → {"effect": f64, "confidence": f64}
            Matches Rust ModulationEffect struct
        audit              : dict  — full audit trace
    """
    _ensure_loaded(
        model_path=Path(request["model_path"]) if "model_path" in request else None,
    )

    cal_id = str(uuid.uuid4())

    # -- Inject flux delta features into graph (future: implement properly) --
    # For now, use seed features; flux deltas would be added here
    # wild_type = request.get("wild_type_fluxes", {})
    # perturbed = request.get("perturbed_fluxes", {})

    # -- Run forward pass ----------------------------------------------------
    with torch.no_grad():
        out = _model(_graph)

    # -- Build audit trace ---------------------------------------------------
    trace = _build_audit_trace(cal_id, out)

    # -- Format per-edge hidden state results --------------------------------
    hidden_states = {}
    edge_types = [
        ("modulates", "metabolite", "enzyme"),
        ("regulates", "enzyme", "enzyme"),
        ("signaling", "metabolite", "metabolite"),
        ("bridges", "metabolite", "enzyme"),  # Updated: bridges are metabolite→enzyme
    ]

    for etype, stype, dtype in edge_types:
        values = out.get(f"{etype}_hidden", torch.tensor([]))
        confs = out.get(f"{etype}_conf", torch.tensor([]))

        edges = []
        for i in range(values.numel()):
            edges.append({
                "edge_idx": i,
                "value": round(values[i].item(), 6),
                "confidence": round(confs[i].item(), 4) if confs.numel() > i else 0.0,
            })
        hidden_states[etype] = edges

    # -- Build bridge_saturations for Rust consumption -------------------------
    # Maps bridge_id → {value: f64, confidence: f64}
    # Matches Rust BridgeSaturation struct
    bridge_saturations: dict[str, dict[str, float]] = {}
    bridges_hidden = out.get("bridges_hidden", torch.tensor([]))
    bridges_conf = out.get("bridges_conf", torch.tensor([]))

    if _meta is not None:
        for edge_idx in range(len(_meta.bridge_idx_to_id)):
            bridge_id = _meta.bridge_idx_to_id.get(edge_idx)
            if bridge_id and edge_idx < bridges_hidden.numel():
                # Clamp value to [0, 1] — saturation must be non-negative
                sat = bridges_hidden[edge_idx].item()
                conf = bridges_conf[edge_idx].item() if edge_idx < bridges_conf.numel() else 1.0
                bridge_saturations[bridge_id] = {
                    "value": round(max(0.0, min(1.0, sat)), 6),
                    "confidence": round(max(0.0, min(1.0, conf)), 4),
                }

    # -- Build modulates_effects for Rust consumption -------------------------
    # Maps "metabolite_enzyme" → {effect: f64, confidence: f64}
    # Matches Rust ModulationEffect struct
    modulates_effects: dict[str, dict[str, float]] = {}
    modulates_hidden = out.get("modulates_hidden", torch.tensor([]))
    modulates_conf = out.get("modulates_conf", torch.tensor([]))

    if _meta is not None:
        for edge_idx in range(len(_meta.modulates_idx_to_src)):
            src_id = _meta.modulates_idx_to_src.get(edge_idx)
            dst_id = _meta.modulates_idx_to_dst.get(edge_idx)
            if src_id and dst_id and edge_idx < modulates_hidden.numel():
                # Key format: "metabolite_enzyme" (matches Rust lookup)
                key = f"{src_id}_{dst_id}"
                effect = modulates_hidden[edge_idx].item()
                conf = modulates_conf[edge_idx].item() if edge_idx < modulates_conf.numel() else 1.0
                modulates_effects[key] = {
                    "effect": round(max(-1.0, min(1.0, effect)), 6),  # Clamp to [-1, 1]
                    "confidence": round(max(0.0, min(1.0, conf)), 4),
                }

    return {
        "calibration_id": cal_id,
        "hidden_states": hidden_states,
        "bridge_saturations": bridge_saturations,  # For Rust flux engine
        "modulates_effects": modulates_effects,    # For Rust allosteric modulation
        "audit": trace.to_dict(),
    }


def _build_audit_trace(cal_id: str, out: dict[str, Any]) -> AuditTrace:
    """Extract audit information from model outputs."""
    trace = AuditTrace(calibration_id=cal_id)

    # -- Top edges by confidence per type ------------------------------------
    edge_types = ["modulates", "regulates", "signaling", "bridges"]
    all_edges: list[EdgeAttribution] = []

    for etype in edge_types:
        hidden = out.get(f"{etype}_hidden", torch.tensor([]))
        conf = out.get(f"{etype}_conf", torch.tensor([]))

        if conf.numel() == 0:
            continue

        top_k = min(10, conf.numel())
        top_vals, top_idx = torch.topk(conf, top_k)

        for val, idx in zip(top_vals, top_idx):
            all_edges.append(EdgeAttribution(
                edge_type=etype,
                edge_idx=idx.item(),
                hidden_value=hidden[idx].item() if hidden.numel() > idx else 0.0,
                confidence=val.item(),
            ))

    # Sort by confidence, keep top 20
    all_edges.sort(key=lambda e: e.confidence, reverse=True)
    trace.top_edges = all_edges[:20]

    # -- Attention weight summary from GAT layers ----------------------------
    attention_weights = out.get("attention_weights", {})
    for etype in edge_types:
        attn = attention_weights.get(etype)
        if attn is None or attn.numel() == 0:
            continue

        # attn shape: (num_edges, num_heads) — average across heads
        attn_mean = attn.mean(dim=-1)  # (num_edges,)
        top_k = min(10, attn_mean.numel())
        _, top_idx = torch.topk(attn_mean, top_k)

        trace.attention_summary.append(AttentionSummary(
            edge_type=etype,
            num_edges=attn_mean.numel(),
            mean_attention=attn_mean.mean().item(),
            max_attention=attn_mean.max().item(),
            top_edge_indices=[i.item() for i in top_idx],
        ))

    return trace

