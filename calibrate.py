"""
calibrate — JSON in → JSON out entry point for the GNN Calibrator.

This is the single public API surface for the inference package.
The Rust engine calls this; the GNN returns hidden states + audit trace.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
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
_loaded_model_path: Path | None = None
_has_trained_weights = False


@dataclass(frozen=True)
class _InterventionAction:
    """Normalized intervention action for runtime personalization."""

    target_id: str
    sign: float
    weight: float
    mechanism: str


@dataclass(frozen=True)
class _ResolvedIntervention:
    """Intervention resolved against canonical seed IDs."""

    canonical_ids: tuple[str, ...]
    exposure: float


@dataclass
class _InterventionKnowledge:
    """Canonical intervention mapping loaded from seed files."""

    aliases: dict[str, set[str]]
    actions_by_substance: dict[str, list[_InterventionAction]]
    bridge_ids_by_mediator: dict[str, list[str]]
    transport_pairs_by_metabolite: dict[str, list[tuple[str, str]]]
    transport_pairs_by_transporter: dict[str, list[tuple[str, str]]]


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]+", "", text)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _source_strength_weight(value: Any) -> float:
    grade = str(value or "").strip().upper()
    return {
        "A": 1.0,
        "B": 0.85,
        "C": 0.7,
        "D": 0.55,
    }.get(grade, 0.5)


def _action_sign(action: str, mechanism: str) -> float:
    token = f"{action} {mechanism}".upper()
    if any(k in token for k in ("INHIBIT", "ANTAGON", "BLOCK", "NEGATIVE_ALLOSTERIC")):
        return -1.0
    if any(k in token for k in ("ACTIV", "AGON", "INDUC", "POSITIVE_ALLOSTERIC", "UPREGUL")):
        return 1.0
    if "MODULATE" in token:
        return 0.35
    if "BIND" in token:
        return 0.2
    return 0.0


def _route_factor(route: str | None) -> float:
    token = str(route or "").strip().lower()
    if not token:
        return 0.85
    if any(k in token for k in ("iv", "intravenous", "injection", "intramuscular", "subcutaneous")):
        return 1.0
    if any(k in token for k in ("oral", "po")):
        return 0.85
    if any(k in token for k in ("sublingual", "buccal", "intranasal", "nasal", "inhal")):
        return 0.9
    if any(k in token for k in ("topical", "dermal", "transdermal")):
        return 0.6
    return 0.8


def _schedule_factor(schedule: str | None) -> float:
    token = str(schedule or "").strip().lower()
    if not token:
        return 0.85
    if "prn" in token or "as needed" in token:
        return 0.5
    if any(k in token for k in ("qid", "tid", "3x", "4x")):
        return 1.0
    if any(k in token for k in ("bid", "2x", "twice")):
        return 0.92
    if any(k in token for k in ("daily", "qd", "once")):
        return 0.85
    if "weekly" in token:
        return 0.4
    return 0.75


def _dose_factor(dose: Any) -> float:
    try:
        dv = float(dose)
    except (TypeError, ValueError):
        return 0.65
    if dv <= 0:
        return 0.55
    # Log compression keeps very large doses from dominating:
    # factor = tanh(log1p(dose)/4), bounded in [0,1).
    return float(max(0.2, min(1.0, math.tanh(math.log1p(abs(dv)) / 4.0))))


def _recency_factor(last_taken_at: str | None) -> float:
    ts = _parse_iso_datetime(last_taken_at)
    if ts is None:
        return 0.85
    now = datetime.now(timezone.utc)
    hours = max(0.0, (now - ts).total_seconds() / 3600.0)
    # Exponential decay with a 24h half-life:
    # factor = exp(-ln(2) * dt / 24h)
    return float(max(0.05, min(1.0, math.exp(-math.log(2.0) * hours / 24.0))))


def _intervention_exposure(entry: dict[str, Any]) -> float:
    adherence = entry.get("adherence")
    try:
        adherence_f = float(adherence)
    except (TypeError, ValueError):
        adherence_f = 1.0
    adherence_f = max(0.1, min(1.0, adherence_f))
    return float(max(
        0.0,
        min(
            1.0,
            _dose_factor(entry.get("dose"))
            * _route_factor(entry.get("route"))
            * _schedule_factor(entry.get("schedule"))
            * _recency_factor(entry.get("last_taken_at"))
            * adherence_f,
        ),
    ))


@lru_cache(maxsize=1)
def _load_intervention_knowledge(seed_root: str) -> _InterventionKnowledge:
    root = Path(seed_root)
    substances_path = root / "core" / "substances.jsonl"
    modulates_path = root / "edges" / "edges_modulates.jsonl"
    bridges_path = root / "edges" / "bridges.jsonl"
    transport_path = root / "edges" / "edges_transport.jsonl"

    aliases: dict[str, set[str]] = {}
    actions_by_substance: dict[str, list[_InterventionAction]] = {}
    bridge_ids_by_mediator: dict[str, list[str]] = {}
    transport_pairs_by_metabolite: dict[str, list[tuple[str, str]]] = {}
    transport_pairs_by_transporter: dict[str, list[tuple[str, str]]] = {}

    if substances_path.exists():
        with open(substances_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("id", "")).strip()
                if not sid:
                    continue
                canonical = sid.upper()
                keys = [
                    sid,
                    rec.get("name"),
                    rec.get("drugbank_id"),
                    rec.get("chebi_id"),
                    rec.get("pubchem_cid"),
                ]
                synonyms = rec.get("synonyms", [])
                if isinstance(synonyms, list):
                    keys.extend(synonyms)
                for key in keys:
                    nk = _norm_key(key)
                    if nk:
                        aliases.setdefault(nk, set()).add(canonical)
                if canonical.startswith("DRUGBANK_DB"):
                    aliases.setdefault(_norm_key(canonical.replace("DRUGBANK_", "")), set()).add(canonical)

    if modulates_path.exists():
        with open(modulates_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = str(rec.get("substance_id", "")).strip().upper()
                target = str(rec.get("target_id") or rec.get("enzyme_id") or "").strip().upper()
                if not sid or not target:
                    continue
                sign = _action_sign(str(rec.get("action", "")), str(rec.get("mechanism", "")))
                if sign == 0.0:
                    continue
                action = _InterventionAction(
                    target_id=target,
                    sign=sign,
                    weight=_source_strength_weight(rec.get("source_strength")),
                    mechanism=str(rec.get("mechanism", "")).strip().lower(),
                )
                actions_by_substance.setdefault(sid, []).append(action)
                aliases.setdefault(_norm_key(sid), set()).add(sid)

    if bridges_path.exists():
        with open(bridges_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                bridge_id = str(rec.get("bridge_id", "")).strip()
                mediator = str(rec.get("mediator_id", "")).strip().upper()
                if bridge_id and mediator:
                    bridge_ids_by_mediator.setdefault(mediator, []).append(bridge_id)
                    aliases.setdefault(_norm_key(mediator), set()).add(mediator)

    if transport_path.exists():
        with open(transport_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                met = str(rec.get("metabolite_id", "")).strip().upper()
                transporter = str(rec.get("transporter_id", "")).strip().upper()
                from_comp = str(rec.get("from_compartment", "")).strip().lower()
                to_comp = str(rec.get("to_compartment", "")).strip().lower()
                if not met:
                    continue
                if from_comp == "systemic" and to_comp == "brain":
                    pair = (met, f"{met}_BRAIN")
                elif from_comp == "brain" and to_comp == "systemic":
                    pair = (f"{met}_BRAIN", met)
                elif from_comp == "cytosol" and to_comp == "vesicle":
                    pair = (met, met)
                elif from_comp == "vesicle" and to_comp == "cytosol":
                    pair = (met, met)
                else:
                    continue
                transport_pairs_by_metabolite.setdefault(met, []).append(pair)
                if transporter:
                    transport_pairs_by_transporter.setdefault(transporter, []).append(pair)
                aliases.setdefault(_norm_key(met), set()).add(met)
                if transporter:
                    aliases.setdefault(_norm_key(transporter), set()).add(transporter)

    return _InterventionKnowledge(
        aliases=aliases,
        actions_by_substance=actions_by_substance,
        bridge_ids_by_mediator=bridge_ids_by_mediator,
        transport_pairs_by_metabolite=transport_pairs_by_metabolite,
        transport_pairs_by_transporter=transport_pairs_by_transporter,
    )


def _resolve_interventions(
    request: dict[str, Any],
    knowledge: _InterventionKnowledge,
) -> list[_ResolvedIntervention]:
    interventions: list[dict[str, Any]] = []
    meds = request.get("medications")
    sups = request.get("supplements")
    if isinstance(meds, list):
        interventions.extend([m for m in meds if isinstance(m, dict)])
    if isinstance(sups, list):
        interventions.extend([s for s in sups if isinstance(s, dict)])

    resolved: list[_ResolvedIntervention] = []
    for rec in interventions:
        keys = [
            rec.get("id"),
            rec.get("name"),
        ]
        canonical: set[str] = set()
        for key in keys:
            nk = _norm_key(key)
            if not nk:
                continue
            canonical.update(knowledge.aliases.get(nk, set()))
            # Accept raw DrugBank IDs even if not prefixed in seed.
            if nk.startswith("DB"):
                canonical.update(knowledge.aliases.get(_norm_key(f"DRUGBANK_{nk}"), set()))
                canonical.update(knowledge.aliases.get(_norm_key(f"DRUGBANK_{nk.upper()}"), set()))
        if not canonical and rec.get("id"):
            raw = str(rec.get("id", "")).strip().upper()
            if raw:
                canonical.add(raw)
        if not canonical:
            continue
        resolved.append(_ResolvedIntervention(
            canonical_ids=tuple(sorted(canonical)),
            exposure=_intervention_exposure(rec),
        ))
    return resolved


def _ensure_loaded(
    model_path: Path | None = None,
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    require_trained: bool = True,
) -> None:
    """Lazy-load graph and model on first call."""
    global _graph, _meta, _model, _loaded_model_path, _has_trained_weights

    if _graph is None:
        logger.info("Building graph from seed data...")
        _graph, _meta = build_graph(data_cfg or DataConfig())

    reload_model = _model is None or (model_path is not None and model_path != _loaded_model_path)
    if reload_model:
        model_cfg = model_cfg or ModelConfig()
        _model = PathwayGNN(model_cfg)
        _has_trained_weights = False
        _loaded_model_path = model_path
        if model_path and model_path.exists():
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            _model.load_state_dict(state)
            logger.info("Loaded model weights from %s", model_path)
            _has_trained_weights = True
        elif require_trained:
            raise FileNotFoundError(
                "A trained checkpoint is required. Provide request['model_path'] "
                "pointing to an existing weight file."
            )
        else:
            logger.warning("No trained weights — running with random initialisation.")
        _model.eval()
    elif require_trained and not _has_trained_weights:
        raise RuntimeError(
            "Model is loaded without trained weights. Reload with a valid "
            "request['model_path'] checkpoint."
        )


def _apply_request_context(
    g,
    meta: GraphMeta | None,
    request: dict[str, Any],
) -> Any:
    """Inject per-request context features without changing base graph schema.

    We attach a 1-D context channel per node type (`ctx`) and let the model consume
    it via per-type projections. This keeps node feature dimensions stable while
    making inference request-conditioned.
    """
    g = g.clone()

    # Initialize context channels.
    for ntype in g.ntypes:
        g.nodes[ntype].data["ctx"] = torch.zeros((g.num_nodes(ntype), 1), dtype=torch.float32)

    wild_type = request.get("wild_type_fluxes", {}) or {}
    perturbed = request.get("perturbed_fluxes", {}) or {}
    wild_tp = request.get("wild_type_timepoints", []) or []
    pert_tp = request.get("perturbed_timepoints", []) or []
    genotype = request.get("genotype", {}) or {}
    symptoms = request.get("symptoms", []) or []

    # Intervention context (medications/supplements): resolve canonical IDs from
    # seed, estimate exposure, and fold into per-node context.
    # This makes calibrator inference state-dependent before hidden states are
    # consumed by the Rust flux solve.
    try:
        knowledge = _load_intervention_knowledge(str(DataConfig().seed_root))
        resolved = _resolve_interventions(request, knowledge)
    except Exception as exc:  # pragma: no cover - resilience fallback
        logger.warning("intervention context disabled: %s", exc)
        resolved = []
        knowledge = None

    # Reaction-level context from relative flux deltas.
    # delta = (perturbed - wild_type) / max(abs(wild_type), eps), squashed by tanh.
    if meta is not None and hasattr(meta, "rxn_map"):
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
            rxn_ctx[idx, 0] = float(math.tanh(max(-3.0, min(3.0, rel))))

    # Global context from genotype + symptoms, lightly injected into enzyme/metabolite.
    geno_count = len(genotype) if isinstance(genotype, dict) else 0
    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0
    global_ctx = float(math.tanh(0.05 * geno_count + 0.1 * symptom_count))
    if g.num_nodes("enzyme") > 0:
        g.nodes["enzyme"].data["ctx"][:] = global_ctx
    if g.num_nodes("metabolite") > 0:
        g.nodes["metabolite"].data["ctx"][:] = 0.5 * global_ctx

    if meta is not None and resolved and knowledge is not None:
        enz_ctx = g.nodes["enzyme"].data["ctx"]
        met_ctx = g.nodes["metabolite"].data["ctx"]
        for item in resolved:
            if item.exposure <= 0:
                continue
            for sid in item.canonical_ids:
                sid_u = sid.upper()
                # Substance supply/cofactor pressure on metabolite nodes.
                for met_id in (sid_u, f"{sid_u}_BRAIN"):
                    midx = meta.met_map.get(met_id)
                    if midx is not None:
                        met_ctx[midx, 0] += float(0.3 * item.exposure)
                # Pharmacologic action pressure on target enzymes.
                for action in knowledge.actions_by_substance.get(sid_u, []):
                    eidx = meta.enz_map.get(action.target_id)
                    if eidx is not None:
                        delta = 0.4 * item.exposure * action.sign * action.weight
                        enz_ctx[eidx, 0] += float(delta)

    # Trajectory-aware context (optional):
    # If timepoints are provided, inject concentration dynamics summary into
    # metabolite and enzyme context channels. This keeps schema stable and
    # gives calibrator temporal signal without changing model dimensions.
    if isinstance(wild_tp, list) and isinstance(pert_tp, list) and meta is not None:
        wt_last = wild_tp[-1].get("concentrations", {}) if wild_tp else {}
        pt_last = pert_tp[-1].get("concentrations", {}) if pert_tp else {}
        wt_peak: dict[str, float] = {}
        pt_peak: dict[str, float] = {}

        for rows, peak in ((wild_tp, wt_peak), (pert_tp, pt_peak)):
            for tp in rows:
                conc = tp.get("concentrations", {})
                if not isinstance(conc, dict):
                    continue
                for mid, val in conc.items():
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    prev = peak.get(mid, fval)
                    if fval > prev:
                        peak[mid] = fval
                    elif mid not in peak:
                        peak[mid] = fval

        met_ctx = g.nodes["metabolite"].data["ctx"]
        for mid, idx in getattr(meta, "met_map", {}).items():
            wt_end = float(wt_last.get(mid, 0.0)) if isinstance(wt_last, dict) else 0.0
            pt_end = float(pt_last.get(mid, 0.0)) if isinstance(pt_last, dict) else 0.0
            wt_pk = wt_peak.get(mid, wt_end)
            pt_pk = pt_peak.get(mid, pt_end)
            end_rel = (pt_end - wt_end) / (abs(wt_end) + 1e-6)
            peak_rel = (pt_pk - wt_pk) / (abs(wt_pk) + 1e-6)
            dyn_score = math.tanh(max(-4.0, min(4.0, 0.6 * end_rel + 0.4 * peak_rel)))
            met_ctx[idx, 0] = float(dyn_score)

    return g


def _apply_intervention_output_bias(
    out: dict[str, Any],
    meta: GraphMeta | None,
    request: dict[str, Any],
) -> None:
    """Apply intervention effects directly onto edge hidden states.

    This step is executed after model forward and before Rust consumes
    bridge/modulates/transport outputs. It keeps intervention dynamics in the
    simulation path even when learned context features are weak.
    """
    if meta is None:
        return
    try:
        knowledge = _load_intervention_knowledge(str(DataConfig().seed_root))
        resolved = _resolve_interventions(request, knowledge)
    except Exception as exc:  # pragma: no cover
        logger.warning("intervention output bias disabled: %s", exc)
        return
    if not resolved:
        return

    bridge_bias_by_id: dict[str, float] = {}
    enzyme_bias: dict[str, float] = {}
    transport_bias_by_key: dict[str, float] = {}

    for item in resolved:
        if item.exposure <= 0:
            continue
        for sid in item.canonical_ids:
            sid_u = sid.upper()
            for bridge_id in knowledge.bridge_ids_by_mediator.get(sid_u, []):
                bridge_bias_by_id[bridge_id] = bridge_bias_by_id.get(bridge_id, 0.0) + 0.6 * item.exposure
            for action in knowledge.actions_by_substance.get(sid_u, []):
                enzyme_bias[action.target_id] = (
                    enzyme_bias.get(action.target_id, 0.0) + item.exposure * action.sign * action.weight
                )
            for src_id, dst_id in knowledge.transport_pairs_by_metabolite.get(sid_u, []):
                key = f"{src_id}->{dst_id}"
                transport_bias_by_key[key] = transport_bias_by_key.get(key, 0.0) + 0.4 * item.exposure
            for src_id, dst_id in knowledge.transport_pairs_by_transporter.get(sid_u, []):
                key = f"{src_id}->{dst_id}"
                transport_bias_by_key[key] = transport_bias_by_key.get(key, 0.0) + 0.55 * item.exposure

    # Bridges: saturation shift in [0, 1].
    bridges_hidden = out.get("bridges_hidden")
    if bridges_hidden is not None and bridges_hidden.numel() > 0:
        for edge_idx, bridge_id in meta.bridge_idx_to_id.items():
            if edge_idx >= bridges_hidden.numel():
                continue
            bias = bridge_bias_by_id.get(bridge_id, 0.0)
            if abs(bias) < 1e-9:
                continue
            bridges_hidden[edge_idx] = torch.clamp(bridges_hidden[edge_idx] + 0.20 * bias, 0.0, 1.0)

    # Modulates: apply enzyme-level pharmacology to edges targeting each enzyme.
    mod_hidden = out.get("modulates_hidden")
    if mod_hidden is not None and mod_hidden.numel() > 0:
        for edge_idx, target_id in meta.modulates_idx_to_dst.items():
            if edge_idx >= mod_hidden.numel():
                continue
            bias = enzyme_bias.get(target_id, 0.0)
            if abs(bias) < 1e-9:
                continue
            mod_hidden[edge_idx] = torch.clamp(mod_hidden[edge_idx] + 0.30 * bias, -1.0, 1.0)

    # Transport: multiplier shift in [0, 2].
    trans_hidden = out.get("transports_to_hidden")
    if trans_hidden is not None and trans_hidden.numel() > 0:
        for edge_idx, pair in meta.transport_idx_to_pair.items():
            if edge_idx >= trans_hidden.numel():
                continue
            src_id, dst_id = pair
            bias = transport_bias_by_key.get(f"{src_id}->{dst_id}", 0.0)
            if abs(bias) < 1e-9:
                continue
            trans_hidden[edge_idx] = torch.clamp(trans_hidden[edge_idx] + 0.35 * bias, 0.0, 2.0)


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
            require_trained  : bool   — fail if checkpoint missing (default True)

    Returns
    -------
    dict with keys:
        calibration_id     : str
        hidden_states      : dict  — per-edge-type hidden states (for audit)
            modulates  : list[dict] — {edge_idx, value, confidence}
            regulates  : list[dict]
            signaling  : list[dict]
            bridges    : list[dict]
            transports_to : list[dict]
        bridge_saturations : dict  — for Rust flux engine
            Maps bridge_id → {"value": f64, "confidence": f64}
            Matches Rust BridgeSaturation struct
        modulates_effects  : dict  — for Rust allosteric modulation
            Maps "metabolite_enzyme" → {"effect": f64, "confidence": f64}
            Matches Rust ModulationEffect struct
        transport_effects  : dict  — for compartment transport
            Maps "src->dst" → {"multiplier": f64, "confidence": f64}
        audit              : dict  — full audit trace
    """
    _ensure_loaded(
        model_path=Path(request["model_path"]) if "model_path" in request else None,
        require_trained=bool(request.get("require_trained", True)),
    )

    cal_id = str(uuid.uuid4())

    g = _apply_request_context(_graph, _meta, request)

    # -- Run forward pass ----------------------------------------------------
    with torch.no_grad():
        out = _model(g)
    _apply_intervention_output_bias(out, _meta, request)

    # -- Build audit trace ---------------------------------------------------
    trace = _build_audit_trace(cal_id, out)

    # -- Format per-edge hidden state results --------------------------------
    hidden_states = {}
    edge_types = [
        ("modulates", "metabolite", "enzyme"),
        ("regulates", "enzyme", "enzyme"),
        ("signaling", "metabolite", "metabolite"),
        ("bridges", "metabolite", "enzyme"),  # Updated: bridges are metabolite→enzyme
        ("transports_to", "metabolite", "metabolite"),
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

    # -- Build transport_effects for compartmentalization ----------------------
    # Maps "src_metabolite->dst_metabolite" -> {multiplier, confidence}
    transport_effects: dict[str, dict[str, float]] = {}
    transports_hidden = out.get("transports_to_hidden", torch.tensor([]))
    transports_conf = out.get("transports_to_conf", torch.tensor([]))
    if _meta is not None:
        for edge_idx in range(len(_meta.transport_idx_to_pair)):
            pair = _meta.transport_idx_to_pair.get(edge_idx)
            if pair is None or edge_idx >= transports_hidden.numel():
                continue
            src_id, dst_id = pair
            key = f"{src_id}->{dst_id}"
            mult = transports_hidden[edge_idx].item()
            conf = transports_conf[edge_idx].item() if edge_idx < transports_conf.numel() else 1.0
            transport_effects[key] = {
                "multiplier": round(max(0.0, min(2.0, mult)), 6),
                "confidence": round(max(0.0, min(1.0, conf)), 4),
            }

    return {
        "calibration_id": cal_id,
        "hidden_states": hidden_states,
        "bridge_saturations": bridge_saturations,  # For Rust flux engine
        "modulates_effects": modulates_effects,    # For Rust allosteric modulation
        "transport_effects": transport_effects,    # For compartment transport
        "audit": trace.to_dict(),
    }


def _build_audit_trace(cal_id: str, out: dict[str, Any]) -> AuditTrace:
    """Extract audit information from model outputs."""
    trace = AuditTrace(calibration_id=cal_id)

    # -- Top edges by confidence per type ------------------------------------
    edge_types = ["modulates", "regulates", "signaling", "bridges", "transports_to"]
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
