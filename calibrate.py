"""
calibrate — JSON in → JSON out entry point for the GNN Calibrator.

This is the single public API surface for the inference package.
The Rust engine calls this; the GNN returns hidden states + audit trace.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .audit import AuditTrace, EdgeAttribution, AttentionSummary, ResolvedEdge
from .config import DataConfig, ModelConfig
from .graph import build_graph, GraphMeta
from .model import PathwayGNN
from .observation_features import OBSERVATION_FEATURE_DIM, build_observation_feature_vector, canonical_observation_lookup

logger = logging.getLogger(__name__)

WHY_TODAY_LATENT_NAMES: tuple[str, ...] = (
    "histamine_load_today",
    "catecholamine_clearance_stress",
    "glutamate_gaba_instability",
    "methyl_donor_strain",
    "infection_like_sickness_state",
    "circadian_disruption",
    "sleep_pressure_disruption",
    "absorption_impairment",
    "acute_cofactor_depletion",
    "transport_constraint_state",
    "intervention_rebound_or_withdrawal",
    "stress_hormone_load",
)

# Module-level singletons (loaded once, reused across calls)
_graph = None
_meta: GraphMeta | None = None
_model = None
_loaded_model_path: Path | None = None
_has_trained_weights = False
_ROOT = Path(__file__).resolve().parents[1]
_GLUGABA_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "glugaba_v1" / "glugaba_obs_only.pkl"
_CATECHOL_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "catechol_v1" / "catechol_obs_only.pkl"
_TRANSPORT_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "transport_v1" / "transport_obs_only.pkl"
_INFECTION_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "infection_v1_balanced" / "infection_obs_only.pkl"
_HISTAMINE_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "histamine_load_today" / "histamine_load_today_obs_only.pkl"
_STRESS_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "stress_hormone_load" / "stress_hormone_load_obs_only.quick12k.pkl"
_SLEEP_PRESSURE_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "sleep_pressure_disruption" / "sleep_pressure_disruption_obs_only.quick12k.pkl"
_CIRCADIAN_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "circadian_disruption" / "circadian_disruption_obs_only.quick12k.pkl"
_METHYL_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "methyl_donor_strain" / "methyl_donor_strain_obs_only.quick12k.pkl"
_ABSORPTION_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "absorption_impairment" / "absorption_impairment_obs_only.quick12k.pkl"
_COFACTOR_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "acute_cofactor_depletion" / "acute_cofactor_depletion_obs_only.quick12k.pkl"
_REBOUND_TEACHER_PATH = _ROOT / "data-experiments" / "raw" / "training" / "mvp_run_20260314" / "specialists" / "intervention_rebound_or_withdrawal" / "intervention_rebound_or_withdrawal_obs_only.quick12k.pkl"


def _load_state_dict_shape_safe(model: PathwayGNN, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(filtered, strict=False)


@lru_cache(maxsize=1)
def _load_glugaba_teacher(path_str: str) -> Any | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        blob = pickle.load(path.open("rb"))
    except Exception as exc:
        logger.warning("Failed to load glutamate teacher from %s: %s", path, exc)
        return None
    pipeline = blob.get("pipeline") if isinstance(blob, dict) else None
    return pipeline


def _predict_glugaba_teacher(request: dict[str, Any]) -> float | None:
    pipeline = _load_glugaba_teacher(str(_GLUGABA_TEACHER_PATH))
    if pipeline is None:
        return None
    feats = build_observation_feature_vector(request)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = pipeline.predict([feats])
        if len(raw) <= 0:
            return None
        return max(0.0, min(1.0, float(raw[0])))
    except Exception as exc:
        logger.warning("Glutamate teacher prediction failed: %s", exc)
        return None


def _predict_specialist_teacher(request: dict[str, Any], path: Path, label: str) -> float | None:
    pipeline = _load_glugaba_teacher(str(path))
    if pipeline is None:
        return None
    feats = build_observation_feature_vector(request)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = pipeline.predict([feats])
        if len(raw) <= 0:
            return None
        return max(0.0, min(1.0, float(raw[0])))
    except Exception as exc:
        logger.warning("%s teacher prediction failed: %s", label, exc)
        return None


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


@dataclass(frozen=True)
class _SpecialistSpec:
    latent_id: str
    owner_mode: str
    status: str
    support_profile: str
    activation_threshold: float
    confidence_floor: float
    blend_bias: float = 0.35
    blend_support_weight: float = 0.45
    blend_score_weight: float = 0.20
    teacher_path: Path | None = None
    teacher_label: str | None = None
    teacher_weight: float = 1.0
    demotes: tuple[str, ...] = ()


def _norm_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]+", "", text)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float, high: float) -> float:
    return max(low, min(high, _safe_float(value)))


def _clamp01(value: Any) -> float:
    return _clamp(value, 0.0, 1.0)


def _specialist_registry() -> dict[str, _SpecialistSpec]:
    return {
        "histamine_load_today": _SpecialistSpec(
            latent_id="histamine_load_today",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="histamine",
            activation_threshold=0.56,
            confidence_floor=0.58,
            teacher_path=_HISTAMINE_TEACHER_PATH,
            teacher_label="Histamine",
            demotes=("infection_like_sickness_state",),
        ),
        "stress_hormone_load": _SpecialistSpec(
            latent_id="stress_hormone_load",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="stress",
            activation_threshold=0.56,
            confidence_floor=0.58,
            teacher_path=_STRESS_TEACHER_PATH,
            teacher_label="Stress",
            teacher_weight=0.90,
        ),
        "sleep_pressure_disruption": _SpecialistSpec(
            latent_id="sleep_pressure_disruption",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="sleep_pressure",
            activation_threshold=0.54,
            confidence_floor=0.56,
            teacher_path=_SLEEP_PRESSURE_TEACHER_PATH,
            teacher_label="SleepPressure",
        ),
        "circadian_disruption": _SpecialistSpec(
            latent_id="circadian_disruption",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="circadian",
            activation_threshold=0.54,
            confidence_floor=0.56,
            teacher_path=_CIRCADIAN_TEACHER_PATH,
            teacher_label="Circadian",
        ),
        "methyl_donor_strain": _SpecialistSpec(
            latent_id="methyl_donor_strain",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="methyl",
            activation_threshold=0.52,
            confidence_floor=0.54,
            teacher_path=_METHYL_TEACHER_PATH,
            teacher_label="Methyl",
        ),
        "absorption_impairment": _SpecialistSpec(
            latent_id="absorption_impairment",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="absorption",
            activation_threshold=0.54,
            confidence_floor=0.56,
            teacher_path=_ABSORPTION_TEACHER_PATH,
            teacher_label="Absorption",
        ),
        "acute_cofactor_depletion": _SpecialistSpec(
            latent_id="acute_cofactor_depletion",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="cofactor",
            activation_threshold=0.52,
            confidence_floor=0.54,
            teacher_path=_COFACTOR_TEACHER_PATH,
            teacher_label="Cofactor",
        ),
        "intervention_rebound_or_withdrawal": _SpecialistSpec(
            latent_id="intervention_rebound_or_withdrawal",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="rebound",
            activation_threshold=0.52,
            confidence_floor=0.54,
            teacher_path=_REBOUND_TEACHER_PATH,
            teacher_label="Rebound",
            teacher_weight=0.90,
        ),
        "glutamate_gaba_instability": _SpecialistSpec(
            latent_id="glutamate_gaba_instability",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="glutamate",
            activation_threshold=0.58,
            confidence_floor=0.62,
            teacher_path=_GLUGABA_TEACHER_PATH,
            teacher_label="Glutamate",
        ),
        "catecholamine_clearance_stress": _SpecialistSpec(
            latent_id="catecholamine_clearance_stress",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="catechol",
            activation_threshold=0.60,
            confidence_floor=0.62,
            teacher_path=_CATECHOL_TEACHER_PATH,
            teacher_label="Catecholamine",
        ),
        "transport_constraint_state": _SpecialistSpec(
            latent_id="transport_constraint_state",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="transport",
            activation_threshold=0.56,
            confidence_floor=0.60,
            teacher_path=_TRANSPORT_TEACHER_PATH,
            teacher_label="Transport",
        ),
        "infection_like_sickness_state": _SpecialistSpec(
            latent_id="infection_like_sickness_state",
            owner_mode="primary_specialist",
            status="active_teacher",
            support_profile="infection",
            activation_threshold=0.62,
            confidence_floor=0.65,
            teacher_path=_INFECTION_TEACHER_PATH,
            teacher_label="Infection",
            teacher_weight=0.85,
            demotes=("stress_hormone_load",),
        ),
    }


def _merge_numeric_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for src in (base, extra):
        for key, value in src.items():
            if isinstance(value, (int, float)):
                merged[str(key)] = float(value)
    return merged


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


@lru_cache(maxsize=1)
def _load_symptom_entity_map(seed_root: str) -> dict[str, set[str]]:
    """Build symptom_id → {enzyme/metabolite IDs} from seed phenotype data.

    Uses the chain: effector_symptom_links → effector_states to resolve
    which enzymes and metabolites are mechanistically linked to each symptom.
    """
    root = Path(seed_root)
    links_path = root / "phenotype" / "effector_symptom_links.jsonl"
    states_path = root / "phenotype" / "effector_states.jsonl"

    # Step 1: Load effector_states → mechanism_ids + aliases
    effector_to_entities: dict[str, set[str]] = {}
    if states_path.exists():
        with open(states_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                state_id = rec.get("effector_state_id", "")
                if not state_id:
                    continue
                entities: set[str] = set()
                # mechanism_ids are enzyme/gene IDs
                for mid in rec.get("mechanism_ids", []):
                    if isinstance(mid, str) and mid.strip():
                        entities.add(mid.strip())
                # aliases include metabolite IDs
                for alias in rec.get("aliases", []):
                    if isinstance(alias, str) and alias.strip():
                        # Strip compartment suffixes like @brain, @systemic
                        clean = alias.split("@")[0].strip()
                        if clean:
                            entities.add(clean)
                effector_to_entities[state_id] = entities

    # Step 2: Load effector_symptom_links → symptom_id → effector_state_ids
    symptom_map: dict[str, set[str]] = {}
    if links_path.exists():
        with open(links_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sym_id = str(rec.get("symptom_id", "")).strip().lower()
                state_id = rec.get("effector_state_id", "")
                if not sym_id or not state_id:
                    continue
                entities = effector_to_entities.get(state_id, set())
                if entities:
                    symptom_map.setdefault(sym_id, set()).update(entities)

    logger.info(
        "Loaded symptom→entity map: %d symptoms, %d effector states",
        len(symptom_map), len(effector_to_entities),
    )
    return symptom_map


@lru_cache(maxsize=1)
def _load_lifestyle_entity_map(seed_root: str) -> tuple[
    dict[str, dict[str, float]],  # factor_id → {enzyme_id: direction_sign}
    dict[str, str],               # domain → factor_id (from lifestyle_factors.jsonl)
]:
    """Build lifestyle factor → enzyme mappings from seed data.

    Uses two seed files:
    - lifestyle_factors.jsonl: domain → factor_id mapping
    - lifestyle_expression_effects.jsonl: factor_id → target_id with direction

    Returns:
        factor_map: factor_id → {target_id: direction_float}
        domain_to_factor: domain → factor_id (for resolving request keys)
    """
    root = Path(seed_root)
    factors_path = root / "core" / "lifestyle_factors.jsonl"
    effects_path = root / "kinetics" / "lifestyle_expression_effects.jsonl"

    # Step 1: Load domain → factor_id from lifestyle_factors.jsonl
    domain_to_factor: dict[str, str] = {}
    if factors_path.exists():
        with open(factors_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                domain = rec.get("domain", "").strip().lower()
                factor_id = rec.get("factor_id", "").strip()
                if domain and factor_id:
                    domain_to_factor[domain] = factor_id

    # Step 2: Load factor_id → {target_id: direction} from effects
    factor_map: dict[str, dict[str, float]] = {}
    if effects_path.exists():
        direction_sign = {"increased": 1.0, "decreased": -1.0}
        with open(effects_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                factor_id = rec.get("factor_id", "")
                target_id = rec.get("target_id", "")
                direction = rec.get("direction", "unknown")
                if not factor_id or not target_id:
                    continue
                sign = direction_sign.get(direction, 0.0)
                if sign == 0.0:
                    continue
                targets = factor_map.setdefault(factor_id, {})
                targets[target_id] = targets.get(target_id, 0.0) + sign
    else:
        logger.info("No lifestyle_expression_effects.jsonl found at %s", effects_path)

    logger.info(
        "Loaded lifestyle entity map: %d factors (%d domains), %d total targets",
        len(factor_map), len(domain_to_factor),
        sum(len(v) for v in factor_map.values()),
    )
    return factor_map, domain_to_factor


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
            _load_state_dict_shape_safe(_model, state)
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

    We attach an 8-D context vector per node type (`ctx`) and let the model
    consume it via per-type projections. This keeps node feature dimensions
    stable while making inference request-conditioned.

    Context channels:
        [0] flux delta (existing reaction-level signal)
        [1] symptom count (normalized)
        [2] symptom severity hash (category-weighted)
        [3] symptom category signal (per-node relevance from seed)
        [4] lifestyle enzyme effects (per-node from seed targets)
        [5] lifestyle burden (global, normalized factor count)
        [6] intervention exposure aggregate
        [7] genotype burden (SNP count, weighted)
    """
    g = g.clone()

    # Initialize 8-dim context channels (zeros = no signal, not noise).
    for ntype in g.ntypes:
        g.nodes[ntype].data["ctx"] = torch.zeros((g.num_nodes(ntype), 8), dtype=torch.float32)
        g.nodes[ntype].data["obs"] = torch.zeros((g.num_nodes(ntype), OBSERVATION_FEATURE_DIM), dtype=torch.float32)

    dag_prior_packet = request.get("dag_prior_packet", {}) or {}
    observation_packet = request.get("observation_packet", {}) or {}
    wild_type = request.get("wild_type_fluxes", {}) or {}
    perturbed = request.get("perturbed_fluxes", {}) or {}
    wild_tp = request.get("wild_type_timepoints", []) or []
    pert_tp = request.get("perturbed_timepoints", []) or []
    genotype = request.get("genotype", {}) or {}
    symptoms = observation_packet.get("symptoms", request.get("symptoms", [])) or []
    lifestyle = request.get("lifestyle", {}) or {}
    wearables = _extract_wearable_lookup(observation_packet, request)
    packet_context = observation_packet.get("context", {}) or {}
    context = _merge_numeric_dict(request.get("context", {}) or {}, packet_context)
    observation_vec = torch.tensor(build_observation_feature_vector(request), dtype=torch.float32)
    for ntype in g.ntypes:
        if g.num_nodes(ntype) > 0:
            g.nodes[ntype].data["obs"][:, :] = observation_vec.unsqueeze(0)

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

    # -- Channel 0: Reaction-level context from relative flux deltas ----------
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

    # -- Channels 1-3: Symptom burden encoding --------------------------------
    symptom_count = len(symptoms) if isinstance(symptoms, list) else 0
    # Channel 1: normalized symptom count (tanh-squashed)
    sym_count_signal = float(math.tanh(0.1 * symptom_count))

    # Channel 2: severity hash — weighted sum of severity values
    severity_map = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}
    severity_sum = 0.0
    symptom_categories: list[str] = []
    if isinstance(symptoms, list):
        for sym in symptoms:
            if isinstance(sym, dict):
                if sym.get("severity_0_1") is not None:
                    severity_sum += max(0.0, min(1.0, _safe_float(sym.get("severity_0_1"), 0.5)))
                elif sym.get("value") is not None:
                    severity_sum += max(0.0, min(1.0, _safe_float(sym.get("value"), 5.0) / 10.0))
                else:
                    sev = str(sym.get("severity", "moderate")).lower()
                    severity_sum += severity_map.get(sev, 0.5)
                cat = sym.get("category", "")
                if cat:
                    symptom_categories.append(str(cat).lower())
            elif isinstance(sym, str):
                severity_sum += 0.5  # default moderate for string-only symptoms
    sym_severity_signal = float(math.tanh(0.2 * severity_sum))

    # Channel 3: per-node symptom relevance from seed data.
    # Uses effector_symptom_links → effector_states chain to resolve
    # symptom_id → mechanism_ids (enzymes) + aliases (metabolites).
    symptom_entity_map = _load_symptom_entity_map(str(DataConfig().seed_root))
    sym_node_relevance_enz: dict[int, float] = {}
    sym_node_relevance_met: dict[int, float] = {}
    if meta is not None and isinstance(symptoms, list):
        for sym in symptoms:
            sym_id = ""
            sym_weight = 1.0
            if isinstance(sym, dict):
                sym_id = str(sym.get("symptom_id", sym.get("id", sym.get("name", "")))).strip().lower()
                if sym.get("severity_0_1") is not None:
                    sym_weight = max(0.0, min(1.0, _safe_float(sym.get("severity_0_1"), 0.5)))
                elif sym.get("value") is not None:
                    sym_weight = max(0.0, min(1.0, _safe_float(sym.get("value"), 5.0) / 10.0))
                else:
                    sev = str(sym.get("severity", "moderate")).lower()
                    sym_weight = severity_map.get(sev, 0.5)
            elif isinstance(sym, str):
                sym_id = sym.strip().lower()
            if not sym_id:
                continue
            entity_ids = symptom_entity_map.get(sym_id, set())
            for eid in entity_ids:
                eid_upper = eid.upper()
                eidx = meta.enz_map.get(eid_upper)
                if eidx is not None:
                    sym_node_relevance_enz[eidx] = (
                        sym_node_relevance_enz.get(eidx, 0.0) + 0.3 * sym_weight
                    )
                midx = meta.met_map.get(eid_upper)
                if midx is not None:
                    sym_node_relevance_met[midx] = (
                        sym_node_relevance_met.get(midx, 0.0) + 0.3 * sym_weight
                    )

    # -- Channels 4-5: Lifestyle factors (per-node from seed data) -------------
    # Channel 4: per-node lifestyle enzyme effects (aggregated across all
    #            reported lifestyle factors, resolved via seed enzyme targets)
    # Channel 5: global lifestyle burden (how many factors reported, normalized)
    #
    # Resolution chain (all from seed):
    #   request key → domain (lifestyle_factors.jsonl) → factor_id
    #   factor_id → target enzymes with direction (lifestyle_expression_effects.jsonl)
    factor_map, domain_to_factor = _load_lifestyle_entity_map(str(DataConfig().seed_root))

    lifestyle_burden = 0.0
    lifestyle_node_enz_signals: dict[int, float] = {}  # eidx → aggregated signal
    if meta is not None and isinstance(lifestyle, dict):
        for lf_key, lf_val in lifestyle.items():
            try:
                intensity = float(max(0.0, min(1.0, float(lf_val))))
            except (TypeError, ValueError):
                continue
            if intensity <= 0:
                continue
            # Resolve request key → domain → factor_id via seed.
            # Request keys should match seed domain names (e.g. "sleep",
            # "stress", "physical_activity"). If not found, this is a
            # data curation gap — not papered over with hardcoded mappings.
            factor_id = domain_to_factor.get(lf_key)
            if not factor_id:
                logger.debug("Lifestyle key %r not in seed domains; skipping", lf_key)
                continue
            lifestyle_burden += intensity
            targets = factor_map.get(factor_id, {})
            for target_id, direction_sign in targets.items():
                eidx = meta.enz_map.get(target_id)
                if eidx is not None:
                    lifestyle_node_enz_signals[eidx] = (
                        lifestyle_node_enz_signals.get(eidx, 0.0)
                        + intensity * direction_sign * 0.3
                    )
    lifestyle_burden_signal = float(math.tanh(0.2 * lifestyle_burden))

    # -- Channel 7: Genotype burden -------------------------------------------
    geno_count = len(genotype) if isinstance(genotype, dict) else 0
    geno_burden = float(math.tanh(0.05 * geno_count))
    sleep_hours = _safe_float(
        wearables.get(
            "sleep_duration_hours",
            context.get("sleep_duration_hours", context.get("sleep_duration_h")),
        ),
        0.0,
    )
    hrv = _safe_float(
        wearables.get("hrv_rmssd_ms", wearables.get("hrv", context.get("hrv"))),
        0.0,
    )
    hrv_delta = _safe_float(
        wearables.get("hrv_rmssd_delta_from_baseline_pct"),
        0.0,
    )
    resting_hr = _safe_float(
        wearables.get(
            "resting_heart_rate_bpm",
            wearables.get("resting_heart_rate", context.get("resting_heart_rate")),
        ),
        0.0,
    )
    rhr_delta = _safe_float(
        wearables.get("resting_heart_rate_delta_from_baseline_bpm"),
        0.0,
    )
    readiness = _safe_float(
        wearables.get("readiness_score_0_100", wearables.get("readiness", context.get("readiness"))),
        0.0,
    )
    temp_dev = _safe_float(
        wearables.get(
            "temperature_deviation_c",
            wearables.get("temperature_deviation", context.get("temperature_deviation")),
        ),
        0.0,
    )
    stress_load = _safe_float(
        context.get("stress_load", context.get("stress_level_0_10", context.get("stress"))),
        0.0,
    )
    late_caffeine = _safe_float(
        context.get(
            "late_caffeine",
            context.get("caffeine_late", context.get("caffeine_after_2pm_days_7d")),
        ),
        0.0,
    )
    prior_solver = dag_prior_packet.get("solver", {}) if isinstance(dag_prior_packet, dict) else {}
    prior_iterations = 0.0
    if isinstance(prior_solver, dict):
        prior_iterations = _safe_float(prior_solver.get("prior_iterations"), 0.0)
    prior_iteration_signal = float(math.tanh(0.05 * prior_iterations))
    wearable_burden = float(
        math.tanh(
            max(0.0, 6.5 - sleep_hours) * 0.35
            + max(0.0, (-hrv_delta) / 35.0)
            + max(0.0, rhr_delta / 8.0)
            + (0.4 * max(0.0, (55.0 - hrv) / 25.0) if abs(hrv_delta) < 1e-6 else 0.0)
            + (0.4 * max(0.0, (resting_hr - 70.0) / 18.0) if abs(rhr_delta) < 1e-6 else 0.0)
            + max(0.0, abs(temp_dev))
            + max(0.0, (0.6 - readiness) * 1.5)
            + max(0.0, stress_load)
            + max(0.0, late_caffeine),
        )
    )

    # -- Apply global channels to enzyme/metabolite nodes ---------------------
    if g.num_nodes("enzyme") > 0:
        enz_ctx = g.nodes["enzyme"].data["ctx"]
        enz_ctx[:, 1] = sym_count_signal
        enz_ctx[:, 2] = sym_severity_signal
        enz_ctx[:, 5] = torch.clamp(
            enz_ctx[:, 5] + lifestyle_burden_signal + 0.5 * wearable_burden,
            -1.0,
            1.0,
        )
        enz_ctx[:, 6] = torch.clamp(enz_ctx[:, 6] + prior_iteration_signal, -1.0, 1.0)
        enz_ctx[:, 7] = torch.clamp(
            enz_ctx[:, 7] + geno_burden + 0.5 * stress_load,
            -1.0,
            1.0,
        )
        # Per-node symptom relevance (channel 3)
        for eidx, val in sym_node_relevance_enz.items():
            if eidx < enz_ctx.shape[0]:
                enz_ctx[eidx, 3] = float(math.tanh(val))
        # Per-node lifestyle enzyme effects (channel 4)
        for eidx, val in lifestyle_node_enz_signals.items():
            if eidx < enz_ctx.shape[0]:
                enz_ctx[eidx, 4] = float(math.tanh(val))

    if g.num_nodes("metabolite") > 0:
        met_ctx = g.nodes["metabolite"].data["ctx"]
        met_ctx[:, 1] = sym_count_signal
        met_ctx[:, 2] = sym_severity_signal
        met_ctx[:, 5] = torch.clamp(
            met_ctx[:, 5] + lifestyle_burden_signal + wearable_burden,
            -1.0,
            1.0,
        )
        met_ctx[:, 6] = torch.clamp(met_ctx[:, 6] + 0.5 * prior_iteration_signal, -1.0, 1.0)
        met_ctx[:, 7] = torch.clamp(
            met_ctx[:, 7] + 0.5 * geno_burden + 0.3 * stress_load,
            -1.0,
            1.0,
        )
        # Per-node symptom relevance (channel 3)
        for midx, val in sym_node_relevance_met.items():
            if midx < met_ctx.shape[0]:
                met_ctx[midx, 3] = float(math.tanh(val))

    # -- Channel 6: Intervention exposure aggregate ---------------------------
    if meta is not None and resolved and knowledge is not None:
        enz_ctx = g.nodes["enzyme"].data["ctx"]
        met_ctx = g.nodes["metabolite"].data["ctx"]
        for item in resolved:
            if item.exposure <= 0:
                continue
            for sid in item.canonical_ids:
                sid_u = sid.upper()
                # Substance supply/cofactor pressure on metabolite nodes (ch 0 + ch 6).
                for met_id in (sid_u, f"{sid_u}_BRAIN"):
                    midx = meta.met_map.get(met_id)
                    if midx is not None:
                        met_ctx[midx, 0] += float(0.3 * item.exposure)
                        met_ctx[midx, 6] += float(0.4 * item.exposure)
                # Pharmacologic action pressure on target enzymes (ch 0 + ch 6).
                for action in knowledge.actions_by_substance.get(sid_u, []):
                    eidx = meta.enz_map.get(action.target_id)
                    if eidx is not None:
                        delta = 0.4 * item.exposure * action.sign * action.weight
                        enz_ctx[eidx, 0] += float(delta)
                        enz_ctx[eidx, 6] += float(abs(delta))

    # Trajectory-aware context (optional):
    # If timepoints are provided, inject concentration dynamics summary into
    # metabolite context channel 0. This keeps schema stable and
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


def _latent_entry(value: float, confidence: float) -> dict[str, float]:
    return {
        "value": round(max(0.0, min(1.0, value)), 6),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


@lru_cache(maxsize=1)
def _load_why_today_mapping_registry(seed_root: str) -> list[dict[str, Any]]:
    root = Path(seed_root)
    candidates = [
        root / "phenotype" / "why_today_mapping_registry.jsonl",
        root.parent / "phenotype" / "why_today_mapping_registry.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if rows:
            break
    return rows


def _extract_wearable_lookup(observation_packet: dict[str, Any], request: dict[str, Any]) -> dict[str, float]:
    wearables = observation_packet.get("wearables", request.get("wearables", {})) or {}
    wearable_norm = (
        observation_packet.get("wearable_normalized", {})
        or observation_packet.get("wearable_normalized_packet", {})
        or {}
    )

    lookup = _merge_numeric_dict(wearables if isinstance(wearables, dict) else {}, {})
    if isinstance(wearable_norm, dict):
        norm_feats = wearable_norm.get("normalized_features", wearable_norm)
        if isinstance(norm_feats, dict):
            lookup.update(_merge_numeric_dict(norm_feats, {}))

    alias_pairs = {
        "hrv": "hrv_rmssd_ms",
        "resting_heart_rate": "resting_heart_rate_bpm",
        "temperature_deviation": "temperature_deviation_c",
        "readiness": "readiness_score_0_100",
        "late_caffeine": "caffeine_after_2pm_days_7d",
        "caffeine_late": "caffeine_after_2pm_days_7d",
        "alcohol_load": "alcohol_days_7d",
        "alcohol": "alcohol_days_7d",
        "high_histamine_meal": "high_histamine_meals_days_7d",
        "high_histamine_meals": "high_histamine_meals_days_7d",
        "gi_burden": "gi_symptom_burden_0_10",
        "gi_symptom_load": "gi_symptom_burden_0_10",
        "stress_load": "stress_level_0_10",
        "stress": "stress_level_0_10",
        "exercise_recovery": "exercise_recovery_1_5",
        "activity_strain": "activity_strain_score_0_100",
    }
    for src, dst in alias_pairs.items():
        if src in lookup and dst not in lookup:
            lookup[dst] = lookup[src]
    return lookup


def _build_observation_lookup(request: dict[str, Any]) -> dict[str, float]:
    observation_packet = request.get("observation_packet", {}) or {}
    symptoms = observation_packet.get("symptoms", request.get("symptoms", [])) or []
    medications = observation_packet.get("medications", request.get("medications", [])) or []
    supplements = observation_packet.get("supplements", request.get("supplements", [])) or []
    lookup = canonical_observation_lookup(request)

    def _avg_adherence(rows: list[Any]) -> float | None:
        vals = [
            max(0.0, min(1.0, _safe_float(r.get("adherence"), 1.0)))
            for r in rows
            if isinstance(r, dict) and r.get("adherence") is not None
        ]
        if not vals:
            return None
        return (sum(vals) / len(vals)) * 100.0

    med_adherence = _avg_adherence(list(medications))
    supp_adherence = _avg_adherence(list(supplements))
    if med_adherence is not None:
        lookup["med_adherence_pct_7d"] = med_adherence
    if supp_adherence is not None:
        lookup["supplement_adherence_pct_7d"] = supp_adherence
    lookup["symptom_count"] = float(len(symptoms)) if isinstance(symptoms, list) else 0.0
    return lookup


def _has_supporting_channel(channel: str, request: dict[str, Any], lookup: dict[str, float]) -> bool:
    observation_packet = request.get("observation_packet", {}) or {}
    symptoms = observation_packet.get("symptoms", request.get("symptoms", [])) or []
    medications = observation_packet.get("medications", request.get("medications", [])) or []
    supplements = observation_packet.get("supplements", request.get("supplements", [])) or []
    if channel == "symptoms":
        return isinstance(symptoms, list) and len(symptoms) > 0
    if channel == "wearable":
        return any(
            k in lookup
            for k in (
                "sleep_duration_hours",
                "hrv_rmssd_ms",
                "resting_heart_rate_bpm",
                "hrv_rmssd_delta_from_baseline_pct",
                "resting_heart_rate_delta_from_baseline_bpm",
            )
        )
    if channel == "questionnaire":
        return len(lookup) > 0
    if channel == "intervention_log":
        return bool(medications or supplements)
    return False


def _mapping_is_active(mapping: dict[str, Any], lookup: dict[str, float]) -> bool:
    field = str(mapping.get("source_field", "")).strip()
    if not field or field not in lookup:
        return False
    primary = lookup[field]
    rule = mapping.get("activation_rule", {}) or {}
    operator = str(rule.get("operator", "") or "").strip().lower()
    threshold = _safe_float(rule.get("threshold"), 0.0)
    secondary = [str(v) for v in (rule.get("secondary_fields") or []) if str(v).strip()]

    if rule.get("rule_type") == "threshold":
        if operator == "lt":
            return primary < threshold
        if operator == "lte":
            return primary <= threshold
        if operator == "gt":
            return primary > threshold
        return primary >= threshold

    if rule.get("rule_type") == "composite":
        values = [lookup.get(name) for name in secondary]
        if operator == "lt+gt" and len(values) >= 1 and values[0] is not None:
            return primary < threshold and values[0] > threshold
        if operator == "gt+lt" and len(values) >= 1 and values[0] is not None:
            return primary > threshold and values[0] < threshold
        if operator == "gte+low_sleep+temp_dev" and len(values) >= 2:
            sleep_val, temp_val = values[0], values[1]
            return (
                primary >= threshold
                and sleep_val is not None
                and sleep_val < 6.0
                and temp_val is not None
                and temp_val > 0.0
            )
        if operator == "gte+low_bm" and len(values) >= 1 and values[0] is not None:
            return primary >= threshold and values[0] < 5.0
        if operator == "gte+low_bm+stool_pattern" and len(values) >= 4:
            bm, stool_loose, stool_hard, malabs = values[0], values[1], values[2], values[3]
            stool_pattern = ((stool_loose or 0.0) + (stool_hard or 0.0)) > 0.0
            malabs_flag = (malabs or 0.0) > 0.0
            return primary >= threshold and bm is not None and bm < 5.0 and (stool_pattern or malabs_flag)
        if operator == "low_recovery+high_activity" and len(values) >= 2:
            sessions, strain = values[0], values[1]
            return (
                primary <= threshold
                and sessions is not None
                and sessions >= 3.0
                and strain is not None
                and strain >= 50.0
            )
        if operator == "low_timing+low_adherence":
            adherence_vals = [v for v in values if v is not None]
            return primary <= threshold and any(v < 80.0 for v in adherence_vals)
        if operator == "gte+acute_stressor" and len(values) >= 1 and values[0] is not None:
            return primary >= threshold and values[0] > 0.0
        if operator == "gte+symptom_bundle" and len(values) >= 2:
            supporting = sum(1 for v in values if v is not None and v >= 0.5)
            return primary >= threshold and supporting >= 2
        if operator == "baseline_drop+baseline_rise+stimulant" and len(values) >= 2:
            rise, stimulant = values[0], values[1]
            return (
                primary <= -10.0
                and rise is not None
                and rise >= 3.0
                and stimulant is not None
                and stimulant >= 6.0
            )
        if operator == "gte+meal_flare+nonresponse" and len(values) >= 2:
            meal_flare, nonresponse = values[0], values[1]
            return (
                primary >= threshold
                and meal_flare is not None
                and meal_flare >= 5.0
                and nonresponse is not None
                and nonresponse >= 5.0
            )
        if operator == "gte+late_hour" and len(values) >= 1 and values[0] is not None:
            latest_hour = values[0]
            return primary >= threshold and latest_hour >= 16.0
        if operator == "baseline_drop+baseline_rise" and len(values) >= 1 and values[0] is not None:
            rise = values[0]
            return primary <= -10.0 and rise >= 3.0

    return False


def _infer_why_today_latents(
    request: dict[str, Any],
    out: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], float]:
    registry_rows = _load_why_today_mapping_registry(str(DataConfig().seed_root))
    lookup = _build_observation_lookup(request)
    latents = {
        name: _latent_entry(0.0, 0.0)
        for name in WHY_TODAY_LATENT_NAMES
    }
    for mapping in registry_rows:
        latent = str(mapping.get("target_latent", "")).strip()
        if latent not in latents:
            continue
        if not _mapping_is_active(mapping, lookup):
            continue
        bounded = mapping.get("bounded_output", {}) or {}
        confidence_policy = mapping.get("confidence_policy", {}) or {}
        value = _safe_float(
            bounded.get("max_value"),
            _safe_float(bounded.get("default_value"), 0.0),
        )
        confidence = _safe_float(confidence_policy.get("base_weight"), 0.0)
        required_channels = confidence_policy.get("requires_supporting_channels", []) or []
        if any(
            not _has_supporting_channel(str(channel), request, lookup)
            for channel in required_channels
        ):
            confidence *= 0.5
        conflict_fields = confidence_policy.get("downweight_if_conflict_with", []) or []
        if any(field in lookup for field in conflict_fields):
            confidence *= 0.8
        current = latents[latent]
        if value > current["value"]:
            latents[latent] = _latent_entry(value, confidence)
    pred_latents = out.get("why_today_latents")
    pred_conf = out.get("why_today_latents_conf")
    if pred_latents is not None and pred_latents.numel() > 0:
        flat_latents = pred_latents.reshape(-1, pred_latents.shape[-1])
        mean_latents = flat_latents.mean(dim=0)
        if pred_conf is not None and pred_conf.numel() > 0:
            mean_conf = pred_conf.reshape(-1, pred_conf.shape[-1]).mean(dim=0)
        else:
            mean_conf = torch.zeros_like(mean_latents)
        for idx, name in enumerate(WHY_TODAY_LATENT_NAMES):
            if idx >= mean_latents.numel():
                break
            pred_value = max(0.0, min(1.0, float(mean_latents[idx].item())))
            pred_confidence = max(
                0.0,
                min(1.0, float(mean_conf[idx].item()) if idx < mean_conf.numel() else 0.0),
            )
            current = latents[name]
            # Treat curated registry rows as a prior, not a hard winner. When the
            # model has any non-trivial confidence, blend toward the learned value
            # instead of requiring it to fully beat the prior confidence.
            registry_value = _safe_float(current.get("value"), 0.0)
            registry_confidence = _safe_float(current.get("confidence"), 0.0)
            model_gate = max(0.0, min(1.0, 0.25 + 0.75 * pred_confidence))
            blended_value = (1.0 - model_gate) * registry_value + model_gate * pred_value
            blended_confidence = max(pred_confidence, 0.6 * registry_confidence)
            latents[name] = _latent_entry(blended_value, blended_confidence)
    for name in WHY_TODAY_LATENT_NAMES:
        latents.setdefault(name, _latent_entry(0.0, 0.0))
    overall_conf = sum(v["confidence"] for v in latents.values()) / max(1, len(latents))
    return latents, round(max(0.0, min(1.0, overall_conf)), 4)


def _build_specialist_signal_bundle(request: dict[str, Any]) -> dict[str, Any]:
    lookup = canonical_observation_lookup(request)
    temp_dev = abs(_safe_float(lookup.get("temperature_deviation_c"), 0.0))
    crp = _safe_float(lookup.get("crp_mg_l"), 0.0)
    readiness = _safe_float(lookup.get("readiness_score_0_100"), 100.0)
    chills = _clamp01(lookup.get("chills"))
    myalgias = _clamp01(lookup.get("myalgias"))
    anxiety = _clamp01(lookup.get("anxiety"))
    irritability = _clamp01(lookup.get("irritability"))
    agitation = _clamp01(lookup.get("agitation"))
    palpitations = _clamp01(lookup.get("palpitations"))
    headache = _clamp01(lookup.get("headache"))
    fatigue = _clamp01(lookup.get("fatigue"))
    brain_fog = _clamp01(lookup.get("brain_fog"))
    bloating = _clamp01(lookup.get("bloating"))
    abdominal_pain = _clamp01(lookup.get("abdominal_pain"))
    food_intolerance = _clamp01(lookup.get("food_intolerance"))
    insomnia = _clamp01(lookup.get("insomnia"))
    wired_tired = _clamp01(lookup.get("wired_tired"))
    sensory = _clamp01(lookup.get("sensory_overstimulation"))
    stress = _clamp01(_safe_float(lookup.get("stress_level_0_10"), 0.0) / 10.0)
    stimulant_sensitivity = _clamp01(_safe_float(lookup.get("stimulant_sensitivity_0_10"), 0.0) / 10.0)
    poor_stress_fit = _clamp01(_safe_float(lookup.get("poor_response_to_generic_stress_0_10"), 0.0) / 10.0)
    postprandial_flare = _clamp01(_safe_float(lookup.get("postprandial_symptom_flare_0_10"), 0.0) / 10.0)
    oral_nonresponse = _clamp01(_safe_float(lookup.get("oral_intervention_nonresponse_0_10"), 0.0) / 10.0)
    gi_burden = _clamp01(_safe_float(lookup.get("gi_symptom_burden_0_10"), 0.0) / 10.0)
    histamine_days = _clamp01(_safe_float(lookup.get("high_histamine_meals_days_7d"), 0.0) / 7.0)
    alcohol_days = _clamp01(_safe_float(lookup.get("alcohol_days_7d"), 0.0) / 7.0)
    hrv_drop = _clamp01(max(0.0, -_safe_float(lookup.get("hrv_rmssd_delta_from_baseline_pct"), 0.0)) / 35.0)
    rhr_rise = _clamp01(max(0.0, _safe_float(lookup.get("resting_heart_rate_delta_from_baseline_bpm"), 0.0)) / 10.0)
    malabs_flag = _clamp01(lookup.get("diagnosed_malabsorption_flag"))
    sleep_hours = _safe_float(lookup.get("sleep_duration_hours"), _safe_float(lookup.get("sleep_duration_h"), 0.0))
    sleep_deficit = _clamp01((7.5 - sleep_hours) / 3.5)
    caffeine_after_2pm = _clamp01(_safe_float(lookup.get("caffeine_after_2pm_days_7d"), 0.0) / 7.0)
    caffeine_latest_hour = _safe_float(lookup.get("caffeine_latest_local_hour"), 0.0)
    late_hour_gate = _clamp01((caffeine_latest_hour - 14.0) / 6.0)
    meal_irregularity = _clamp01((5.0 - _safe_float(lookup.get("meal_timing_regularity_1_5"), 5.0)) / 4.0)
    adherence_inconsistency = _clamp01((5.0 - _safe_float(lookup.get("intervention_timing_consistency_1_5"), 5.0)) / 4.0)
    methyl_tolerance = _safe_float(
        lookup.get("methyl_donor_tolerance_7d"),
        _safe_float(lookup.get("methyl_donor_tolerance_0_10"), 10.0),
    )
    methyl_strain = _clamp01((10.0 - methyl_tolerance) / 10.0)
    recovery_deficit = _clamp01((5.0 - _safe_float(lookup.get("exercise_recovery_1_5"), 5.0)) / 4.0)
    exercise_intensity = _clamp01(_safe_float(lookup.get("exercise_intensity_1_5"), 0.0) / 5.0)

    scores = {
        "infection": _clamp01(
            0.24 * _clamp01(temp_dev / 1.5)
            + 0.24 * _clamp01(crp / 25.0)
            + 0.16 * _clamp01((55.0 - readiness) / 55.0)
            + 0.18 * chills
            + 0.18 * myalgias
            - 0.10 * stress
            - 0.08 * anxiety
            - 0.08 * palpitations
            - 0.06 * histamine_days
            - 0.06 * alcohol_days
        ),
        "stress": _clamp01(
            0.28 * hrv_drop
            + 0.24 * rhr_rise
            + 0.20 * stress
            + 0.14 * anxiety
            + 0.14 * palpitations
            - 0.08 * _clamp01(crp / 25.0)
            - 0.08 * chills
        ),
        "glutamate": _clamp01(
            0.18 * wired_tired
            + 0.18 * sensory
            + 0.18 * insomnia
            + 0.14 * irritability
            + 0.10 * anxiety
            + 0.10 * stimulant_sensitivity
            + 0.12 * poor_stress_fit
            - 0.10 * palpitations
            - 0.08 * stress
        ),
        "catechol": _clamp01(
            0.15 * palpitations
            + 0.14 * agitation
            + 0.12 * anxiety
            + 0.16 * stimulant_sensitivity
            + 0.14 * stress
            + 0.12 * hrv_drop
            + 0.10 * rhr_rise
            - 0.10 * _clamp01(crp / 25.0)
        ),
        "transport": _clamp01(
            0.16 * gi_burden
            + 0.18 * postprandial_flare
            + 0.18 * oral_nonresponse
            + 0.12 * food_intolerance
            + 0.10 * abdominal_pain
            + 0.08 * bloating
            + 0.10 * malabs_flag
        ),
        "histamine": _clamp01(
            0.34 * histamine_days
            + 0.18 * alcohol_days
            + 0.14 * bloating
            + 0.10 * headache
            + 0.10 * food_intolerance
            + 0.08 * insomnia
            + 0.08 * brain_fog
            - 0.08 * _clamp01(crp / 25.0)
            - 0.06 * chills
        ),
        "sleep_pressure": _clamp01(
            0.30 * sleep_deficit
            + 0.22 * caffeine_after_2pm
            + 0.16 * insomnia
            + 0.12 * wired_tired
            + 0.12 * fatigue
            + 0.08 * stress
        ),
        "circadian": _clamp01(
            0.26 * late_hour_gate
            + 0.20 * meal_irregularity
            + 0.14 * _clamp01((60.0 - readiness) / 60.0)
            + 0.12 * insomnia
            + 0.12 * fatigue
            + 0.08 * caffeine_after_2pm
            + 0.08 * sleep_deficit
        ),
        "methyl": _clamp01(
            0.30 * methyl_strain
            + 0.18 * irritability
            + 0.16 * brain_fog
            + 0.12 * headache
            + 0.12 * insomnia
            + 0.12 * food_intolerance
        ),
        "absorption": _clamp01(
            0.28 * gi_burden
            + 0.22 * malabs_flag
            + 0.18 * food_intolerance
            + 0.14 * abdominal_pain
            + 0.10 * bloating
            + 0.08 * postprandial_flare
        ),
        "cofactor": _clamp01(
            0.24 * fatigue
            + 0.20 * recovery_deficit
            + 0.16 * sleep_deficit
            + 0.14 * brain_fog
            + 0.10 * headache
            + 0.08 * exercise_intensity
            + 0.08 * alcohol_days
        ),
        "rebound": _clamp01(
            0.30 * adherence_inconsistency
            + 0.18 * agitation
            + 0.16 * anxiety
            + 0.12 * palpitations
            + 0.12 * insomnia
            + 0.12 * stress
        ),
    }
    return {"lookup": lookup, "scores": scores}


def _run_specialist_orchestrator(
    request: dict[str, Any],
    latents: dict[str, dict[str, float]],
    posterior_gain: float,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], float]:
    signals = _build_specialist_signal_bundle(request)
    scores = signals["scores"]
    registry = _specialist_registry()
    final_latents = {name: _latent_entry(val["value"], val["confidence"]) for name, val in latents.items()}
    info: dict[str, Any] = {}

    for latent_id in WHY_TODAY_LATENT_NAMES:
        spec = registry.get(latent_id)
        current = latents.get(latent_id, _latent_entry(0.0, 0.0))
        generic_value = _safe_float(current.get("value"))
        generic_conf = _safe_float(current.get("confidence"))
        support = _clamp01(scores.get(spec.support_profile if spec else "", generic_value)) if spec else generic_value
        teacher_pred = None
        teacher_path = None
        teacher_label = None
        source = "generic_only"
        specialist_value = None
        activation_gate = support
        final_value = generic_value
        final_conf = generic_conf
        if spec is not None and spec.teacher_path is not None:
            teacher_pred = _predict_specialist_teacher(request, spec.teacher_path, spec.teacher_label or spec.latent_id)
            teacher_path = str(spec.teacher_path)
            teacher_label = spec.teacher_label
            teacher_support = max(support, _safe_float(teacher_pred, 0.0))
            activation_gate = teacher_support
            if teacher_pred is not None and teacher_support >= spec.activation_threshold:
                teacher_target = spec.teacher_weight * teacher_pred + (1.0 - spec.teacher_weight) * generic_value
                activation_gate = _clamp01(
                    spec.blend_bias
                    + spec.blend_support_weight * teacher_support
                    + spec.blend_score_weight * max(generic_value, support)
                )
                specialist_value = (1.0 - activation_gate) * generic_value + activation_gate * teacher_target
                final_value = specialist_value
                final_conf = max(generic_conf, spec.confidence_floor + 0.25 * support)
                source = "specialist_teacher"
        elif spec is not None and spec.status == "active_evidence" and support >= spec.activation_threshold:
            activation_gate = support
            specialist_value = max(generic_value, support)
            final_value = specialist_value
            final_conf = max(generic_conf, spec.confidence_floor + 0.20 * support)
            source = "specialist_evidence"

        final_latents[latent_id] = _latent_entry(final_value, final_conf)
        info[latent_id] = {
            "mode": "specialist_orchestrator",
            "source": source,
            "owner_mode": spec.owner_mode if spec else "generic_only",
            "status": spec.status if spec else "generic_only",
            "support_profile": spec.support_profile if spec else None,
            "teacher_label": teacher_label,
            "teacher_pred": round(float(teacher_pred), 6) if teacher_pred is not None else None,
            "teacher_path": teacher_path,
            "generic_value": round(generic_value, 6),
            "generic_confidence": round(generic_conf, 4),
            "specialist_value": round(float(specialist_value), 6) if specialist_value is not None else None,
            "evidence_support": round(float(support), 6),
            "teacher_support": round(float(max(support, _safe_float(teacher_pred, 0.0))), 6),
            "activation_gate": round(float(activation_gate), 6),
            "posterior_gain_gate": round(_clamp01(posterior_gain), 6),
        }

    infection_row = info.get("infection_like_sickness_state")
    if infection_row and infection_row.get("source") == "specialist_teacher":
        infection_support = _safe_float(infection_row.get("evidence_support"), 0.0)
        stress_lat = final_latents.get("stress_hormone_load")
        if isinstance(stress_lat, dict):
            final_latents["stress_hormone_load"] = _latent_entry(
                max(0.0, _safe_float(stress_lat.get("value")) * (1.0 - 0.55 * infection_support)),
                max(0.0, min(1.0, _safe_float(stress_lat.get("confidence")) * (1.0 - 0.25 * infection_support))),
            )

    top_active = [
        latent_id
        for latent_id, estimate in final_latents.items()
        if _safe_float(estimate.get("value")) >= 0.45 and _safe_float(estimate.get("confidence")) >= 0.45
    ]
    for latent_id, row in info.items():
        final_entry = final_latents.get(latent_id, _latent_entry(0.0, 0.0))
        value = _safe_float(final_entry.get("value"))
        conflicts = [
            other
            for other in top_active
            if other != latent_id and abs(value - _safe_float(final_latents.get(other, {}).get("value"), 0.0)) <= 0.12
        ]
        row["final_value"] = round(value, 6)
        row["final_confidence"] = round(_safe_float(final_entry.get("confidence")), 4)
        row["won_arbitration"] = row.get("source") in {"specialist_teacher", "specialist_evidence"}
        row["conflict_with"] = sorted(conflicts)[:3]

    overall_conf = sum(_safe_float(v.get("confidence")) for v in final_latents.values()) / max(1, len(final_latents))
    return final_latents, info, round(_clamp01(overall_conf), 4)


def _infer_posterior_gain(out: dict[str, Any]) -> float:
    pred = out.get("posterior_gain")
    if pred is None:
        return 0.0
    try:
        if pred.numel() > 0:
            return round(max(0.0, min(1.0, float(pred.reshape(-1)[0].item()))), 6)
    except Exception:
        return 0.0
    return 0.0


def _build_specialist_orchestration_info(
    orchestrator_rows: dict[str, Any],
    why_today_latents: dict[str, dict[str, float]],
    posterior_gain: float,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for latent_id, raw in orchestrator_rows.items():
        estimate = why_today_latents.get(latent_id, _latent_entry(0.0, 0.0))
        row = dict(raw)
        row["final_value"] = round(_safe_float(estimate.get("value")), 6)
        row["final_confidence"] = round(_safe_float(estimate.get("confidence")), 4)
        row["posterior_gain_gate"] = round(_clamp01(posterior_gain), 6)
        info[latent_id] = row
    return info


def _blend_to_identity(value: float, identity: float, gate: float) -> float:
    g = max(0.0, min(1.0, float(gate)))
    return identity + (value - identity) * g


def _apply_posterior_gain_gate(
    hidden_states: dict[str, list[dict[str, float]]],
    bridge_saturations: dict[str, dict[str, float]],
    modulates_effects: dict[str, dict[str, float]],
    regulates_effects: dict[str, dict[str, float]],
    transport_effects: dict[str, dict[str, float]],
    why_today_latents: dict[str, dict[str, float]],
    why_today_confidence: float,
    posterior_gain: float,
) -> tuple[
    dict[str, list[dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    float,
]:
    gate = max(0.0, min(1.0, float(posterior_gain)))

    hidden_identities = {
        "modulates": 0.0,
        "regulates": 0.0,
        "signaling": 1.0,
        "bridges": 1.0,
        "transports_to": 1.0,
    }
    gated_hidden_states: dict[str, list[dict[str, float]]] = {}
    for etype, rows in hidden_states.items():
        identity = hidden_identities.get(etype, 0.0)
        gated_rows: list[dict[str, float]] = []
        for row in rows:
            gated_rows.append(
                {
                    **row,
                    "value": round(
                        _blend_to_identity(_safe_float(row.get("value")), identity, gate),
                        6,
                    ),
                    "confidence": round(
                        max(0.0, min(1.0, _safe_float(row.get("confidence")) * gate)),
                        4,
                    ),
                }
            )
        gated_hidden_states[etype] = gated_rows

    gated_bridges = {
        key: {
            "value": round(
                max(0.0, min(1.0, _blend_to_identity(_safe_float(val.get("value")), 1.0, gate))),
                6,
            ),
            "confidence": round(max(0.0, min(1.0, _safe_float(val.get("confidence")) * gate)), 4),
        }
        for key, val in bridge_saturations.items()
    }
    gated_modulates = {
        key: {
            "effect": round(
                max(-1.0, min(1.0, _blend_to_identity(_safe_float(val.get("effect")), 0.0, gate))),
                6,
            ),
            "confidence": round(max(0.0, min(1.0, _safe_float(val.get("confidence")) * gate)), 4),
        }
        for key, val in modulates_effects.items()
    }
    gated_regulates = {
        key: {
            "effect": round(
                max(0.0, min(1.0, _blend_to_identity(_safe_float(val.get("effect")), 0.0, gate))),
                6,
            ),
            "confidence": round(max(0.0, min(1.0, _safe_float(val.get("confidence")) * gate)), 4),
        }
        for key, val in regulates_effects.items()
    }
    gated_transport = {
        key: {
            "multiplier": round(
                max(0.0, min(2.0, _blend_to_identity(_safe_float(val.get("multiplier")), 1.0, gate))),
                6,
            ),
            "confidence": round(max(0.0, min(1.0, _safe_float(val.get("confidence")) * gate)), 4),
        }
        for key, val in transport_effects.items()
    }
    gated_latents = {
        key: {
            "value": round(max(0.0, min(1.0, _safe_float(val.get("value")) * gate)), 6),
            "confidence": round(max(0.0, min(1.0, _safe_float(val.get("confidence")) * gate)), 4),
        }
        for key, val in why_today_latents.items()
    }
    gated_why_today_confidence = round(max(0.0, min(1.0, why_today_confidence * gate)), 4)
    return (
        gated_hidden_states,
        gated_bridges,
        gated_modulates,
        gated_regulates,
        gated_transport,
        gated_latents,
        gated_why_today_confidence,
    )


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
    why_today_latents, _ = _infer_why_today_latents(request, out)
    posterior_gain = _infer_posterior_gain(out)
    why_today_latents, specialist_rows, why_today_confidence = _run_specialist_orchestrator(
        request,
        why_today_latents,
        posterior_gain,
    )

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

    # -- Build regulates_effects for Rust consumption --------------------------
    # Maps "reg:SRC_ENZYME->DST_ENZYME" → {effect: f64, confidence: f64}
    # Matches regulation_target_from_key() in sim.rs
    regulates_effects: dict[str, dict[str, float]] = {}
    regulates_hidden = out.get("regulates_hidden", torch.tensor([]))
    regulates_conf = out.get("regulates_conf", torch.tensor([]))
    if _meta is not None:
        for edge_idx in range(len(_meta.regulates_idx_to_src)):
            src_id = _meta.regulates_idx_to_src.get(edge_idx)
            dst_id = _meta.regulates_idx_to_dst.get(edge_idx)
            if src_id and dst_id and edge_idx < regulates_hidden.numel():
                key = f"reg:{src_id}->{dst_id}"
                effect = regulates_hidden[edge_idx].item()
                conf = regulates_conf[edge_idx].item() if edge_idx < regulates_conf.numel() else 1.0
                regulates_effects[key] = {
                    "effect": round(max(0.0, min(1.0, effect)), 6),
                    "confidence": round(max(0.0, min(1.0, conf)), 4),
                }

    (
        hidden_states,
        bridge_saturations,
        modulates_effects,
        regulates_effects,
        transport_effects,
        why_today_latents,
        why_today_confidence,
    ) = _apply_posterior_gain_gate(
        hidden_states,
        bridge_saturations,
        modulates_effects,
        regulates_effects,
        transport_effects,
        why_today_latents,
        why_today_confidence,
        posterior_gain,
    )

    correction_packet = {
        "bridge_saturations": bridge_saturations,
        "modulates_effects": modulates_effects,
        "regulates_effects": regulates_effects,
        "transport_effects": transport_effects,
    }
    specialist_orchestration = _build_specialist_orchestration_info(
        specialist_rows,
        why_today_latents,
        posterior_gain,
    )

    return {
        "calibration_id": cal_id,
        "hidden_states": hidden_states,
        "bridge_saturations": bridge_saturations,  # For Rust flux engine
        "modulates_effects": modulates_effects,    # For Rust allosteric modulation
        "transport_effects": transport_effects,    # For compartment transport
        "regulates_effects": regulates_effects,    # For Rust regulatory modulation
        "correction_packet": correction_packet,
        "why_today_latents": why_today_latents,
        "why_today_confidence": why_today_confidence,
        "posterior_gain": posterior_gain,
        "hybrid_priors": specialist_orchestration,
        "specialist_orchestration": specialist_orchestration,
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
    # Edge-type → (idx_to_src, idx_to_dst) resolver maps from _meta.
    _edge_resolvers: dict[str, tuple[dict[int, str], dict[int, str]]] = {}
    if _meta is not None:
        _edge_resolvers = {
            "modulates": (_meta.modulates_idx_to_src, _meta.modulates_idx_to_dst),
            "regulates": (_meta.regulates_idx_to_src, _meta.regulates_idx_to_dst),
            "bridges": (
                {i: _meta.bridges.get(bid, {}).get("mediator_id", bid)
                 for i, bid in _meta.bridge_idx_to_id.items()},
                {i: _meta.bridges.get(bid, {}).get("target_enzyme_id", bid)
                 for i, bid in _meta.bridge_idx_to_id.items()},
            ),
            "transports_to": (
                {i: p[0] for i, p in _meta.transport_idx_to_pair.items()},
                {i: p[1] for i, p in _meta.transport_idx_to_pair.items()},
            ),
        }

    attention_weights = out.get("attention_weights", {})
    for etype in edge_types:
        attn = attention_weights.get(etype)
        if attn is None or attn.numel() == 0:
            continue

        # attn shape: (num_edges, num_heads) — average across heads
        attn_mean = attn.mean(dim=-1)  # (num_edges,)
        top_k = min(10, attn_mean.numel())
        top_vals, top_idx = torch.topk(attn_mean, top_k)

        # Resolve edge indices to human-readable entity IDs
        resolved_edges: list[ResolvedEdge] = []
        idx_to_src, idx_to_dst = _edge_resolvers.get(etype, ({}, {}))
        for val, idx in zip(top_vals, top_idx):
            eidx = idx.item()
            src_id = idx_to_src.get(eidx, f"?{eidx}")
            dst_id = idx_to_dst.get(eidx, f"?{eidx}")
            resolved_edges.append(ResolvedEdge(
                edge_type=etype,
                src_id=src_id,
                dst_id=dst_id,
                attention=val.item(),
            ))

        trace.attention_summary.append(AttentionSummary(
            edge_type=etype,
            num_edges=attn_mean.numel(),
            mean_attention=attn_mean.mean().item(),
            max_attention=attn_mean.max().item(),
            top_edge_indices=[i.item() for i in top_idx],
            top_attention_edges=resolved_edges,
        ))

    return trace


def _static_baselines(request: dict[str, Any]) -> dict[str, Any]:
    """Return static baseline calibration dict (GNN has zero effect).

    These defaults match the Rust engine's blending formula:
        effective = baseline + (predicted - baseline) * confidence
    With confidence=0.0, the engine always uses its own baseline.
    """
    _ensure_loaded(require_trained=False)

    cal_id = str(uuid.uuid4())

    bridge_saturations: dict[str, dict[str, float]] = {}
    modulates_effects: dict[str, dict[str, float]] = {}
    transport_effects: dict[str, dict[str, float]] = {}
    regulates_effects: dict[str, dict[str, float]] = {}

    if _meta is not None:
        # Bridges: all at 1.0 (fully saturated), confidence 0.0
        for edge_idx, bridge_id in _meta.bridge_idx_to_id.items():
            bridge_saturations[bridge_id] = {"value": 1.0, "confidence": 0.0}

        # Modulates: all at 0.0 (no allosteric shift), confidence 0.0
        for edge_idx in range(len(_meta.modulates_idx_to_src)):
            src_id = _meta.modulates_idx_to_src.get(edge_idx)
            dst_id = _meta.modulates_idx_to_dst.get(edge_idx)
            if src_id and dst_id:
                modulates_effects[f"{src_id}_{dst_id}"] = {"effect": 0.0, "confidence": 0.0}

        # Transport: all at 1.0 (baseline permeability), confidence 0.0
        for edge_idx, pair in _meta.transport_idx_to_pair.items():
            src_id, dst_id = pair
            transport_effects[f"{src_id}->{dst_id}"] = {"multiplier": 1.0, "confidence": 0.0}

        # Regulates: all at 0.0 (no regulatory shift), confidence 0.0
        for edge_idx in range(len(_meta.regulates_idx_to_src)):
            src_id = _meta.regulates_idx_to_src.get(edge_idx)
            dst_id = _meta.regulates_idx_to_dst.get(edge_idx)
            if src_id and dst_id:
                regulates_effects[f"reg:{src_id}->{dst_id}"] = {"effect": 0.0, "confidence": 0.0}

    return {
        "calibration_id": cal_id,
        "hidden_states": {},
        "bridge_saturations": bridge_saturations,
        "modulates_effects": modulates_effects,
        "transport_effects": transport_effects,
        "regulates_effects": regulates_effects,
        "correction_packet": {
            "bridge_saturations": bridge_saturations,
            "modulates_effects": modulates_effects,
            "regulates_effects": regulates_effects,
            "transport_effects": transport_effects,
        },
        "why_today_latents": {
            name: {"value": 0.0, "confidence": 0.0}
            for name in WHY_TODAY_LATENT_NAMES
        },
        "why_today_confidence": 0.0,
        "posterior_gain": 0.0,
        "audit": AuditTrace(calibration_id=cal_id).to_dict(),
    }


def calibrate_or_default(request: dict[str, Any]) -> dict[str, Any]:
    """Run GNN calibration if trained weights exist, else return static baselines.

    This is the primary entry point. The GNN is an optional enhancement —
    the Rust engine works fully without it using these defaults:
    - bridges = 1.0 (fully saturated)
    - modulates = 0.0 (no allosteric shift)
    - transport = 1.0 (baseline permeability)
    - regulates = 0.0 (no regulatory shift)
    """
    try:
        return calibrate(request)
    except (FileNotFoundError, RuntimeError):
        return _static_baselines(request)
