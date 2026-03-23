from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _mean(vals: list[float], default: float = 0.0) -> float:
    if not vals:
        return float(default)
    return float(sum(vals) / len(vals))


def _extract_symptom_burden(request: dict[str, Any], observed: dict[str, Any]) -> float:
    ss = request.get("symptom_scores_0_1", {})
    vals: list[float] = []
    if isinstance(ss, dict):
        for v in ss.values():
            try:
                vals.append(_clamp(float(v), 0.0, 1.0))
            except Exception:
                continue
    if vals:
        return _mean(vals, 0.5)
    symptoms = observed.get("symptoms", [])
    if isinstance(symptoms, list):
        return _clamp(len(symptoms) / 10.0, 0.0, 1.0)
    return 0.5


def _extract_lifestyle_burden(request: dict[str, Any], observed: dict[str, Any]) -> float:
    lg = request.get("lifestyle_generic", request.get("lifestyle", {}))
    vals: list[float] = []
    if isinstance(lg, dict):
        for v in lg.values():
            try:
                vals.append(_clamp(float(v), 0.0, 1.0))
            except Exception:
                continue
    if vals:
        return _mean(vals, 0.5)
    lifestyle = observed.get("lifestyle_generic", observed.get("lifestyle", []))
    if isinstance(lifestyle, list):
        rel_vals: list[float] = []
        for row in lifestyle:
            if isinstance(row, dict):
                try:
                    rel_vals.append(_clamp(float(row.get("value", 0.5)) / 10.0, 0.0, 1.0))
                except Exception:
                    continue
        if rel_vals:
            return _mean(rel_vals, 0.5)
    return 0.5


def _extract_objective_signal(request: dict[str, Any], observed: dict[str, Any]) -> tuple[float, float]:
    obj = request.get("objective_channel_reliability", {})
    obj_feats = request.get("objective_channel_features", {})
    rel = 0.0
    if isinstance(obj, dict):
        try:
            rel = max(
                _clamp(float(obj.get("labs_reliability", 0.0)), 0.0, 1.0),
                _clamp(float(obj.get("wearables_reliability", 0.0)), 0.0, 1.0),
            )
        except Exception:
            rel = 0.0
    feat = 0.0
    if isinstance(obj_feats, dict):
        vals: list[float] = []
        for k in ("labs_delta_score", "wearables_delta_score"):
            try:
                vals.append(_clamp(float(obj_feats.get(k, 0.0)), -1.0, 1.0))
            except Exception:
                continue
        if vals:
            feat = _mean(vals, 0.0)
    labs = observed.get("labs", [])
    wearables = observed.get("wearables", [])
    present = 1.0 if (isinstance(labs, list) and labs) or (isinstance(wearables, list) and wearables) else 0.0
    return feat, max(rel, present * 0.6)


def _extract_signaling_prior(request: dict[str, Any]) -> float:
    sc = request.get("signaling_context", {})
    if not isinstance(sc, dict):
        return 0.0
    vals: list[float] = []
    for k in ("receptor_sensitivity_0_2", "inflammatory_tone_0_2", "crosstalk_gain_0_2"):
        try:
            vals.append((_clamp(float(sc.get(k, 1.0)), 0.0, 2.0) - 1.0))
        except Exception:
            continue
    return _mean(vals, 0.0)


@dataclass(frozen=True)
class LatentPosterior:
    prior: dict[str, float]
    posterior: dict[str, float]
    uncertainty: dict[str, float]
    gains: dict[str, float]


def infer_latent_posterior(row: dict[str, Any]) -> LatentPosterior:
    request = row.get("request", {}) if isinstance(row.get("request"), dict) else {}
    observed = row.get("observed_channels", {}) if isinstance(row.get("observed_channels"), dict) else {}
    symptom = _extract_symptom_burden(request, observed)
    lifestyle = _extract_lifestyle_burden(request, observed)
    objective_signal, objective_rel = _extract_objective_signal(request, observed)
    signaling_prior = _extract_signaling_prior(request)

    prior = {
        "bridge_state": _clamp(0.20 + 0.45 * signaling_prior, -1.0, 1.0),
        "signaling_state": _clamp(0.25 + 0.55 * signaling_prior, -1.0, 1.0),
        "crosstalk_state": _clamp(0.15 + 0.50 * signaling_prior, -1.0, 1.0),
    }

    # Reliability-weighted update:
    # objective channels carry stronger gain than lifestyle/symptoms when available.
    weak_gain = 0.18
    strong_gain = 0.42 * _clamp(objective_rel, 0.0, 1.0)
    k_bridge = _clamp(weak_gain + 0.8 * strong_gain, 0.05, 0.55)
    k_signaling = _clamp(weak_gain + 1.0 * strong_gain, 0.05, 0.60)
    k_crosstalk = _clamp(weak_gain + 0.9 * strong_gain, 0.05, 0.58)

    obs_bridge = _clamp(0.55 * symptom + 0.20 * lifestyle + 0.25 * ((objective_signal + 1.0) / 2.0), 0.0, 1.0) * 2.0 - 1.0
    obs_signaling = _clamp(0.60 * symptom + 0.10 * lifestyle + 0.30 * ((objective_signal + 1.0) / 2.0), 0.0, 1.0) * 2.0 - 1.0
    obs_crosstalk = _clamp(0.40 * symptom + 0.30 * lifestyle + 0.30 * ((objective_signal + 1.0) / 2.0), 0.0, 1.0) * 2.0 - 1.0

    posterior = {
        "bridge_state": _clamp((1.0 - k_bridge) * prior["bridge_state"] + k_bridge * obs_bridge, -1.0, 1.0),
        "signaling_state": _clamp((1.0 - k_signaling) * prior["signaling_state"] + k_signaling * obs_signaling, -1.0, 1.0),
        "crosstalk_state": _clamp((1.0 - k_crosstalk) * prior["crosstalk_state"] + k_crosstalk * obs_crosstalk, -1.0, 1.0),
    }

    # Lower uncertainty when stronger objective evidence is present.
    base_std = _clamp(0.55 - 0.35 * objective_rel, 0.15, 0.70)
    uncertainty = {
        "bridge_state_std": _clamp(base_std + 0.10 * abs(posterior["bridge_state"] - prior["bridge_state"]), 0.05, 1.0),
        "signaling_state_std": _clamp(base_std + 0.10 * abs(posterior["signaling_state"] - prior["signaling_state"]), 0.05, 1.0),
        "crosstalk_state_std": _clamp(base_std + 0.10 * abs(posterior["crosstalk_state"] - prior["crosstalk_state"]), 0.05, 1.0),
    }

    return LatentPosterior(
        prior=prior,
        posterior=posterior,
        uncertainty=uncertainty,
        gains={
            "k_bridge": k_bridge,
            "k_signaling": k_signaling,
            "k_crosstalk": k_crosstalk,
            "objective_reliability": _clamp(objective_rel, 0.0, 1.0),
        },
    )
