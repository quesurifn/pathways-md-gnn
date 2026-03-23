from __future__ import annotations

from typing import Any

OBSERVATION_FEATURE_NAMES: tuple[str, ...] = (
    "symptom_count",
    "symptom_mean_severity",
    "fatigue",
    "brain_fog",
    "headache",
    "myalgias",
    "chills",
    "anxiety",
    "irritability",
    "agitation",
    "palpitations",
    "bloating",
    "abdominal_pain",
    "food_intolerance",
    "insomnia",
    "wired_tired",
    "sensory_overstimulation",
    "sleep_duration_hours",
    "readiness_score_0_100",
    "temperature_deviation_c",
    "crp_mg_l",
    "hrv_rmssd_delta_from_baseline_pct",
    "resting_heart_rate_delta_from_baseline_bpm",
    "stress_level_0_10",
    "alcohol_days_7d",
    "high_histamine_meals_days_7d",
    "caffeine_after_2pm_days_7d",
    "exercise_intensity_1_5",
    "exercise_recovery_1_5",
    "meal_timing_regularity_1_5",
    "intervention_timing_consistency_1_5",
    "gi_symptom_burden_0_10",
    "stimulant_sensitivity_0_10",
    "poor_response_to_generic_stress_0_10",
    "postprandial_symptom_flare_0_10",
    "oral_intervention_nonresponse_0_10",
    "febrile_signature_score",
    "stress_autonomic_signature_score",
    "glutamate_gaba_signature_score",
    "catecholamine_signature_score",
    "transport_constraint_signature_score",
)

OBSERVATION_FEATURE_DIM = len(OBSERVATION_FEATURE_NAMES)

_SYMPTOM_KEYS = {
    "fatigue": "fatigue",
    "brain_fog": "brain_fog",
    "headache": "headache",
    "myalgias": "myalgias",
    "chills": "chills",
    "anxiety": "anxiety",
    "irritability": "irritability",
    "agitation": "agitation",
    "palpitations": "palpitations",
    "bloating": "bloating",
    "abdominal_pain": "abdominal_pain",
    "food_intolerance": "food_intolerance",
    "insomnia": "insomnia",
    "wired_tired": "wired_tired",
    "sensory_overstimulation": "sensory_overstimulation",
    "overstimulated": "sensory_overstimulation",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _numeric_dict(obj: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def canonical_observation_lookup(request: dict[str, Any]) -> dict[str, float]:
    observation_packet = request.get("observation_packet", {}) or {}
    context = {}
    context.update(_numeric_dict(request.get("context", {}) or {}))
    context.update(_numeric_dict(observation_packet.get("context", {}) or {}))
    wearables = {}
    wearables.update(_numeric_dict(request.get("wearables", {}) or {}))
    wearables.update(_numeric_dict(observation_packet.get("wearables", {}) or {}))
    wearables.update(_numeric_dict(observation_packet.get("wearable_normalized", {}) or {}))
    wearables.update(_numeric_dict(observation_packet.get("wearable_normalized_packet", {}) or {}))
    lookup = {}
    lookup.update(context)
    lookup.update(wearables)

    alias_pairs = (
        ("sleep_duration_h", "sleep_duration_hours"),
        ("stress_load", "stress_level_0_10"),
        ("exercise_intensity", "exercise_intensity_1_5"),
        ("exercise_recovery", "exercise_recovery_1_5"),
        ("meal_regular", "meal_timing_regularity_1_5"),
        ("adherence", "intervention_timing_consistency_1_5"),
        ("stimulant_sensitivity", "stimulant_sensitivity_0_10"),
        ("caffeine_sensitivity", "stimulant_sensitivity_0_10"),
        ("poor_response_to_generic_stress", "poor_response_to_generic_stress_0_10"),
        ("postprandial_symptom_flare", "postprandial_symptom_flare_0_10"),
        ("oral_intervention_nonresponse", "oral_intervention_nonresponse_0_10"),
        ("alcohol_units", "alcohol_days_7d"),
        ("histamine_exposure", "high_histamine_meals_days_7d"),
        ("temperature_deviation", "temperature_deviation_c"),
        ("readiness_score", "readiness_score_0_100"),
        ("readiness", "readiness_score_0_100"),
    )
    for src_key, dst_key in alias_pairs:
        if dst_key not in lookup and src_key in lookup:
            lookup[dst_key] = float(lookup[src_key])

    symptoms = observation_packet.get("symptoms", request.get("symptoms", [])) or []
    symptom_scores: dict[str, float] = {v: 0.0 for v in _SYMPTOM_KEYS.values()}
    symptom_total = 0.0
    symptom_count = 0.0
    gi_burden = 0.0
    gi_terms = {"bloating", "nausea", "constipation", "diarrhea", "abdominal_pain", "gi"}
    if isinstance(symptoms, list):
        for sym in symptoms:
            if not isinstance(sym, dict):
                continue
            sym_id = str(sym.get("symptom_id") or sym.get("id") or "").strip().lower()
            sev = _safe_float(sym.get("severity_0_1"), -1.0)
            if sev < 0.0:
                raw_val = _safe_float(sym.get("value"), 5.0)
                sev = max(0.0, min(1.0, raw_val / 10.0))
            sev = max(0.0, min(1.0, sev))
            symptom_total += sev
            symptom_count += 1.0
            if sym_id in _SYMPTOM_KEYS:
                symptom_scores[_SYMPTOM_KEYS[sym_id]] = max(symptom_scores[_SYMPTOM_KEYS[sym_id]], sev)
            if sym_id in gi_terms:
                gi_burden = max(gi_burden, sev * 10.0)
    lookup["symptom_count"] = symptom_count
    lookup["symptom_mean_severity"] = (symptom_total / symptom_count) if symptom_count > 0 else 0.0
    for name, val in symptom_scores.items():
        lookup[name] = val
    if "gi_symptom_burden_0_10" not in lookup:
        lookup["gi_symptom_burden_0_10"] = gi_burden

    medications = observation_packet.get("medications", request.get("medications", [])) or []
    supplements = observation_packet.get("supplements", request.get("supplements", [])) or []
    adherence_vals = []
    for rows in (medications, supplements):
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("adherence") is not None:
                adherence_vals.append(max(0.0, min(1.0, _safe_float(row.get("adherence"), 1.0))) * 5.0)
    if adherence_vals and "intervention_timing_consistency_1_5" not in lookup:
        lookup["intervention_timing_consistency_1_5"] = sum(adherence_vals) / len(adherence_vals)

    return lookup


def build_observation_feature_vector(request: dict[str, Any]) -> list[float]:
    lookup = canonical_observation_lookup(request)
    temp_dev = abs(float(lookup.get("temperature_deviation_c", 0.0)))
    crp = float(lookup.get("crp_mg_l", 0.0))
    readiness = float(lookup.get("readiness_score_0_100", 0.0))
    chills = float(lookup.get("chills", 0.0))
    myalgias = float(lookup.get("myalgias", 0.0))
    anxiety = float(lookup.get("anxiety", 0.0))
    irritability = float(lookup.get("irritability", 0.0))
    agitation = float(lookup.get("agitation", 0.0))
    palpitations = float(lookup.get("palpitations", 0.0))
    insomnia = float(lookup.get("insomnia", 0.0))
    wired_tired = float(lookup.get("wired_tired", 0.0))
    sensory = float(lookup.get("sensory_overstimulation", 0.0))
    stimulant_sensitivity = float(lookup.get("stimulant_sensitivity_0_10", 0.0))
    poor_stress_fit = float(lookup.get("poor_response_to_generic_stress_0_10", 0.0))
    postprandial_flare = float(lookup.get("postprandial_symptom_flare_0_10", 0.0))
    oral_nonresponse = float(lookup.get("oral_intervention_nonresponse_0_10", 0.0))
    food_intolerance = float(lookup.get("food_intolerance", 0.0))
    abdominal_pain = float(lookup.get("abdominal_pain", 0.0))
    bloating = float(lookup.get("bloating", 0.0))
    gi_burden = float(lookup.get("gi_symptom_burden_0_10", 0.0))
    stress = float(lookup.get("stress_level_0_10", 0.0))
    hrv_drop = max(0.0, -float(lookup.get("hrv_rmssd_delta_from_baseline_pct", 0.0)))
    rhr_rise = max(0.0, float(lookup.get("resting_heart_rate_delta_from_baseline_bpm", 0.0)))
    histamine = float(lookup.get("high_histamine_meals_days_7d", 0.0))
    alcohol = float(lookup.get("alcohol_days_7d", 0.0))

    febrile_sig = (
        0.30 * max(0.0, min(1.0, temp_dev / 1.5))
        + 0.30 * max(0.0, min(1.0, crp / 25.0))
        + 0.15 * max(0.0, min(1.0, (55.0 - readiness) / 55.0))
        + 0.15 * max(0.0, min(1.0, chills))
        + 0.10 * max(0.0, min(1.0, myalgias))
        - 0.10 * max(0.0, min(1.0, stress / 10.0))
        - 0.08 * max(0.0, min(1.0, palpitations))
        - 0.07 * max(0.0, min(1.0, anxiety))
        - 0.05 * max(0.0, min(1.0, histamine / 7.0))
        - 0.05 * max(0.0, min(1.0, alcohol / 7.0))
    )
    stress_sig = (
        0.32 * max(0.0, min(1.0, hrv_drop / 35.0))
        + 0.26 * max(0.0, min(1.0, rhr_rise / 10.0))
        + 0.20 * max(0.0, min(1.0, stress / 10.0))
        + 0.12 * max(0.0, min(1.0, anxiety))
        + 0.10 * max(0.0, min(1.0, palpitations))
        - 0.08 * max(0.0, min(1.0, crp / 25.0))
        - 0.08 * max(0.0, min(1.0, chills))
    )
    lookup["febrile_signature_score"] = max(0.0, min(1.0, febrile_sig))
    lookup["stress_autonomic_signature_score"] = max(0.0, min(1.0, stress_sig))
    glu_gaba_sig = (
        0.20 * max(0.0, min(1.0, wired_tired))
        + 0.20 * max(0.0, min(1.0, sensory))
        + 0.18 * max(0.0, min(1.0, insomnia))
        + 0.14 * max(0.0, min(1.0, irritability))
        + 0.10 * max(0.0, min(1.0, anxiety))
        + 0.10 * max(0.0, min(1.0, stimulant_sensitivity / 10.0))
        + 0.08 * max(0.0, min(1.0, poor_stress_fit / 10.0))
        - 0.10 * max(0.0, min(1.0, palpitations))
    )
    catechol_sig = (
        0.18 * max(0.0, min(1.0, palpitations))
        + 0.17 * max(0.0, min(1.0, agitation))
        + 0.15 * max(0.0, min(1.0, anxiety))
        + 0.15 * max(0.0, min(1.0, stimulant_sensitivity / 10.0))
        + 0.12 * max(0.0, min(1.0, stress / 10.0))
        + 0.12 * max(0.0, min(1.0, hrv_drop / 35.0))
        + 0.11 * max(0.0, min(1.0, rhr_rise / 10.0))
        - 0.08 * max(0.0, min(1.0, crp / 25.0))
    )
    transport_sig = (
        0.18 * max(0.0, min(1.0, gi_burden / 10.0))
        + 0.18 * max(0.0, min(1.0, postprandial_flare / 10.0))
        + 0.16 * max(0.0, min(1.0, oral_nonresponse / 10.0))
        + 0.16 * max(0.0, min(1.0, food_intolerance))
        + 0.12 * max(0.0, min(1.0, abdominal_pain))
        + 0.10 * max(0.0, min(1.0, bloating))
        + 0.10 * max(0.0, min(1.0, float(lookup.get("diagnosed_malabsorption_flag", 0.0))))
    )
    lookup["glutamate_gaba_signature_score"] = max(0.0, min(1.0, glu_gaba_sig))
    lookup["catecholamine_signature_score"] = max(0.0, min(1.0, catechol_sig))
    lookup["transport_constraint_signature_score"] = max(0.0, min(1.0, transport_sig))
    values = [float(lookup.get(name, 0.0)) for name in OBSERVATION_FEATURE_NAMES]

    # Light normalization into comparable ranges for the dedicated encoder.
    normed: list[float] = []
    for name, value in zip(OBSERVATION_FEATURE_NAMES, values):
        v = float(value)
        if name == "symptom_count":
            normed.append(max(0.0, min(1.0, v / 8.0)))
        elif name == "sleep_duration_hours":
            normed.append(max(0.0, min(1.0, v / 10.0)))
        elif name == "readiness_score_0_100":
            normed.append(max(0.0, min(1.0, v / 100.0)))
        elif name == "temperature_deviation_c":
            normed.append(max(-1.0, min(1.0, v / 2.0)))
        elif name == "crp_mg_l":
            normed.append(max(0.0, min(1.0, v / 25.0)))
        elif name == "hrv_rmssd_delta_from_baseline_pct":
            normed.append(max(-1.0, min(1.0, v / 50.0)))
        elif name == "resting_heart_rate_delta_from_baseline_bpm":
            normed.append(max(-1.0, min(1.0, v / 15.0)))
        elif name.endswith("_0_10"):
            normed.append(max(0.0, min(1.0, v / 10.0)))
        elif name.endswith("_1_5"):
            normed.append(max(0.0, min(1.0, v / 5.0)))
        elif name.endswith("_days_7d"):
            normed.append(max(0.0, min(1.0, v / 7.0)))
        else:
            normed.append(max(0.0, min(1.0, v)))
    return normed
