"""
Loss function for posterior correction inference on learned edges.

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THE GNN DOES
# ═══════════════════════════════════════════════════════════════════════════════
#
# The GNN Calibrator is a POSTERIOR CORRECTION ENGINE.
#
# It does NOT predict metabolite concentrations directly.
# It does NOT predict enzyme activities directly.
#
# Instead, it infers posterior correction states on LEARNED EDGES that represent:
#   - How strongly a metabolite modulates an enzyme (allosteric effects)
#   - How strongly an enzyme regulates another enzyme (feedback loops)
#   - How a signaling molecule affects pathway crosstalk
#   - How saturated a cofactor bridge is between pathways
#   - How current-state evidence shifts mechanistic state today
#
# These correction states are then fed to the RUST FLUX ENGINE, which uses them
# as bounded posterior adjustments in a deterministic Michaelis-Menten simulation to compute
# actual metabolite concentrations and reaction fluxes.
#
# ═══════════════════════════════════════════════════════════════════════════════
# WHY THIS ARCHITECTURE?
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. PHYSICS-INFORMED: The Rust engine encodes biochemistry (M-M kinetics).
#    The GNN only needs to infer what we DON'T know from first principles.
#
# 2. INTERPRETABLE: Posterior corrections have clear biological meaning.
#    A bridge saturation of 0.3 means "cofactor is 30% available".
#
# 3. AUDITABLE: Attention weights show which evidence the GNN used.
#    If it predicts low BH4 saturation, we can trace which edges influenced that.
#
# 4. PERSONALIZED: Genotype is handled in the DAG prior and current-state
#    observations shift the posterior correction packet.
#
# ═══════════════════════════════════════════════════════════════════════════════
# DATA FLOW
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │                         CANONICAL SEED DATA                             │
#   │  (enzymes, metabolites, reactions, pathways, bridges, SNPs, signaling)  │
#   └─────────────────────────────────────────────────────────────────────────┘
#                                      │
#                                      ▼
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │                      HETEROGENEOUS GRAPH (DGL)                          │
#   │                                                                         │
#   │  Node Types:  enzyme, metabolite, reaction                              │
#   │                                                                         │
#   │  Physics Edges (message passing, no output):                            │
#   │    - catalyzes: enzyme → reaction                                       │
#   │    - substrate_of: metabolite → reaction                                │
#   │    - produces: reaction → metabolite                                    │
#   │    - cofactor_for: metabolite → enzyme                                  │
#   │                                                                         │
#   │  Learned Edges (attention + hidden state output):                       │
#   │    - modulates: metabolite → enzyme (allosteric)                        │
#   │    - regulates: enzyme → enzyme (feedback)                              │
#   │    - signaling: metabolite → metabolite (crosstalk)                     │
#   │    - bridges: metabolite → enzyme (cofactor saturation)                 │
#   │    - no SNP nodes; genotype belongs in the DAG prior                    │
#   └─────────────────────────────────────────────────────────────────────────┘
#                                      │
#                                      ▼
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │                         GNN CALIBRATOR                                  │
#   │                                                                         │
#   │  Input:   Node features (from seed data)                                │
#   │           Edge features (evidence quality, SNP effects, signaling)      │
#   │                                                                         │
#   │  Process: Pre-LN Transformer layers with:                               │
#   │           - KineticConv on physics edges (encodes M-M structure)        │
#   │           - Scaled dot-product attention on learned edges               │
#   │           - Edge features as attention bias (Graphormer-style)          │
#   │                                                                         │
#   │  Output:  Hidden state + confidence for each learned edge               │
#   │           Attention weights for each learned edge (audit trace)         │
#   └─────────────────────────────────────────────────────────────────────────┘
#                                      │
#                                      ▼
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │                       RUST FLUX ENGINE                                  │
#   │                                                                         │
#   │  Input:   Hidden states from GNN (saturations, modulation effects)      │
#   │           Baseline kinetics from seed data (Vmax, Km)                   │
#   │           Initial metabolite concentrations                             │
#   │                                                                         │
#   │  Process: Deterministic Michaelis-Menten simulation                     │
#   │           - Apply SNP modifiers: Vmax_eff = Vmax × affects_hidden       │
#   │           - Apply allosteric effects: Km_eff = Km × (1 - modulates)     │
#   │           - Apply cofactor limits: v = v × min(saturations)             │
#   │           - Euler integration until steady state                        │
#   │                                                                         │
#   │  Output:  Steady-state metabolite concentrations                        │
#   │           Reaction fluxes                                               │
#   │           Per-enzyme contributions                                      │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════

# Total Loss

```
L = L_pred + λ_smooth·L_smooth + λ_bridge·L_bridge + λ_conf·L_conf
```

# Components

1. **L_pred**: MSE between predicted and target hidden states
2. **L_smooth**: L2 regularization on predictions (Occam's razor)
3. **L_bridge**: Bridge coherence (bridges in same pathway should agree)
4. **L_conf**: Confidence calibration (confidence predicts accuracy)

# Edge Types and Their Semantics

The GNN predicts hidden states on 5 learned edge types:

## 1. Modulates (metabolite → enzyme)
- **Range**: [-1, +1]
- **Meaning**: Allosteric modulation effect on Km
- **Downstream**: Km_eff = Km × (1 - modulates_hidden)
- **Example**: ATP modulating phosphofructokinase

## 2. Regulates (enzyme → enzyme)
- **Range**: [0, +1]
- **Meaning**: Regulatory effect strength
- **Downstream**: Gene expression or post-translational modification
- **Example**: Kinase phosphorylating target enzyme

## 3. Signaling (signal_substance → target_bridge)
- **Range**: [0.5, 2.0]
- **Meaning**: Multiplier on pathway flux
- **Downstream**: Scales bridge saturation
- **Example**: Cortisol affecting HPA axis bridges

## 4. Bridges (pathway → pathway)
- **Range**: [0, 1]
- **Meaning**: Cofactor saturation level
- **Downstream**: Limits enzyme activity via Liebig's Law
- **Example**: BH4 bridging dopamine and serotonin synthesis

## 5. Affects (snp → enzyme) [P0 CRITICAL - PERSONALIZATION]
- **Range**: [0, 1]
- **Meaning**: SNP effect multiplier on enzyme Vmax
- **Downstream**: Vmax_eff = Vmax × affects_hidden
- **Example**: MTHFR C677T reducing activity to 0.3

The affects edge is the CORE PERSONALIZATION mechanism:
- 0.0 = complete loss of function (homozygous harmful)
- 0.3 = 70% reduction (typical heterozygous effect)
- 1.0 = normal activity (wild-type)

# Scientific Foundation

## Mean Squared Error (MSE)

We use MSE loss for regression because:

1. **Continuous outputs**: Hidden states are continuous scalars
2. **Gaussian assumption**: MSE is the MLE for Gaussian noise
3. **Gradient properties**: Smooth, convex, well-conditioned

MSE is derived from maximum likelihood under Gaussian noise:
```
L(θ) = -log p(y|x,θ) = -log N(y; f(x;θ), σ²) ∝ (y - f(x))²
```

Reference:
- Bishop (2006) "Pattern Recognition and Machine Learning" Ch. 1.2.5

## L2 Regularization (Smoothness)

The smoothness term penalizes large hidden states:
```
L_smooth = λ × Σ(hidden²)
```

This implements:
1. **Occam's razor**: Prefer simple explanations (small deviations from baseline)
2. **Ridge regression**: Equivalent to Gaussian prior on weights
3. **Numerical stability**: Prevents extreme predictions

Reference:
- Tikhonov (1943) "On the stability of inverse problems" (Tikhonov reg.)
- Hoerl & Kennard (1970) Technometrics 12:55-67 (Ridge regression)

## Confidence Calibration

We train confidence to predict prediction accuracy:
```
L_conf = MSE(confidence, 1 - |error|/range)
```

A well-calibrated model has:
```
E[|pred - true| | confidence = c] = (1 - c) × range
```

This means:
- confidence = 0.9 → expected error is 10% of output range
- confidence = 0.5 → expected error is 50% of output range

Why calibration matters:
1. **Downstream use**: Flux engine blends predictions by confidence
2. **Trustworthiness**: Users can assess prediction reliability
3. **Active learning**: Low confidence indicates need for more data

References:
- Guo et al. (2017) ICML. "On Calibration of Modern Neural Networks"
- Naeini et al. (2015) AAAI. "Obtaining Well Calibrated Probabilities
  Using Bayesian Binning" (calibration metrics)

## Bridge Coherence (Not Yet Implemented)

Bridges connecting the same pathway pair should predict similar saturations.
This implements pathway-level consistency as a soft constraint.

```
L_bridge = Var(saturations within pathway pair)
```

This is related to:
- Graph Laplacian regularization (smoothness on graph structure)
- Multitask learning (sharing information across related tasks)
"""

from __future__ import annotations

from typing import Any

import torch

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
WHY_TODAY_FAMILY_NAMES: tuple[str, ...] = (
    "histamine",
    "stress",
    "sleep",
    "gi_transport",
    "nutrient",
    "rebound",
    "neuro",
    "infection_like",
)
WHY_TODAY_LATENT_TO_FAMILY: dict[str, str] = {
    "histamine_load_today": "histamine",
    "catecholamine_clearance_stress": "stress",
    "glutamate_gaba_instability": "neuro",
    "methyl_donor_strain": "nutrient",
    "infection_like_sickness_state": "infection_like",
    "circadian_disruption": "sleep",
    "sleep_pressure_disruption": "sleep",
    "absorption_impairment": "gi_transport",
    "acute_cofactor_depletion": "nutrient",
    "transport_constraint_state": "gi_transport",
    "intervention_rebound_or_withdrawal": "rebound",
    "stress_hormone_load": "stress",
}
_FAMILY_INDEX = {name: idx for idx, name in enumerate(WHY_TODAY_FAMILY_NAMES)}
import torch.nn.functional as F

from .config import ModelConfig

_HARD_WHY_TODAY_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (11, 6),
    (3, 8),
    (7, 4),
)


def hidden_state_loss(
    outputs: dict[str, Any],
    targets: dict[str, torch.Tensor],
    target_masks: dict[str, torch.Tensor] | None = None,
    cfg: ModelConfig | None = None,
    physics_residual: torch.Tensor | None = None,
    physics_residual_vector: torch.Tensor | None = None,
    physics_vector_mask: torch.Tensor | None = None,
    posterior_gain_target: torch.Tensor | None = None,
    posterior_gain_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    physics_mask: torch.Tensor | None = None,
    edge_loss_weights: dict[str, float] | None = None,
    edge_size_normalize: bool = False,
    smooth_max_ratio_vs_pred: float = -1.0,
    conf_max_ratio_vs_pred: float = -1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the full loss for posterior correction inference.

    # Loss Components

    1. **L_pred**: Mean Squared Error on posterior corrections
       - Standard regression loss (Bishop 2006)
       - Assumes Gaussian observation noise

    2. **L_smooth**: L2 regularization on predictions
       - Tikhonov regularization / ridge penalty
       - Encourages small deviations from baseline

    3. **L_conf**: Confidence calibration loss
       - MSE between predicted confidence and actual accuracy
       - Ensures confidence is meaningful (Guo et al. 2017)

    Parameters
    ----------
    outputs : dict from model forward pass
        Posterior correction predictions and confidence scores per edge type:
        - modulates_hidden: Tensor[num_modulates] — allosteric effects ∈ [-1, 1]
        - regulates_hidden: Tensor[num_regulates] — regulatory effects ∈ [0, 1]
        - signaling_hidden: Tensor[num_signaling] — pathway multipliers ∈ [0.5, 2]
        - bridges_hidden: Tensor[num_bridges] — cofactor saturations ∈ [0, 1]
        - transports_to_hidden: Tensor[num_transports] — transport multipliers ∈ [0, 2]
        - *_conf: Tensor — confidence scores ∈ [0, 1] for each edge type
    targets : dict of ground-truth posterior corrections
        Keys: modulates, regulates, signaling, bridges, transports_to
        - For unsupervised training: targets derived from Rust simulation
        - For supervised training: targets from experimental data
    cfg : ModelConfig (for loss weights)
        - lambda_smooth: weight for L2 regularization (default 0.01)
        - lambda_bridge: weight for bridge coherence (default 0.1)
        - lambda_confidence: weight for calibration (default 0.1)

    Returns
    -------
    total_loss : scalar tensor
        Combined weighted loss for backpropagation
    breakdown : dict of component losses (detached, for logging)
        Keys: pred, smooth, bridge, confidence, total

    Notes
    -----
    The loss function is designed to work with the Rust flux engine:
    1. GNN predicts posterior corrections on learned edges
    2. Those corrections are fed to Rust engine as bounded adjustments
    3. Engine computes downstream metabolite concentrations
    4. Prediction error drives GNN training

    This creates a physics-informed loop where the GNN learns posterior
    corrections that produce biologically plausible flux distributions.
    """
    cfg = cfg or ModelConfig()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Prediction Loss (MSE)
    #
    # L_pred = Σ_e MSE(pred_e, target_e)
    #
    # MSE is the standard loss for continuous regression:
    # - Equivalent to MLE under Gaussian noise assumption
    # - Convex, smooth, well-conditioned optimization landscape
    # - Reference: Bishop (2006) "Pattern Recognition" Ch. 1.2.5
    # ─────────────────────────────────────────────────────────────────────────
    l_pred = torch.tensor(0.0, device=_get_device(outputs))
    l_pred_u = torch.tensor(0.0, device=_get_device(outputs))
    # Learned edge types currently emitted by PathwayGNN.
    edge_types = ["modulates", "regulates", "signaling", "bridges", "transports_to"]

    bsz = int(sample_weights.numel()) if sample_weights is not None else 0
    norm_weights = edge_loss_weights or {}

    pred_by_edge_w: dict[str, float] = {}
    pred_by_edge_u: dict[str, float] = {}
    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        target = targets.get(etype)
        if pred is not None and target is not None and pred.numel() > 0:
            n = min(pred.numel(), target.numel())
            if n <= 0:
                continue
            p = pred[:n]
            t = target[:n]
            m = None
            if target_masks is not None:
                raw_m = target_masks.get(etype)
                if raw_m is not None and raw_m.numel() > 0:
                    m = raw_m.reshape(-1)[:n].to(dtype=torch.bool, device=p.device)
            mse = (p - t) ** 2
            ew = float(norm_weights.get(etype, 1.0))
            if edge_size_normalize:
                ew = ew / max(1.0, float(n))
            if sample_weights is not None and bsz > 0:
                if m is not None:
                    per_sample = _reduce_per_sample_masked(mse, m, bsz)
                    per_sample_has = _reduce_per_sample_has(m, bsz)
                else:
                    per_sample = _reduce_per_sample(mse, bsz)
                    per_sample_has = None
                w_m = _weighted_mean(per_sample, sample_weights, per_sample_has)
                u_m = _weighted_mean(per_sample, None, per_sample_has)
                l_pred = l_pred + (ew * w_m)
                l_pred_u = l_pred_u + (ew * u_m)
                pred_by_edge_w[etype] = float((ew * w_m).item())
                pred_by_edge_u[etype] = float((ew * u_m).item())
            else:
                mm = _weighted_mean(mse, None, m) if m is not None else torch.mean(mse)
                l_pred = l_pred + (ew * mm)
                l_pred_u = l_pred_u + (ew * mm)
                pred_by_edge_w[etype] = float((ew * mm).item())
                pred_by_edge_u[etype] = float((ew * mm).item())

    # Direct trajectory-level latent supervision.
    # These targets come from posterior state estimation packets and provide
    # denser supervision for bridge/signaling/crosstalk when edge labels are sparse.
    global_latent_terms = (
        ("global_bridge_state", "global_bridge_state"),
        ("global_signaling_state", "global_signaling_state"),
        ("global_crosstalk_state", "global_crosstalk_state"),
    )
    for out_k, tgt_k in global_latent_terms:
        pred = outputs.get(out_k)
        target = targets.get(tgt_k)
        if pred is None or target is None or pred.numel() == 0 or target.numel() == 0:
            continue
        n = min(pred.numel(), target.numel())
        if n <= 0:
            continue
        p = pred[:n]
        t = target[:n]
        m = None
        if target_masks is not None:
            raw_m = target_masks.get(tgt_k)
            if raw_m is not None and raw_m.numel() > 0:
                m = raw_m.reshape(-1)[:n].to(dtype=torch.bool, device=p.device)
        mse = (p - t) ** 2
        if sample_weights is not None and bsz > 0:
            per_sample = mse.reshape(-1)[:bsz]
            per_sample_has = m.reshape(-1)[:bsz] if m is not None else None
            w_m = _weighted_mean(per_sample, sample_weights, per_sample_has)
            u_m = _weighted_mean(per_sample, None, per_sample_has)
            l_pred = l_pred + (float(cfg.lambda_global_latent) * w_m)
            l_pred_u = l_pred_u + (float(cfg.lambda_global_latent) * u_m)
            pred_by_edge_w[out_k] = float((float(cfg.lambda_global_latent) * w_m).item())
            pred_by_edge_u[out_k] = float((float(cfg.lambda_global_latent) * u_m).item())
        else:
            mm = _weighted_mean(mse, None, m) if m is not None else torch.mean(mse)
            l_pred = l_pred + (float(cfg.lambda_global_latent) * mm)
            l_pred_u = l_pred_u + (float(cfg.lambda_global_latent) * mm)
            pred_by_edge_w[out_k] = float((float(cfg.lambda_global_latent) * mm).item())
            pred_by_edge_u[out_k] = float((float(cfg.lambda_global_latent) * mm).item())

    why_today_pred = outputs.get("why_today_latents")
    why_today_logits = outputs.get("why_today_latent_logits")
    why_today_target = targets.get("why_today_latents")
    why_today_teacher = targets.get("teacher_why_today_latents")
    l_top_driver = torch.tensor(0.0, device=_get_device(outputs))
    l_top_driver_u = torch.tensor(0.0, device=_get_device(outputs))
    l_contrast = torch.tensor(0.0, device=_get_device(outputs))
    l_contrast_u = torch.tensor(0.0, device=_get_device(outputs))
    l_family = torch.tensor(0.0, device=_get_device(outputs))
    l_family_u = torch.tensor(0.0, device=_get_device(outputs))
    l_teacher = torch.tensor(0.0, device=_get_device(outputs))
    l_teacher_u = torch.tensor(0.0, device=_get_device(outputs))
    if (
        why_today_pred is not None
        and why_today_target is not None
        and why_today_pred.numel() > 0
        and why_today_target.numel() > 0
    ):
        n = min(why_today_pred.shape[-1], why_today_target.shape[-1])
        p = why_today_pred[..., :n]
        t = why_today_target[..., :n]
        m = None
        if target_masks is not None:
            raw_m = target_masks.get("why_today_latents")
            if raw_m is not None and raw_m.numel() > 0:
                m = raw_m[..., :n].to(dtype=torch.bool, device=p.device)
        mse = (p - t) ** 2
        if sample_weights is not None and bsz > 0:
            per_sample = _reduce_per_sample_masked(mse, m, bsz) if m is not None else _reduce_per_sample(mse, bsz)
            per_sample_has = _reduce_per_sample_has(m, bsz) if m is not None else None
            w_m = _weighted_mean(per_sample, sample_weights, per_sample_has)
            u_m = _weighted_mean(per_sample, None, per_sample_has)
            l_pred = l_pred + (float(cfg.lambda_global_latent) * w_m)
            l_pred_u = l_pred_u + (float(cfg.lambda_global_latent) * u_m)
            pred_by_edge_w["why_today_latents"] = float((float(cfg.lambda_global_latent) * w_m).item())
            pred_by_edge_u["why_today_latents"] = float((float(cfg.lambda_global_latent) * u_m).item())
        else:
            mm = _weighted_mean(mse, None, m) if m is not None else torch.mean(mse)
            l_pred = l_pred + (float(cfg.lambda_global_latent) * mm)
            l_pred_u = l_pred_u + (float(cfg.lambda_global_latent) * mm)
            pred_by_edge_w["why_today_latents"] = float((float(cfg.lambda_global_latent) * mm).item())
            pred_by_edge_u["why_today_latents"] = float((float(cfg.lambda_global_latent) * mm).item())

        logits = (
            why_today_logits[..., :n].reshape(-1, n)
            if why_today_logits is not None and why_today_logits.numel() > 0
            else p.reshape(-1, n)
        )
        target_rows = t.reshape(-1, n)
        mask_rows = m.reshape(-1, n) if m is not None and m.ndim >= 2 else None
        if mask_rows is not None:
            valid_rows = mask_rows.any(dim=1)
            masked_target = target_rows.masked_fill(~mask_rows, float("-inf"))
            target_idx = masked_target.argmax(dim=1)
        else:
            valid_rows = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
            target_idx = target_rows.argmax(dim=1)
        if bool(valid_rows.any()):
            ce = F.cross_entropy(logits[valid_rows], target_idx[valid_rows], reduction="none")
            if sample_weights is not None and bsz > 0:
                ce_rows = ce.reshape(-1)[:bsz]
                valid_ce = torch.ones_like(ce_rows, dtype=torch.bool)
                l_top_driver = float(cfg.lambda_top_driver) * _weighted_mean(ce_rows, sample_weights, valid_ce)
                l_top_driver_u = float(cfg.lambda_top_driver) * _weighted_mean(ce_rows, None, valid_ce)
            else:
                ce_mean = ce.mean()
                l_top_driver = float(cfg.lambda_top_driver) * ce_mean
                l_top_driver_u = float(cfg.lambda_top_driver) * ce_mean

            pair_losses = []
            pair_weights = []
            margin = float(cfg.contrastive_driver_margin)
            for left_idx, right_idx in _HARD_WHY_TODAY_PAIRS:
                if left_idx >= n or right_idx >= n:
                    continue
                left_t = target_rows[:, left_idx]
                right_t = target_rows[:, right_idx]
                direction = torch.sign(left_t - right_t)
                valid_pair = direction != 0
                if mask_rows is not None:
                    valid_pair = valid_pair & mask_rows[:, left_idx] & mask_rows[:, right_idx]
                if not bool(valid_pair.any()):
                    continue
                diff = logits[:, left_idx] - logits[:, right_idx]
                hinge = F.relu(margin - direction[valid_pair] * diff[valid_pair])
                pair_losses.append(hinge)
                if sample_weights is not None and bsz > 0:
                    pair_weights.append(sample_weights[valid_pair[:bsz]])
            if pair_losses:
                stacked = torch.cat(pair_losses, dim=0)
                if pair_weights and sum(int(w.numel()) for w in pair_weights) == int(stacked.numel()):
                    stacked_w = torch.cat(pair_weights, dim=0)
                    l_contrast = float(cfg.lambda_contrastive_driver) * torch.sum(stacked * stacked_w) / torch.clamp(stacked_w.sum(), min=1e-8)
                else:
                    l_contrast = float(cfg.lambda_contrastive_driver) * stacked.mean()
                l_contrast_u = float(cfg.lambda_contrastive_driver) * stacked.mean()

        family_logits = outputs.get("why_today_family_logits")
        if family_logits is not None and family_logits.numel() > 0:
            fam_n = min(family_logits.shape[-1], len(WHY_TODAY_FAMILY_NAMES))
            fam_logits = family_logits[..., :fam_n].reshape(-1, fam_n)
            if mask_rows is not None:
                valid_rows = mask_rows.any(dim=1)
                masked_target = target_rows.masked_fill(~mask_rows, float("-inf"))
                target_idx = masked_target.argmax(dim=1)
            else:
                valid_rows = torch.ones(fam_logits.shape[0], dtype=torch.bool, device=fam_logits.device)
                target_idx = target_rows.argmax(dim=1)
            if bool(valid_rows.any()):
                fam_target_idx = torch.tensor(
                    [_FAMILY_INDEX[WHY_TODAY_LATENT_TO_FAMILY[WHY_TODAY_LATENT_NAMES[int(i.item())]]] for i in target_idx[valid_rows]],
                    device=fam_logits.device,
                    dtype=torch.long,
                )
                fam_ce = F.cross_entropy(fam_logits[valid_rows], fam_target_idx, reduction="none")
                if sample_weights is not None and bsz > 0:
                    fam_rows = fam_ce.reshape(-1)[:bsz]
                    valid_fam = torch.ones_like(fam_rows, dtype=torch.bool)
                    l_family = float(cfg.lambda_family_driver) * _weighted_mean(fam_rows, sample_weights, valid_fam)
                    l_family_u = float(cfg.lambda_family_driver) * _weighted_mean(fam_rows, None, valid_fam)
                else:
                    fam_mean = fam_ce.mean()
                    l_family = float(cfg.lambda_family_driver) * fam_mean
                    l_family_u = float(cfg.lambda_family_driver) * fam_mean

        if why_today_teacher is not None and why_today_teacher.numel() > 0:
            teacher_n = min(p.shape[-1], why_today_teacher.shape[-1])
            teacher_target = why_today_teacher[..., :teacher_n].to(dtype=p.dtype, device=p.device)
            teacher_mask = None
            if target_masks is not None:
                raw_tm = target_masks.get("teacher_why_today_latents")
                if raw_tm is not None and raw_tm.numel() > 0:
                    teacher_mask = raw_tm[..., :teacher_n].to(dtype=torch.bool, device=p.device)
            teacher_mse = (p[..., :teacher_n] - teacher_target) ** 2
            if sample_weights is not None and bsz > 0:
                per_sample = _reduce_per_sample_masked(teacher_mse, teacher_mask, bsz) if teacher_mask is not None else _reduce_per_sample(teacher_mse, bsz)
                per_sample_has = _reduce_per_sample_has(teacher_mask, bsz) if teacher_mask is not None else None
                l_teacher = float(cfg.lambda_teacher_distill) * _weighted_mean(per_sample, sample_weights, per_sample_has)
                l_teacher_u = float(cfg.lambda_teacher_distill) * _weighted_mean(per_sample, None, per_sample_has)
            else:
                mm = _weighted_mean(teacher_mse, None, teacher_mask) if teacher_mask is not None else torch.mean(teacher_mse)
                l_teacher = float(cfg.lambda_teacher_distill) * mm
                l_teacher_u = float(cfg.lambda_teacher_distill) * mm

    posterior_gain_pred = outputs.get("posterior_gain")
    l_gain = torch.tensor(0.0, device=_get_device(outputs))
    l_gain_u = torch.tensor(0.0, device=_get_device(outputs))
    if (
        posterior_gain_pred is not None
        and posterior_gain_target is not None
        and posterior_gain_pred.numel() > 0
        and posterior_gain_target.numel() > 0
    ):
        n = min(posterior_gain_pred.numel(), posterior_gain_target.numel())
        if n > 0:
            p = posterior_gain_pred.reshape(-1)[:n]
            t = posterior_gain_target.reshape(-1)[:n].to(dtype=p.dtype, device=p.device)
            m = None
            if posterior_gain_mask is not None and posterior_gain_mask.numel() > 0:
                m = posterior_gain_mask.reshape(-1)[:n].to(dtype=torch.bool, device=p.device)
            mse = (p - t) ** 2
            if sample_weights is not None and bsz > 0:
                per_sample = mse.reshape(-1)[:bsz]
                per_sample_has = m.reshape(-1)[:bsz] if m is not None else None
                l_gain = float(cfg.lambda_posterior_gain) * _weighted_mean(per_sample, sample_weights, per_sample_has)
                l_gain_u = float(cfg.lambda_posterior_gain) * _weighted_mean(per_sample, None, per_sample_has)
            else:
                mm = _weighted_mean(mse, None, m) if m is not None else torch.mean(mse)
                l_gain = float(cfg.lambda_posterior_gain) * mm
                l_gain_u = float(cfg.lambda_posterior_gain) * mm

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Smoothness Loss (L2 Regularization)
    #
    # L_smooth = Σ_e mean((hidden - baseline)²)
    #
    # This is Tikhonov regularization / ridge penalty centered at each edge
    # type's biological baseline (not zero):
    # - modulates/regulates: baseline = 0.0 (additive identity — no effect)
    # - signaling/transports_to: baseline = 1.0 (multiplicative identity)
    # - bridges: baseline = 1.0 (fully saturated — healthy population avg)
    #
    # Centering at 0 would push multiplicative edges (signaling, transports_to,
    # bridges) away from their biologically correct neutral state toward zero,
    # biasing predictions toward signal attenuation / cofactor depletion.
    #
    # Reference: Tikhonov (1943), Hoerl & Kennard (1970)
    # ─────────────────────────────────────────────────────────────────────────
    # Baselines per edge type: additive edges center at 0, multiplicative at 1.
    _smooth_baselines = {
        "modulates": 0.0,
        "regulates": 0.0,
        "signaling": 1.0,
        "bridges": 1.0,
        "transports_to": 1.0,
    }
    l_smooth = torch.tensor(0.0, device=_get_device(outputs))
    l_smooth_u = torch.tensor(0.0, device=_get_device(outputs))
    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        if pred is not None and pred.numel() > 0:
            baseline = _smooth_baselines.get(etype, 0.0)
            sq = (pred - baseline) ** 2
            if sample_weights is not None and bsz > 0:
                per_sample = _reduce_per_sample(sq, bsz)
                l_smooth = l_smooth + _weighted_mean(per_sample, sample_weights)
                l_smooth_u = l_smooth_u + torch.mean(per_sample)
            else:
                m = torch.mean(sq)
                l_smooth = l_smooth + m
                l_smooth_u = l_smooth_u + m

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Bridge Coherence Loss (Placeholder)
    #
    # Bridges between the same pathway pair should agree:
    # L_bridge = Σ_pair Var(saturations in pair)
    #
    # Uses _bridge_pathway_groups from targets dict (built during dataset construction).
    # ─────────────────────────────────────────────────────────────────────────
    l_bridge = torch.tensor(0.0, device=_get_device(outputs))
    # Bridge coherence: bridges sharing a pathway pair should predict similar
    # saturations. L_bridge = sum over pathway pairs of Var(bridge saturations).
    bridges_hidden = outputs.get("bridges_hidden")
    if bridges_hidden is not None and bridges_hidden.numel() > 1:
        pathway_groups = targets.get("_bridge_pathway_groups")  # dict: pair_key → list[int]
        if pathway_groups:
            for pair_key, indices in pathway_groups.items():
                if len(indices) < 2:
                    continue
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=bridges_hidden.device)
                valid = idx_tensor[idx_tensor < bridges_hidden.numel()]
                if valid.numel() < 2:
                    continue
                group = bridges_hidden[valid]
                l_bridge = l_bridge + torch.var(group)

    # ─────────────────────────────────────────────────────────────────────────
    # 3b. Physics Residual Loss (optional)
    #
    # This term should be computed from flux conservation / stoichiometric
    # imbalance by the training loop and passed in as `physics_residual`.
    # We keep it explicit to make mass-balance penalties auditable.
    # ─────────────────────────────────────────────────────────────────────────
    l_physics = torch.tensor(0.0, device=_get_device(outputs))
    l_physics_u = torch.tensor(0.0, device=_get_device(outputs))
    physics_rows_used = 0
    if physics_residual is not None:
        if sample_weights is not None and bsz > 0:
            if physics_residual.ndim == 0:
                per_sample_phys = physics_residual.reshape(1).repeat(bsz)
            elif physics_residual.ndim == 1:
                if physics_residual.numel() >= bsz:
                    per_sample_phys = physics_residual[:bsz]
                else:
                    pad = torch.zeros(bsz - physics_residual.numel(), device=physics_residual.device)
                    per_sample_phys = torch.cat([physics_residual, pad], dim=0)
            else:
                if physics_residual.shape[0] == bsz:
                    per_sample_phys = torch.mean(physics_residual ** 2, dim=tuple(range(1, physics_residual.ndim)))
                else:
                    per_sample_phys = _reduce_per_sample(physics_residual ** 2, bsz)
            m = physics_mask if physics_mask is not None else torch.ones_like(per_sample_phys, dtype=torch.bool)
            l_physics = _weighted_mean(per_sample_phys, sample_weights, m)
            l_physics_u = _weighted_mean(per_sample_phys, None, m)
            physics_rows_used = int(torch.count_nonzero(m).item())
        else:
            l_physics = torch.mean(physics_residual ** 2)
            l_physics_u = l_physics
            physics_rows_used = int(physics_residual.numel())

    # Optional vector-level physics supervision.
    # When available, this provides metabolite-localized residual penalties
    # rather than only a scalar RMS summary, improving bridge/signaling locality.
    l_physics_vec = torch.tensor(0.0, device=_get_device(outputs))
    l_physics_vec_u = torch.tensor(0.0, device=_get_device(outputs))
    physics_vec_rows_used = 0
    if physics_residual_vector is not None:
        if physics_residual_vector.ndim == 1:
            physics_residual_vector = physics_residual_vector.reshape(1, -1)
        per_sample_vec = torch.mean(physics_residual_vector ** 2, dim=1)
        vec_mask = (
            physics_vector_mask
            if physics_vector_mask is not None
            else torch.ones_like(per_sample_vec, dtype=torch.bool)
        )
        if sample_weights is not None and bsz > 0:
            l_physics_vec = _weighted_mean(per_sample_vec, sample_weights, vec_mask)
            l_physics_vec_u = _weighted_mean(per_sample_vec, None, vec_mask)
        else:
            l_physics_vec = _weighted_mean(per_sample_vec, None, vec_mask)
            l_physics_vec_u = l_physics_vec
        physics_vec_rows_used = int(torch.count_nonzero(vec_mask).item())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Confidence Calibration Loss
    #
    # L_conf = MSE(confidence, accuracy)
    # where accuracy = 1 - |pred - target| / output_range
    #
    # A well-calibrated model has:
    #   E[error | confidence = c] = (1 - c) × range
    #
    # This ensures confidence is meaningful for downstream use:
    # - Flux engine blends predictions by confidence
    # - Users can trust high-confidence predictions
    #
    # Reference: Guo et al. (2017) ICML. "On Calibration of Modern NNs"
    # ─────────────────────────────────────────────────────────────────────────
    l_conf = torch.tensor(0.0, device=_get_device(outputs))
    l_conf_u = torch.tensor(0.0, device=_get_device(outputs))
    ranges = {
        "modulates": cfg.modulates_range[1] - cfg.modulates_range[0],
        "regulates": cfg.regulates_range[1] - cfg.regulates_range[0],
        "signaling": cfg.signaling_range[1] - cfg.signaling_range[0],
        "bridges": cfg.bridge_range[1] - cfg.bridge_range[0],
        "transports_to": cfg.transport_range[1] - cfg.transport_range[0],
    }

    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        conf = outputs.get(f"{etype}_conf")
        target = targets.get(etype)

        if pred is not None and conf is not None and target is not None and pred.numel() > 0:
            n = min(pred.numel(), target.numel(), conf.numel())
            if n <= 0:
                continue
            p = pred[:n]
            t = target[:n]
            c = conf[:n]
            m = None
            if target_masks is not None:
                raw_m = target_masks.get(etype)
                if raw_m is not None and raw_m.numel() > 0:
                    m = raw_m.reshape(-1)[:n].to(dtype=torch.bool, device=p.device)
            # Compute accuracy: 1 - normalized absolute error
            abs_error = torch.abs(p - t)
            accuracy = 1.0 - (abs_error / (ranges[etype] + 1e-6))
            accuracy = accuracy.clamp(0.0, 1.0)
            conf_mse = (c - accuracy) ** 2
            if sample_weights is not None and bsz > 0:
                if m is not None:
                    per_sample = _reduce_per_sample_masked(conf_mse, m, bsz)
                    per_sample_has = _reduce_per_sample_has(m, bsz)
                else:
                    per_sample = _reduce_per_sample(conf_mse, bsz)
                    per_sample_has = None
                l_conf = l_conf + _weighted_mean(per_sample, sample_weights, per_sample_has)
                l_conf_u = l_conf_u + _weighted_mean(per_sample, None, per_sample_has)
            else:
                mm = _weighted_mean(conf_mse, None, m) if m is not None else torch.mean(conf_mse)
                l_conf = l_conf + mm
                l_conf_u = l_conf_u + mm

    # -------------------------------------------------------------------------
    # 5. Cross-head coupling (biological consistency)
    #
    # Enforce lightweight consistency:
    # signaling -> bridge saturation -> enzyme-control heads
    # This prevents heads from drifting independently when supervision is sparse.
    # -------------------------------------------------------------------------
    l_coupling = torch.tensor(0.0, device=_get_device(outputs))
    l_coupling_u = torch.tensor(0.0, device=_get_device(outputs))
    sig_h = outputs.get("signaling_hidden")
    br_h = outputs.get("bridges_hidden")
    mod_h = outputs.get("modulates_hidden")
    reg_h = outputs.get("regulates_hidden")
    if sig_h is not None and br_h is not None and sig_h.numel() > 0 and br_h.numel() > 0:
        sig_center = torch.mean(sig_h) - 1.0
        br_center = torch.mean(br_h) - 1.0
        enz_drive = torch.tensor(0.0, device=sig_h.device)
        if mod_h is not None and mod_h.numel() > 0:
            enz_drive = enz_drive + torch.mean(torch.abs(mod_h))
        if reg_h is not None and reg_h.numel() > 0:
            enz_drive = enz_drive + torch.mean(reg_h)
        target_bridge = torch.tanh(sig_center + 0.5 * enz_drive)
        l_coupling = l_coupling + (br_center - target_bridge) ** 2
        l_coupling_u = l_coupling_u + (br_center - target_bridge) ** 2
    g_sig = outputs.get("global_signaling_state")
    g_br = outputs.get("global_bridge_state")
    g_cr = outputs.get("global_crosstalk_state")
    if g_sig is not None and g_br is not None and g_cr is not None and g_sig.numel() > 0:
        n = min(g_sig.numel(), g_br.numel(), g_cr.numel())
        if n > 0:
            tgt = torch.tanh(g_sig.reshape(-1)[:n] + 0.5 * g_cr.reshape(-1)[:n])
            gv = torch.mean((g_br.reshape(-1)[:n] - tgt) ** 2)
            l_coupling = l_coupling + gv
            l_coupling_u = l_coupling_u + gv

    # -------------------------------------------------------------------------
    # Total loss
    # -------------------------------------------------------------------------
    smooth_raw = cfg.lambda_smooth * l_smooth
    smooth_u_raw = cfg.lambda_smooth * l_smooth_u
    conf_raw = cfg.lambda_confidence * l_conf
    conf_u_raw = cfg.lambda_confidence * l_conf_u
    smooth_eff = smooth_raw
    smooth_u_eff = smooth_u_raw
    conf_eff = conf_raw
    conf_u_eff = conf_u_raw

    # Keep regularization/calibration from dominating when predictive signal is weak.
    # This forces optimization pressure onto supervised heads first.
    pred_ref = torch.clamp(l_pred.detach(), min=1e-8)
    pred_ref_u = torch.clamp(l_pred_u.detach(), min=1e-8)
    if float(smooth_max_ratio_vs_pred) >= 0.0:
        cap = float(smooth_max_ratio_vs_pred) * pred_ref
        cap_u = float(smooth_max_ratio_vs_pred) * pred_ref_u
        smooth_eff = torch.minimum(smooth_raw, cap)
        smooth_u_eff = torch.minimum(smooth_u_raw, cap_u)
    if float(conf_max_ratio_vs_pred) >= 0.0:
        cap = float(conf_max_ratio_vs_pred) * pred_ref
        cap_u = float(conf_max_ratio_vs_pred) * pred_ref_u
        conf_eff = torch.minimum(conf_raw, cap)
        conf_u_eff = torch.minimum(conf_u_raw, cap_u)

    primary_total = l_pred + l_gain + l_top_driver + l_contrast + l_family + l_teacher
    regularization_total = (
        cfg.lambda_physics * (l_physics + l_physics_vec)
        + smooth_eff
        + cfg.lambda_bridge * l_bridge
        + conf_eff
        + cfg.lambda_coupling * l_coupling
    )
    primary_total_u = l_pred_u + l_gain_u + l_top_driver_u + l_contrast_u + l_family_u + l_teacher_u
    regularization_total_u = (
        cfg.lambda_physics * (l_physics_u + l_physics_vec_u)
        + smooth_u_eff
        + cfg.lambda_bridge * l_bridge
        + conf_u_eff
        + cfg.lambda_coupling * l_coupling_u
    )

    total = (
        l_pred
        + l_gain
        + l_top_driver
        + l_contrast
        + l_family
        + l_teacher
        + cfg.lambda_physics * (l_physics + l_physics_vec)
        + smooth_eff
        + cfg.lambda_bridge * l_bridge
        + conf_eff
        + cfg.lambda_coupling * l_coupling
    )
    total_u = (
        l_pred_u
        + l_gain_u
        + l_top_driver_u
        + l_contrast_u
        + l_family_u
        + l_teacher_u
        + cfg.lambda_physics * (l_physics_u + l_physics_vec_u)
        + smooth_u_eff
        + cfg.lambda_bridge * l_bridge
        + conf_u_eff
        + cfg.lambda_coupling * l_coupling_u
    )

    breakdown = {
        "pred": l_pred.item(),
        "pred_unweighted": l_pred_u.item(),
        "top_driver": float(l_top_driver.item()),
        "top_driver_unweighted": float(l_top_driver_u.item()),
        "contrastive_driver": float(l_contrast.item()),
        "contrastive_driver_unweighted": float(l_contrast_u.item()),
        "family_driver": float(l_family.item()),
        "family_driver_unweighted": float(l_family_u.item()),
        "teacher_distill": float(l_teacher.item()),
        "teacher_distill_unweighted": float(l_teacher_u.item()),
        "posterior_gain": float(l_gain.item()),
        "posterior_gain_unweighted": float(l_gain_u.item()),
        "physics": l_physics.item(),
        "physics_unweighted": l_physics_u.item(),
        "physics_vector": l_physics_vec.item(),
        "physics_vector_unweighted": l_physics_vec_u.item(),
        "physics_rows_used": float(physics_rows_used),
        "physics_vector_rows_used": float(physics_vec_rows_used),
        "smooth": float(smooth_eff.item()),
        "smooth_unweighted": float(smooth_u_eff.item()),
        "smooth_raw": float(smooth_raw.item()),
        "smooth_raw_unweighted": float(smooth_u_raw.item()),
        "bridge": l_bridge.item(),
        "confidence": float(conf_eff.item()),
        "confidence_unweighted": float(conf_u_eff.item()),
        "confidence_raw": float(conf_raw.item()),
        "confidence_raw_unweighted": float(conf_u_raw.item()),
        "coupling": float((cfg.lambda_coupling * l_coupling).item()),
        "coupling_unweighted": float((cfg.lambda_coupling * l_coupling_u).item()),
        "coupling_raw": float(l_coupling.item()),
        "coupling_raw_unweighted": float(l_coupling_u.item()),
        "primary_total": float(primary_total.item()),
        "primary_total_unweighted": float(primary_total_u.item()),
        "regularization_total": float(regularization_total.item()),
        "regularization_total_unweighted": float(regularization_total_u.item()),
        "total": total.item(),
        "total_unweighted": total_u.item(),
    }
    for et in edge_types:
        breakdown[f"pred_{et}"] = float(pred_by_edge_w.get(et, 0.0))
        breakdown[f"pred_{et}_unweighted"] = float(pred_by_edge_u.get(et, 0.0))
    for out_k, _ in global_latent_terms:
        breakdown[f"pred_{out_k}"] = float(pred_by_edge_w.get(out_k, 0.0))
        breakdown[f"pred_{out_k}_unweighted"] = float(pred_by_edge_u.get(out_k, 0.0))
    breakdown["pred_why_today_latents"] = float(pred_by_edge_w.get("why_today_latents", 0.0))
    breakdown["pred_why_today_latents_unweighted"] = float(pred_by_edge_u.get("why_today_latents", 0.0))

    return total, breakdown


def _get_device(outputs: dict[str, Any]) -> torch.device:
    """Get device from first non-empty tensor in outputs."""
    for key in [
        "modulates_hidden", "regulates_hidden", "signaling_hidden",
        "bridges_hidden", "transports_to_hidden"
    ]:
        t = outputs.get(key)
        if t is not None and hasattr(t, "device"):
            return t.device
    return torch.device("cpu")


def _reduce_per_sample(x: torch.Tensor, bsz: int) -> torch.Tensor:
    if bsz <= 0:
        return torch.mean(x).reshape(1)
    n = int(x.numel())
    if n <= 0:
        return torch.zeros(bsz, device=x.device, dtype=x.dtype)
    per = max(1, n // bsz)
    keep = per * bsz
    if keep <= 0:
        return torch.mean(x).reshape(1).repeat(bsz)
    y = x.reshape(-1)[:keep].view(bsz, per)
    return torch.mean(y, dim=1)


def _reduce_per_sample_masked(x: torch.Tensor, mask: torch.Tensor, bsz: int) -> torch.Tensor:
    if bsz <= 0:
        return _weighted_mean(x, None, mask).reshape(1)
    n = int(min(x.numel(), mask.numel()))
    if n <= 0:
        return torch.zeros(bsz, device=x.device, dtype=x.dtype)
    per = max(1, n // bsz)
    keep = per * bsz
    if keep <= 0:
        return _weighted_mean(x[:n], None, mask[:n]).reshape(1).repeat(bsz)
    xv = x.reshape(-1)[:keep].view(bsz, per)
    mv = mask.reshape(-1)[:keep].view(bsz, per).to(dtype=torch.bool, device=x.device)
    num = torch.sum(xv * mv.to(dtype=xv.dtype), dim=1)
    den = torch.sum(mv.to(dtype=xv.dtype), dim=1)
    out = torch.zeros_like(num)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


def _reduce_per_sample_has(mask: torch.Tensor, bsz: int) -> torch.Tensor:
    if bsz <= 0:
        return torch.ones(1, dtype=torch.bool, device=mask.device)
    n = int(mask.numel())
    if n <= 0:
        return torch.zeros(bsz, dtype=torch.bool, device=mask.device)
    per = max(1, n // bsz)
    keep = per * bsz
    if keep <= 0:
        return torch.zeros(bsz, dtype=torch.bool, device=mask.device)
    mv = mask.reshape(-1)[:keep].view(bsz, per).to(dtype=torch.bool, device=mask.device)
    return torch.any(mv, dim=1)


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    v = values.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1).to(dtype=torch.bool, device=v.device)
        v = v[m]
        if weights is not None:
            w = weights.reshape(-1).to(device=v.device)[m]
        else:
            w = None
    else:
        w = weights.reshape(-1).to(device=v.device) if weights is not None else None
    if v.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    if w is None:
        return torch.mean(v)
    w = torch.clamp(w, min=0.0)
    denom = torch.sum(w) + 1e-8
    return torch.sum(v * w) / denom
