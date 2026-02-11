"""
Loss function for hidden state inference on learned edges.

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THE GNN DOES
# ═══════════════════════════════════════════════════════════════════════════════
#
# The GNN Calibrator is a HIDDEN STATE INFERENCE ENGINE.
#
# It does NOT predict metabolite concentrations directly.
# It does NOT predict enzyme activities directly.
#
# Instead, it infers HIDDEN STATES on LEARNED EDGES that represent:
#   - How strongly a metabolite modulates an enzyme (allosteric effects)
#   - How strongly an enzyme regulates another enzyme (feedback loops)
#   - How a signaling molecule affects pathway crosstalk
#   - How saturated a cofactor bridge is between pathways
#   - How a genetic variant (SNP) affects enzyme function
#
# These hidden states are then fed to the RUST FLUX ENGINE, which uses them
# as constraints in a deterministic Michaelis-Menten simulation to compute
# actual metabolite concentrations and reaction fluxes.
#
# ═══════════════════════════════════════════════════════════════════════════════
# WHY THIS ARCHITECTURE?
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. PHYSICS-INFORMED: The Rust engine encodes biochemistry (M-M kinetics).
#    The GNN only needs to infer what we DON'T know from first principles.
#
# 2. INTERPRETABLE: Hidden states have clear biological meaning.
#    A bridge saturation of 0.3 means "cofactor is 30% available".
#
# 3. AUDITABLE: Attention weights show which evidence the GNN used.
#    If it predicts low BH4 saturation, we can trace which edges influenced that.
#
# 4. PERSONALIZED: The affects edges encode SNP→enzyme effects.
#    Different genotypes produce different hidden states → different fluxes.
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
#   │  Node Types:  enzyme, metabolite, reaction, snp                         │
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
#   │    - affects: snp → enzyme (genetic variants)                           │
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
import torch.nn.functional as F

from .config import ModelConfig


def hidden_state_loss(
    outputs: dict[str, Any],
    targets: dict[str, torch.Tensor],
    cfg: ModelConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the full loss for hidden state inference.

    # Loss Components

    1. **L_pred**: Mean Squared Error on hidden states
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
        Hidden state predictions and confidence scores per edge type:
        - modulates_hidden: Tensor[num_modulates] — allosteric effects ∈ [-1, 1]
        - regulates_hidden: Tensor[num_regulates] — regulatory effects ∈ [0, 1]
        - signaling_hidden: Tensor[num_signaling] — pathway multipliers ∈ [0.5, 2]
        - bridges_hidden: Tensor[num_bridges] — cofactor saturations ∈ [0, 1]
        - affects_hidden: Tensor[num_affects] — SNP effects ∈ [0, 1] (P0)
        - *_conf: Tensor — confidence scores ∈ [0, 1] for each edge type
    targets : dict of ground-truth hidden states
        Keys: modulates, regulates, signaling, bridges, affects
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
    1. GNN predicts hidden states on learned edges
    2. Hidden states are fed to Rust engine as constraints
    3. Engine computes downstream metabolite concentrations
    4. Prediction error drives GNN training

    This creates a physics-informed loop where the GNN learns to infer
    hidden states that produce biologically plausible flux distributions.
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
    # All learned edge types, including P0 affects (SNP→enzyme personalization)
    edge_types = ["modulates", "regulates", "signaling", "bridges", "affects"]

    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        target = targets.get(etype)
        if pred is not None and target is not None and pred.numel() > 0:
            l_pred = l_pred + F.mse_loss(pred, target)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Smoothness Loss (L2 Regularization)
    #
    # L_smooth = Σ_e mean(hidden²)
    #
    # This is Tikhonov regularization / ridge penalty:
    # - Equivalent to Gaussian prior on hidden states centered at 0
    # - Encourages small deviations from baseline (Occam's razor)
    # - Reference: Tikhonov (1943), Hoerl & Kennard (1970)
    # ─────────────────────────────────────────────────────────────────────────
    l_smooth = torch.tensor(0.0, device=_get_device(outputs))
    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        if pred is not None and pred.numel() > 0:
            l_smooth = l_smooth + torch.mean(pred ** 2)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Bridge Coherence Loss (Placeholder)
    #
    # Bridges between the same pathway pair should agree:
    # L_bridge = Σ_pair Var(saturations in pair)
    #
    # Requires pathway annotations on edges — deferred to training loop.
    # ─────────────────────────────────────────────────────────────────────────
    l_bridge = torch.tensor(0.0, device=_get_device(outputs))
    # TODO: Implement once pathway edge annotations are available

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
    ranges = {
        "modulates": cfg.modulates_range[1] - cfg.modulates_range[0],
        "regulates": cfg.regulates_range[1] - cfg.regulates_range[0],
        "signaling": cfg.signaling_range[1] - cfg.signaling_range[0],
        "bridges": cfg.bridge_range[1] - cfg.bridge_range[0],
        "affects": cfg.affects_range[1] - cfg.affects_range[0],  # P0: SNP effects
    }

    for etype in edge_types:
        pred = outputs.get(f"{etype}_hidden")
        conf = outputs.get(f"{etype}_conf")
        target = targets.get(etype)

        if pred is not None and conf is not None and target is not None and pred.numel() > 0:
            # Compute accuracy: 1 - normalized absolute error
            abs_error = torch.abs(pred - target)
            accuracy = 1.0 - (abs_error / (ranges[etype] + 1e-6))
            accuracy = accuracy.clamp(0.0, 1.0)
            l_conf = l_conf + F.mse_loss(conf, accuracy)

    # -------------------------------------------------------------------------
    # Total loss
    # -------------------------------------------------------------------------
    total = (
        l_pred
        + cfg.lambda_smooth * l_smooth
        + cfg.lambda_bridge * l_bridge
        + cfg.lambda_confidence * l_conf
    )

    breakdown = {
        "pred": l_pred.item(),
        "smooth": l_smooth.item(),
        "bridge": l_bridge.item(),
        "confidence": l_conf.item(),
        "total": total.item(),
    }

    return total, breakdown


def _get_device(outputs: dict[str, Any]) -> torch.device:
    """Get device from first non-empty tensor in outputs."""
    for key in [
        "modulates_hidden", "regulates_hidden", "signaling_hidden",
        "bridges_hidden", "affects_hidden"  # P0: SNP→enzyme
    ]:
        t = outputs.get(key)
        if t is not None and hasattr(t, "device"):
            return t.device
    return torch.device("cpu")

