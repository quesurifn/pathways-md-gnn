"""
PathwayGNN — Single heterogeneous GNN for posterior correction inference.

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THIS MODEL DOES
# ═══════════════════════════════════════════════════════════════════════════════
#
# The PathwayGNN is a POSTERIOR CORRECTION ENGINE for metabolic pathways.
#
# INPUT:
#   - Heterogeneous graph with node types: enzyme, metabolite, reaction
#   - Node features from canonical seed data (Km, Vmax, concentrations, etc.)
#   - Edge features encoding evidence quality and biological semantics
#
# OUTPUT:
#   - Residual correction states on LEARNED edges (not physics edges):
#       • modulates_hidden: allosteric effects [-1, +1]
#       • regulates_hidden: regulatory strength [0, 1]
#       • signaling_hidden: pathway multipliers [0.5, 2.0]
#       • bridges_hidden: cofactor saturations [0, 1]
#       • transports_to_hidden: compartment transport [0, 2]
#   - Confidence scores for each prediction [0, 1]
#   - Attention weights for audit trace
#
# NOTE: SNP personalization is handled OUTSIDE the GNN via the DAG prior.
# The GNN operates on a genotype-aware prior packet and predicts bounded
# current-state corrections on top of that prior.
# See personalization-architecture.md for the full design rationale.
#
# DOWNSTREAM USE:
#   The correction states are fed to the RUST FLUX ENGINE, which uses them
#   as bounded posterior adjustments in a deterministic Michaelis-Menten simulation:
#
#       Km_eff = Km × (1 - modulates_hidden)
#       v = Vmax × [S] / (Km_eff + [S])
#       (SNP multipliers are applied separately via variant_kinetics.jsonl)
#
# WHY THIS DESIGN:
#   1. DAG prior estimates baseline mechanistic state
#   2. GNN infers what changed today relative to that baseline
#   3. Together: posterior correction with interpretable outputs
#
# ═══════════════════════════════════════════════════════════════════════════════

# Architecture Overview

The PathwayGNN follows state-of-the-art Graph Transformer patterns (2024/2025):

1. **Per-type input projection** → shared hidden_dim
2. **N Pre-LN Transformer layers** with:
   - KineticConv on physics edges (Michaelis-Menten form locked)
   - LearnedConv (scaled dot-product + edge bias) on learned edges
   - FFN sublayer per node type (4x expansion, GELU)
3. **Edge heads**: per learned-edge-type MLP → hidden state scalars

# Scientific Foundation

## Pre-LN Transformer Pattern

We use Pre-LayerNorm (LN before attention/FFN) instead of Post-LN:

```
Post-LN:  h' = LN(h + Attention(h))    # original Transformer (Vaswani 2017)
Pre-LN:   h' = h + Attention(LN(h))    # modern standard (Xiong 2020)
```

Pre-LN advantages:
- **Better gradient flow**: Gradients bypass LN via residual connection
- **No warmup needed**: Stable training from step 1
- **Faster convergence**: ~2x faster on small datasets

Reference:
- Xiong et al. (2020) ICML. "On Layer Normalization in the Transformer
  Architecture" — Proves Pre-LN has better gradient properties

## 4× FFN Expansion

The FFN sublayer uses 4× hidden dimension expansion:

```
FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂
where W₁: d → 4d, W₂: 4d → d
```

This is the standard Transformer ratio:
- **Why 4×?** Empirically optimal for capacity vs. compute tradeoff
- **Why not more?** Diminishing returns and overfitting risk for small data

Reference:
- Vaswani et al. (2017) NeurIPS. "Attention Is All You Need"

## GELU Activation

We use GELU (Gaussian Error Linear Unit) instead of ReLU:

```
GELU(x) = x · Φ(x)  where Φ is the CDF of N(0,1)
        ≈ 0.5x · (1 + tanh(√(2/π)(x + 0.044715x³)))
```

GELU advantages over ReLU:
- **Smooth**: No discontinuous gradient at x=0
- **Probabilistic interpretation**: Models stochastic regularization
- **Proven in Transformers**: Used by BERT, GPT, and most modern LLMs

Reference:
- Hendrycks & Gimpel (2016) arXiv:1606.08415. "Gaussian Error Linear Units"

## Heterogeneous Graph for Small Datasets

We explicitly model node and edge types rather than using homogeneous GNNs:

- **enzymes**: Catalytic proteins with kinetic parameters (Km, Vmax)
- **metabolites**: Chemical compounds with concentrations
- **reactions**: Biochemical transformations

Edge types:
- **Physics edges**: `catalyzes`, `substrate_of`, `produces`, `cofactor_for`
  → Use KineticConv (M-M form locked)
- **Learned edges**: `modulates`, `regulates`, `signaling`, `bridges`, `transports_to`
  → Use LearnedConv (Transformer attention with edge features)

Note: SNP personalization is handled outside the GNN. See personalization-architecture.md.

This heterogeneous design provides:
1. **Inductive bias**: Edge types constrain message semantics
2. **Interpretability**: Each edge type has clear biological meaning
3. **Data efficiency**: Fewer parameters per edge type than homogeneous

Reference:
- Schlichtkrull et al. (2018) ESWC. "Modeling Relational Data with Graph
  Convolutional Networks" (R-GCN)
- Wang et al. (2019) KDD. "Heterogeneous Graph Attention Network" (HAN)

## Hybrid Local/Global Message Passing

Following the GPS (General, Powerful, Scalable) recipe:
- **Local**: KineticConv uses only direct neighbors (physics edges)
- **Global**: LearnedConv can attend across longer paths (learned edges)

This hybrid approach captures both:
- Local biochemical constraints (enzyme-substrate relationships)
- Long-range dependencies (metabolic regulation, signaling)

Reference:
- Rampášek et al. (2022) NeurIPS. "Recipe for a General, Powerful, Scalable
  Graph Transformer" (GPS)

# Outputs

- **Hidden states**: Per-edge scalars for learned edges (saturations, effects)
- **Confidence scores**: Per-edge uncertainty estimates
- **Attention weights**: For audit trace and explainability
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from .config import ModelConfig, EdgeSemantic
from .graph import PHYSICS_ETYPES, LEARNED_ETYPES, EDGE_SEMANTICS
from .layers import KineticConv, LearnedConv, EdgeHead, ConfidenceHead, MoEEdgeHead
from .observation_features import OBSERVATION_FEATURE_DIM

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

# ═══════════════════════════════════════════════════════════════════════════
# Raw Feature Dimensions per Node Type (must match graph.py)
# ═══════════════════════════════════════════════════════════════════════════
#
# These dimensions encode node-specific features extracted from seed data:
#
#   enzyme: 6 features
#     [vmax, km, confidence, baseline, brain_expr_norm, brain_to_systemic_ratio]
#     - vmax: Maximum reaction velocity (normalized)
#     - km: Michaelis constant (normalized)
#     - confidence: Data quality score
#     - variant_modifier: Baseline for genetic effects
#
#   metabolite: 2 features
#     [baseline_concentration, molecular_weight]
#     - baseline_concentration: Typical physiological level
#     - molecular_weight: For diffusion/transport modeling
#
#   reaction: 2 features
#     [reversibility, stoichiometry_hash]
#     - reversibility: Is this reaction reversible?
#     - stoichiometry_hash: Compact encoding of stoichiometry
#
# NOTE: SNP personalization is handled outside the GNN via lookup tables.
# The GNN learns wild type biochemistry; SNPs are post-GNN multipliers.
# See personalization-architecture.md for the full design rationale.
#
# Reference: graph.py build_graph() for feature construction
# ═══════════════════════════════════════════════════════════════════════════
_RAW_DIMS = {"enzyme": 6, "metabolite": 2, "reaction": 2}


class PathwayGNN(nn.Module):
    """Single heterogeneous GNN for posterior metabolic correction inference.

    # Purpose

    The PathwayGNN infers bounded posterior corrections
    (cofactor saturation shifts, allosteric shifts, regulation shifts)
    from observable evidence conditioned on a DAG prior.

    # Architecture (SOTA Graph Transformer, 2024/2025)

    ```
    Raw features → [Input Projection] → hidden_dim
                        ↓
    N × [Pre-LN Transformer Layer]
        ├── KineticConv on physics edges (M-M locked)
        ├── LearnedConv on learned edges (attention)
        ├── Residual connection
        ├── FFN sublayer (4x, GELU)
        └── Residual connection
                        ↓
    [Edge Heads] → hidden states per learned edge
    [Confidence Heads] → confidence scores per edge
    ```

    # Key Design Choices

    1. **Pre-LN pattern** (Xiong et al., 2020): Better gradient flow, no warmup
    2. **Scaled dot-product attention** (Vaswani et al., 2017): Transformer standard
    3. **Edge features as attention bias** (Ying et al., 2021): Graphormer-style
    4. **Hybrid message passing** (Rampášek et al., 2022): GPS recipe
    5. **Confidence prediction**: Uncertainty quantification for each output

    # Why This Architecture?

    For small metabolic graphs (~100-1000 nodes):
    - **Heterogeneous GNN**: Strong inductive bias from edge types
    - **Physics constraints**: KineticConv enforces M-M form
    - **Global attention**: LearnedConv captures long-range regulation
    - **Interpretable outputs**: Per-edge scalars with confidence

    # References

    - Xiong et al. (2020) ICML. "On Layer Normalization in the Transformer"
    - Vaswani et al. (2017) NeurIPS. "Attention Is All You Need"
    - Ying et al. (2021) NeurIPS. "Graphormer"
    - Rampášek et al. (2022) NeurIPS. "GPS: General, Powerful, Scalable"
    - Schlichtkrull et al. (2018) ESWC. "R-GCN"
    """

    def __init__(
        self,
        cfg: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        h = cfg.hidden_dim
        edge_head_dim = cfg.edge_head_hidden_dim

        # -- 1. Input projections per node type ------------------------------
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(raw_d, h) for ntype, raw_d in _RAW_DIMS.items()
        })
        # Per-node-type context projection for request-conditioned inference.
        # Context is an 8-dim vector per node injected by calibrate():
        #   [0] flux delta, [1-3] symptom burden, [4-5] lifestyle,
        #   [6] intervention exposure, [7] genotype burden.
        self.context_proj = nn.ModuleDict({
            ntype: nn.Linear(8, h, bias=False) for ntype in _RAW_DIMS
        })
        self.observation_proj = nn.Linear(OBSERVATION_FEATURE_DIM, h, bias=False)

        # -- 2. Message-passing layers (physics + learned) -------------------
        self.mp_layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.mp_layers.append(_HeteroLayer(cfg))

        # -- 3. Edge heads (one per learned edge type) -----------------------
        # Input: src_h + dst_h concatenated
        edge_input_dim = h * 2

        if cfg.use_moe_heads:
            self.modulates_head = MoEEdgeHead(
                edge_input_dim,
                edge_head_dim,
                cfg.modulates_range,
                num_experts=cfg.enzyme_regulation_experts,
                gate_hidden_dim=cfg.moe_gate_hidden_dim,
                gate_feat_dim=14,
                dropout=cfg.dropout,
                gate_dropout=cfg.moe_gate_dropout,
            )
            self.regulates_head = MoEEdgeHead(
                edge_input_dim,
                edge_head_dim,
                cfg.regulates_range,
                num_experts=cfg.enzyme_regulation_experts,
                gate_hidden_dim=cfg.moe_gate_hidden_dim,
                gate_feat_dim=14,
                dropout=cfg.dropout,
                gate_dropout=cfg.moe_gate_dropout,
            )
            self.signaling_head = MoEEdgeHead(
                edge_input_dim,
                edge_head_dim,
                cfg.signaling_range,
                num_experts=cfg.crosstalk_signaling_experts,
                gate_hidden_dim=cfg.moe_gate_hidden_dim,
                gate_feat_dim=22,
                dropout=cfg.dropout,
                gate_dropout=cfg.moe_gate_dropout,
            )
            self.bridges_head = MoEEdgeHead(
                edge_input_dim,
                edge_head_dim,
                cfg.bridge_range,
                num_experts=cfg.cofactor_bridge_experts,
                gate_hidden_dim=cfg.moe_gate_hidden_dim,
                gate_feat_dim=11,
                dropout=cfg.dropout,
                gate_dropout=cfg.moe_gate_dropout,
            )
            self.transports_to_head = MoEEdgeHead(
                edge_input_dim,
                edge_head_dim,
                cfg.transport_range,
                num_experts=cfg.transport_context_experts,
                gate_hidden_dim=cfg.moe_gate_hidden_dim,
                gate_feat_dim=15,
                dropout=cfg.dropout,
                gate_dropout=cfg.moe_gate_dropout,
            )
        else:
            self.modulates_head = EdgeHead(
                edge_input_dim, edge_head_dim, cfg.modulates_range, cfg.dropout
            )
            self.regulates_head = EdgeHead(
                edge_input_dim, edge_head_dim, cfg.regulates_range, cfg.dropout
            )
            self.signaling_head = EdgeHead(
                edge_input_dim, edge_head_dim, cfg.signaling_range, cfg.dropout
            )
            self.bridges_head = EdgeHead(
                edge_input_dim, edge_head_dim, cfg.bridge_range, cfg.dropout
            )
            self.transports_to_head = EdgeHead(
                edge_input_dim, edge_head_dim, cfg.transport_range, cfg.dropout
            )

        # NOTE: affects_head removed — SNP personalization is handled in the DAG prior.
        # See personalization-architecture.md for the baseline DAG + posterior GNN design.

        # -- 4. Confidence heads (one per edge type) -------------------------
        # Each learned edge type has its own confidence predictor.
        # Confidence is used by the Rust flux engine to blend predictions:
        #   effective = baseline + (predicted - baseline) × confidence
        self.confidence_heads = nn.ModuleDict({
            "modulates": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "regulates": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "signaling": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "bridges": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "transports_to": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
        })
        # Global latent heads for direct bridge/signaling/crosstalk supervision.
        # This is used for trajectory-level posterior training.
        self.global_latent_head = nn.Sequential(
            nn.Linear(h * 3, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, 3),
        )
        self.global_latent_uncertainty_head = nn.Sequential(
            nn.Linear(h * 3, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, 3),
        )
        # Temporal update gate for global latent state rollout:
        # z_t = (1-a)*z_{t-1} + a*z_raw, with a in [0,1].
        self.global_temporal_gate = nn.Sequential(
            nn.Linear(h * 3, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, 3),
            nn.Sigmoid(),
        )
        why_today_input_dim = h * 4
        self.why_today_latent_head = nn.Sequential(
            nn.Linear(why_today_input_dim, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, len(WHY_TODAY_LATENT_NAMES)),
        )
        self.why_today_obs_only_head = nn.Sequential(
            nn.Linear(h, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, len(WHY_TODAY_LATENT_NAMES)),
        )
        self.why_today_obs_gate = nn.Sequential(
            nn.Linear(why_today_input_dim, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, len(WHY_TODAY_LATENT_NAMES)),
        )
        self.why_today_family_head = nn.Sequential(
            nn.Linear(why_today_input_dim, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, len(WHY_TODAY_FAMILY_NAMES)),
        )
        family_to_latent = torch.zeros(len(WHY_TODAY_FAMILY_NAMES), len(WHY_TODAY_LATENT_NAMES))
        fam_index = {name: idx for idx, name in enumerate(WHY_TODAY_FAMILY_NAMES)}
        for latent_idx, latent_name in enumerate(WHY_TODAY_LATENT_NAMES):
            family_to_latent[fam_index[WHY_TODAY_LATENT_TO_FAMILY[latent_name]], latent_idx] = 1.0
        self.register_buffer("why_today_family_to_latent", family_to_latent, persistent=False)
        self.why_today_family_bias_scale = nn.Parameter(torch.tensor(0.75))
        self.why_today_confidence_head = nn.Sequential(
            nn.Linear(why_today_input_dim, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, len(WHY_TODAY_LATENT_NAMES)),
        )
        self.posterior_gain_head = nn.Sequential(
            nn.Linear(why_today_input_dim, edge_head_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(edge_head_dim, 1),
        )

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
    ) -> dict[str, Any]:
        """Run inference on a heterograph.

        Expects:
            - g.nodes[ntype].data["feat"] to be populated for all node types
            - Node types: enzyme, metabolite, reaction

        Returns
        -------
        dict with keys:
            # Hidden states (per-edge predictions)
            # NOTE: LATENT_CONTEXT edges have hidden=0 (gated out)
            modulates_hidden  : (N_modulates_edges,) — allosteric effects
            regulates_hidden  : (N_regulates_edges,) — enzyme regulation strength
            signaling_hidden  : (N_signaling_edges,) — signal multipliers
            bridges_hidden    : (N_bridges_edges,) — cofactor saturations

            # Confidence scores (uncertainty quantification)
            modulates_conf    : (N_modulates_edges,) — prediction confidence
            regulates_conf    : (N_regulates_edges,) — prediction confidence
            signaling_conf    : (N_signaling_edges,) — prediction confidence
            bridges_conf      : (N_bridges_edges,) — prediction confidence

            # Edge semantics (gated propagation control)
            # 0 = METABOLIC_ANCHOR (affects simulation)
            # 1 = LATENT_CONTEXT (context only, hidden=0)
            modulates_semantic : (N_modulates_edges,) — semantic labels
            regulates_semantic : (N_regulates_edges,) — semantic labels
            signaling_semantic : (N_signaling_edges,) — semantic labels
            bridges_semantic   : (N_bridges_edges,) — semantic labels

            # Audit/traceability
            node_embeddings   : dict[ntype → (N, hidden_dim)] — final embeddings
            attention_weights : dict[etype → (N_edges, num_heads)] — attention

        Note: SNP personalization is handled outside the GNN in the DAG prior.
        See personalization-architecture.md for the baseline DAG + posterior GNN design.

        Edge Semantics:
        - METABOLIC_ANCHOR edges produce full hidden state outputs
        - LATENT_CONTEXT edges have hidden=0 (gated out at output stage)
        - Both participate in message passing for multi-hop reasoning
        See gnn-calibrator.md "Edge Semantics: Gated Propagation" for details.
        """
        # -- 1. Project raw features to hidden dim ----------------------------
        # Each node type has its own input projection to shared hidden_dim.
        # This is standard heterogeneous GNN practice (Schlichtkrull 2018).
        h_dict: dict[str, torch.Tensor] = {}
        for ntype, proj in self.input_proj.items():
            # Check if this node type exists in the graph
            if ntype in g.ntypes and g.num_nodes(ntype) > 0:
                h = torch.relu(proj(g.nodes[ntype].data["feat"]))
                if "ctx" in g.nodes[ntype].data:
                    h = h + self.context_proj[ntype](g.nodes[ntype].data["ctx"])
                h_dict[ntype] = h
            else:
                # Create empty tensor for missing node types
                # This allows forward pass even when some node types have no nodes
                h_dict[ntype] = torch.zeros(0, self.cfg.hidden_dim,
                                           device=next(proj.parameters()).device)

        # -- 2. Message passing (physics + learned) ---------------------------
        rel_graphs: dict[tuple[str, str, str], dgl.DGLHeteroGraph] = {}
        for key in PHYSICS_ETYPES:
            rel_graphs[key] = g[key]
        for key in LEARNED_ETYPES:
            rel_graphs[key] = g[key]

        # Track attention weights for all learned edge types (for audit trace)
        all_attn: dict[str, list[torch.Tensor]] = {
            "modulates": [], "regulates": [], "signaling": [], "bridges": [], "transports_to": [],
        }
        for layer in self.mp_layers:
            h_dict, layer_attn = layer(g, h_dict, rel_graphs=rel_graphs)
            for etype, attn in layer_attn.items():
                if attn is not None:
                    all_attn[etype].append(attn)

        # Average attention across layers for audit trace
        # This gives a single attention score per edge for interpretability
        attention_weights: dict[str, torch.Tensor] = {}
        for etype, attn_list in all_attn.items():
            if attn_list:
                attention_weights[etype] = torch.stack(attn_list).mean(dim=0)

        # -- 3. Edge heads: predict hidden states on learned edges ------------
        outputs: dict[str, Any] = {
            "node_embeddings": h_dict,
            "attention_weights": attention_weights,
        }
        g_emb = self._global_graph_embedding(g, h_dict)
        obs_emb = self._global_observation_embedding(g)
        why_today_input = torch.cat([g_emb, obs_emb], dim=1)
        g_lat = self.global_latent_head(g_emb)
        g_unc = F.softplus(self.global_latent_uncertainty_head(g_emb))
        g_alpha = self.global_temporal_gate(g_emb)
        g_prev = self._global_prev_latent(g)
        g_curr = torch.tanh(g_lat)
        if g_prev is not None and g_prev.shape == g_curr.shape:
            g_state = (1.0 - g_alpha) * g_prev + g_alpha * g_curr
        else:
            g_state = g_curr
        outputs.update(
            {
                "global_bridge_state": g_state[:, 0],
                "global_signaling_state": g_state[:, 1],
                "global_crosstalk_state": g_state[:, 2],
                "global_bridge_state_std": torch.clamp(g_unc[:, 0], min=1e-4),
                "global_signaling_state_std": torch.clamp(g_unc[:, 1], min=1e-4),
                "global_crosstalk_state_std": torch.clamp(g_unc[:, 2], min=1e-4),
                "global_temporal_alpha_bridge": g_alpha[:, 0],
                "global_temporal_alpha_signaling": g_alpha[:, 1],
                "global_temporal_alpha_crosstalk": g_alpha[:, 2],
            }
        )
        why_today_logits = self.why_today_latent_head(why_today_input)
        why_today_obs_logits = self.why_today_obs_only_head(obs_emb)
        why_today_obs_gate = torch.sigmoid(self.why_today_obs_gate(why_today_input))
        why_today_family_logits = self.why_today_family_head(why_today_input)
        why_today_family_probs = torch.softmax(why_today_family_logits, dim=1)
        family_bias = torch.matmul(why_today_family_probs, self.why_today_family_to_latent)
        why_today_logits = why_today_logits + why_today_obs_gate * why_today_obs_logits + self.why_today_family_bias_scale * family_bias
        why_today_conf = torch.sigmoid(self.why_today_confidence_head(why_today_input))
        outputs["why_today_obs_only_logits"] = why_today_obs_logits
        outputs["why_today_obs_gate"] = why_today_obs_gate
        outputs["why_today_family_logits"] = why_today_family_logits
        outputs["why_today_family_probs"] = why_today_family_probs
        outputs["why_today_latent_logits"] = why_today_logits
        outputs["why_today_latents"] = torch.sigmoid(why_today_logits)
        outputs["why_today_latents_conf"] = why_today_conf
        outputs["posterior_gain"] = torch.sigmoid(self.posterior_gain_head(why_today_input)).reshape(-1)
        for idx, name in enumerate(WHY_TODAY_LATENT_NAMES):
            outputs[f"why_today::{name}"] = outputs["why_today_latents"][:, idx]
            outputs[f"why_today::{name}::conf"] = outputs["why_today_latents_conf"][:, idx]

        # modulates: metabolite → enzyme (allosteric effects)
        outputs.update(self._predict_edge_hidden(
            g, "metabolite", "modulates", "enzyme",
            h_dict, self.modulates_head, "modulates"
        ))

        # regulates: enzyme → enzyme (transcriptional/feedback regulation)
        outputs.update(self._predict_edge_hidden(
            g, "enzyme", "regulates", "enzyme",
            h_dict, self.regulates_head, "regulates"
        ))

        # signaling: metabolite → metabolite (crosstalk pathways)
        outputs.update(self._predict_edge_hidden(
            g, "metabolite", "signaling", "metabolite",
            h_dict, self.signaling_head, "signaling"
        ))

        # bridges: metabolite → enzyme (cofactor saturation)
        outputs.update(self._predict_edge_hidden(
            g, "metabolite", "bridges", "enzyme",
            h_dict, self.bridges_head, "bridges"
        ))
        outputs.update(self._predict_edge_hidden(
            g, "metabolite", "transports_to", "metabolite",
            h_dict, self.transports_to_head, "transports_to"
        ))

        # NOTE: affects edges removed — SNP personalization is handled outside GNN.
        # See personalization-architecture.md for the wild type GNN + SNP lookup design.

        return outputs

    def _global_observation_embedding(self, g: dgl.DGLHeteroGraph) -> torch.Tensor:
        for ntype in ("enzyme", "metabolite", "reaction"):
            if ntype not in g.ntypes or g.num_nodes(ntype) <= 0:
                continue
            obs = g.nodes[ntype].data.get("obs")
            if obs is None or obs.numel() == 0:
                continue
            counts = g.batch_num_nodes(ntype) if hasattr(g, "batch_num_nodes") else None
            if counts is None or counts.numel() == 0:
                pooled = torch.mean(obs, dim=0, keepdim=True)
            else:
                pooled = []
                off = 0
                for c in counts.tolist():
                    cc = int(c)
                    if cc <= 0:
                        pooled.append(torch.zeros((obs.shape[-1],), device=obs.device))
                    else:
                        pooled.append(torch.mean(obs[off:off + cc], dim=0))
                    off += max(0, cc)
                pooled = torch.stack(pooled, dim=0)
            return torch.relu(self.observation_proj(pooled))
        return torch.zeros((1, self.cfg.hidden_dim), device=next(self.parameters()).device)

    def _global_graph_embedding(
        self,
        g: dgl.DGLHeteroGraph,
        h_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for ntype in ("enzyme", "metabolite", "reaction"):
            h = h_dict.get(ntype)
            if h is None:
                continue
            if h.numel() == 0:
                bsz = int(g.batch_size) if hasattr(g, "batch_size") else 1
                parts.append(torch.zeros((bsz, self.cfg.hidden_dim), device=next(self.parameters()).device))
                continue
            counts = g.batch_num_nodes(ntype) if hasattr(g, "batch_num_nodes") else None
            if counts is None or counts.numel() == 0:
                parts.append(torch.mean(h, dim=0, keepdim=True))
                continue
            c0 = int(counts[0].item())
            bsz = int(counts.numel())
            if c0 > 0 and int(torch.sum(counts).item()) == h.shape[0]:
                parts.append(h.view(bsz, c0, -1).mean(dim=1))
            else:
                # Fallback for variable-size batching.
                pooled = []
                off = 0
                for c in counts.tolist():
                    cc = int(c)
                    if cc <= 0:
                        pooled.append(torch.zeros((self.cfg.hidden_dim,), device=h.device))
                    else:
                        pooled.append(torch.mean(h[off:off + cc], dim=0))
                    off += max(0, cc)
                parts.append(torch.stack(pooled, dim=0))
        if not parts:
            return torch.zeros((1, self.cfg.hidden_dim * 3), device=next(self.parameters()).device)
        while len(parts) < 3:
            parts.append(torch.zeros_like(parts[0]))
        return torch.cat(parts[:3], dim=1)

    def _global_prev_latent(self, g: dgl.DGLHeteroGraph) -> torch.Tensor | None:
        if "enzyme" not in g.ntypes:
            return None
        n = g.num_nodes("enzyme")
        if n <= 0:
            return None
        if "prev_latent" not in g.nodes["enzyme"].data:
            return None
        t = g.nodes["enzyme"].data["prev_latent"]
        if t.ndim != 2 or t.shape[1] != 3:
            return None
        counts = g.batch_num_nodes("enzyme") if hasattr(g, "batch_num_nodes") else None
        if counts is None or counts.numel() == 0:
            return torch.mean(t, dim=0, keepdim=True)
        c0 = int(counts[0].item())
        bsz = int(counts.numel())
        if c0 > 0 and int(torch.sum(counts).item()) == t.shape[0]:
            return t.view(bsz, c0, 3).mean(dim=1)
        pooled = []
        off = 0
        for c in counts.tolist():
            cc = int(c)
            if cc <= 0:
                pooled.append(torch.zeros((3,), device=t.device))
            else:
                pooled.append(torch.mean(t[off:off + cc], dim=0))
            off += max(0, cc)
        return torch.stack(pooled, dim=0)

    def _predict_edge_hidden(
        self,
        g: dgl.DGLHeteroGraph,
        stype: str,
        etype: str,
        dtype: str,
        h_dict: dict[str, torch.Tensor],
        head: EdgeHead | MoEEdgeHead,
        name: str,
    ) -> dict[str, torch.Tensor]:
        """Predict hidden states for a learned edge type with semantic gating.

        Edge Semantics (Gated Propagation):
        - METABOLIC_ANCHOR edges: Full hidden state output (affects simulation)
        - LATENT_CONTEXT edges: Hidden state zeroed (context only, no simulation effect)

        The semantic is stored in edge data as "semantic" field:
        - 0 = METABOLIC_ANCHOR (simulation_affecting=True)
        - 1 = LATENT_CONTEXT (simulation_affecting=False)

        See gnn-calibrator.md "Edge Semantics: Gated Propagation" for details.
        """
        canonical = (stype, etype, dtype)
        if g.num_edges(canonical) == 0:
            empty = torch.tensor([], device=h_dict[stype].device)
            out = {
                f"{name}_hidden": empty,
                f"{name}_conf": empty,
                f"{name}_semantic": empty,
            }
            if isinstance(head, MoEEdgeHead):
                out[f"{name}_gate_weights"] = torch.zeros((0, head.num_experts), device=h_dict[stype].device)
            return out

        src, dst = g.edges(etype=canonical)
        src_h = h_dict[stype][src]
        dst_h = h_dict[dtype][dst]

        gate_feat = None
        if "feat" in g.edges[canonical].data:
            gate_feat = g.edges[canonical].data["feat"]
        if "semantic" in g.edges[canonical].data:
            sem = g.edges[canonical].data["semantic"].float().view(-1, 1)
            gate_feat = sem if gate_feat is None else torch.cat([gate_feat, sem], dim=1)

        head_out = head(src_h, dst_h, gate_feat=gate_feat) if isinstance(head, MoEEdgeHead) else head(src_h, dst_h)
        if isinstance(head_out, tuple):
            hidden, gate_weights = head_out
        else:
            hidden = head_out
            gate_weights = None
        conf = self.confidence_heads[name](src_h, dst_h)

        # Residual prediction: clamp(baseline + delta, output_range)
        #
        # Why residualization here (and not in data pre-processing):
        # - The edge head can stay expressive, but we explicitly anchor outputs
        #   to mechanistic "no-effect" baselines (e.g., signaling=1, bridges=1).
        # - This preserves interpretable baselines and limits drift in low-signal
        #   regions while still letting the model learn corrections.
        #
        # References:
        # - He et al. (2016) residual formulation H(x)=F(x)+x:
        #   https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        # - Hybrid mechanistic + residual model-error learning:
        #   https://clima.caltech.edu/wp-content/uploads/2023/03/essoar.10509956.1.pdf
        ranges: dict[str, tuple[float, float]] = {
            "modulates": self.cfg.modulates_range,
            "regulates": self.cfg.regulates_range,
            "signaling": self.cfg.signaling_range,
            "bridges": self.cfg.bridge_range,
            "transports_to": self.cfg.transport_range,
        }
        baselines: dict[str, float] = {
            "modulates": 0.0,
            "regulates": 0.0,
            "signaling": 1.0,
            "bridges": 1.0,
            "transports_to": 1.0,
        }
        lo, hi = ranges[name]
        baseline = baselines[name]
        # Clip learned delta before recombining with baseline. This guardrail is
        # intentionally conservative to avoid large early-training excursions that
        # can destabilize downstream simulation.
        delta = torch.clamp(
            hidden - baseline,
            min=-self.cfg.residual_delta_clip,
            max=self.cfg.residual_delta_clip,
        )
        hidden = torch.clamp(baseline + delta, min=lo, max=hi)

        # -- Apply semantic gating ------------------------------------------------
        # LATENT_CONTEXT edges (semantic=1) have their hidden states zeroed.
        # They still participate in message passing (updating node embeddings),
        # but their outputs don't affect the Rust simulation.
        #
        # This implements the gated propagation design:
        # - Hidden-state propagation captures multi-hop crosstalk (during MP)
        # - Metabolic outputs are constrained to anchor-enabled routes (here)
        if "semantic" in g.edges[canonical].data:
            semantic = g.edges[canonical].data["semantic"]
            # Create mask: 1.0 for METABOLIC_ANCHOR (0), 0.0 for LATENT_CONTEXT (1)
            anchor_mask = (semantic == 0).float()
            hidden = hidden * anchor_mask
            # Confidence is also zeroed for LATENT_CONTEXT (no prediction to trust)
            conf = conf * anchor_mask
        else:
            # Fallback: use default semantic from EDGE_SEMANTICS
            default_semantic = EDGE_SEMANTICS.get(etype, EdgeSemantic.LATENT_CONTEXT)
            if default_semantic == EdgeSemantic.LATENT_CONTEXT:
                hidden = hidden * 0.0
                conf = conf * 0.0
            semantic = torch.full(
                (hidden.shape[0],),
                0 if default_semantic == EdgeSemantic.METABOLIC_ANCHOR else 1,
                dtype=torch.long,
                device=hidden.device,
            )

        out = {
            f"{name}_hidden": hidden,
            f"{name}_conf": conf,
            f"{name}_semantic": semantic,
        }
        if gate_weights is not None:
            out[f"{name}_gate_weights"] = gate_weights
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Heterogeneous Transformer layer (Pre-LN + FFN)
# ═══════════════════════════════════════════════════════════════════════════

class _HeteroLayer(nn.Module):
    """Single Transformer-style layer: Pre-LN attention + Pre-LN FFN.

    # Architecture (Pre-LN Pattern)

    ```
    h' = h + Dropout(Attention(LN₁(h)))   # attention sublayer
    h" = h' + Dropout(FFN(LN₂(h')))       # FFN sublayer
    ```

    # Why Pre-LN over Post-LN?

    The original Transformer (Vaswani 2017) used Post-LN:
    ```
    Post-LN:  h' = LN(h + Sublayer(h))
    ```

    But Pre-LN (Xiong et al. 2020) has significant advantages:

    1. **Gradient flow**: In Pre-LN, the gradient from loss to early layers
       passes directly through residual connections, bypassing LayerNorm.
       Post-LN requires the gradient to pass through LN, which can amplify
       or dampen gradients unpredictably.

    2. **No warmup**: Pre-LN is stable from step 1. Post-LN often needs
       learning rate warmup to prevent early training instability.

    3. **Faster convergence**: Empirically ~2x faster on small datasets
       (Liu et al. 2020, "Understanding the Difficulty of Training Transformers")

    # LayerNorm Mechanics

    LayerNorm normalizes across the feature dimension:
    ```
    LN(x) = γ · (x - μ) / (σ + ε) + β
    where μ, σ are computed per-sample over features
    ```

    Unlike BatchNorm (which normalizes across batch), LayerNorm:
    - Works with any batch size (including 1)
    - Is sample-independent (deterministic at inference)
    - Is the standard for Transformers

    Reference:
    - Ba et al. (2016) arXiv:1607.06450. "Layer Normalization"

    # Dropout Placement

    We apply dropout AFTER sublayer, BEFORE residual addition:
    ```
    h' = h + Dropout(Sublayer(LN(h)))
    ```

    This is the standard Transformer pattern:
    - Regularizes the sublayer output
    - Residual connection provides gradient highway

    # References

    - Vaswani et al. (2017) NeurIPS. "Attention Is All You Need" (original)
    - Xiong et al. (2020) ICML. "On Layer Normalization in the Transformer"
    - Ba et al. (2016) arXiv. "Layer Normalization"
    - Liu et al. (2020) ICML. "Understanding the Difficulty of Training
      Transformers"
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.hidden_dim

        # ─── Attention Sublayer ───────────────────────────────────────────
        # Two types of message passing, used in parallel:

        # 1. KineticConv per physics edge type (Michaelis-Menten, no attention)
        #    Physics edges are deterministic — no need for learned attention
        self.physics_convs = nn.ModuleDict()
        for _, etype, _ in PHYSICS_ETYPES:
            self.physics_convs[etype] = KineticConv(h)

        # 2. LearnedConv (Transformer attention) per learned edge type
        #    Learned edges use attention to weight evidence quality
        #
        # Edge feature dimensions by edge type:
        #   - bridges: 10 dims (evidence quality: A/B/C, claim type, tier)
        #   - signaling: 21 dims (effect direction, logic type, param target, etc.)
        #   - modulates: 13 dims (P1: regulatory kinetics EC50/IC50/Kd/Ki)
        #   - transports_to: 14 dims (P1: compartment transport, BBB, direction)
        #   - regulates: None (no edge features yet)
        #
        # NOTE: affects edges removed — SNP personalization is handled outside GNN.
        # See personalization-architecture.md for the wild type GNN + SNP lookup design.
        #
        # Edge features bias attention via Graphormer-style encoding:
        #   score = (Q·K^T)/√d_k + W_e · edge_feat
        #
        # Reference: Ying et al. (2021) NeurIPS. "Graphormer"
        self.learned_convs = nn.ModuleDict()
        edge_feat_dims = {
            "modulates": cfg.modulates_edge_feat_dim,      # P1: 13-dim regulatory kinetics
            "regulates": None,                             # No edge features yet
            "signaling": cfg.signaling_edge_feat_dim,      # P0: 21-dim crosstalk features
            "bridges": cfg.bridge_edge_feat_dim,           # 10-dim evidence quality
            "transports_to": cfg.transport_edge_feat_dim,  # P1: 14-dim compartment transport
        }
        for _, etype, _ in LEARNED_ETYPES:
            self.learned_convs[etype] = LearnedConv(
                in_dim=h,
                out_dim=h,
                num_heads=cfg.gat_heads,  # 4 heads is optimal for small graphs
                feat_drop=cfg.gat_feat_drop,  # 0.1 is conservative
                attn_drop=cfg.gat_attn_drop,  # 0.1 prevents attention collapse
                edge_feat_dim=edge_feat_dims.get(etype),
            )

        # Pre-LN: normalize BEFORE attention (Xiong 2020)
        self.attn_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(h) for ntype in _RAW_DIMS
        })

        # --- FFN sublayer ---
        # Standard Transformer FFN: Linear → GELU → Dropout → Linear → Dropout
        ffn_hidden = cfg.hidden_dim * 4  # standard 4x expansion
        self.ffns = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(h, ffn_hidden),
                nn.GELU(),  # GELU > ReLU for Transformers (Hendrycks & Gimpel, 2016)
                nn.Dropout(cfg.dropout),
                nn.Linear(ffn_hidden, h),
                nn.Dropout(cfg.dropout),
            )
            for ntype in _RAW_DIMS
        })

        # Pre-LN: normalize BEFORE FFN
        self.ffn_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(h) for ntype in _RAW_DIMS
        })

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        h_dict: dict[str, torch.Tensor],
        rel_graphs: dict[tuple[str, str, str], dgl.DGLHeteroGraph] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """Pre-LN Transformer layer: Attention + FFN with residuals.

        Returns
        -------
        h_new : dict[ntype → tensor] — updated node embeddings
        attn  : dict[etype → tensor | None] — attention weights for audit trace
        """
        # ===== Attention sublayer (Pre-LN) =====
        # h' = h + Dropout(Attention(LN(h)))

        # First normalize inputs (Pre-LN)
        # Only normalize node types that are present in h_dict with non-empty tensors
        h_normed: dict[str, torch.Tensor] = {}
        for nt, h in h_dict.items():
            if h.numel() > 0:
                h_normed[nt] = self.attn_norms[nt](h)
            else:
                h_normed[nt] = h  # Keep empty tensor as-is

        # Accumulate messages per destination node type
        agg: dict[str, list[torch.Tensor]] = {nt: [] for nt in _RAW_DIMS}
        attn: dict[str, torch.Tensor | None] = {}

        # Physics edges (KineticConv)
        for stype, etype, dtype in PHYSICS_ETYPES:
            if g.num_edges((stype, etype, dtype)) == 0:
                continue
            key = (stype, etype, dtype)
            sub_g = rel_graphs[key] if rel_graphs is not None else g[key]
            conv = self.physics_convs[etype]
            out = conv(sub_g, h_normed[stype], h_normed[dtype])
            agg[dtype].append(out)

        # Learned edges (Transformer attention)
        # Edge types with edge features:
        #   - bridges: 10-dim evidence quality (grade, claim type, tier)
        #   - signaling: 21-dim crosstalk semantics (direction, logic, target)
        #   - modulates: 13-dim regulatory kinetics (P1: EC50/IC50/Kd/Ki features)
        #   - transports_to: 14-dim compartment transport (P1: BBB, direction, permeability)
        #
        # NOTE: affects edges removed — SNP personalization is handled outside GNN.
        edge_feature_types = {"bridges", "signaling", "modulates", "transports_to"}
        for stype, etype, dtype in LEARNED_ETYPES:
            if g.num_edges((stype, etype, dtype)) == 0:
                attn[etype] = None
                continue
            key = (stype, etype, dtype)
            sub_g = rel_graphs[key] if rel_graphs is not None else g[key]
            conv = self.learned_convs[etype]

            # Extract edge features if available for this edge type
            # Edge features bias attention: score = Q·K/√d_k + W_e·feat
            edge_feat = None
            if etype in edge_feature_types and "feat" in sub_g.edata:
                edge_feat = sub_g.edata["feat"]

            out = conv(sub_g, h_normed[stype], h_normed[dtype], edge_feat=edge_feat)
            agg[dtype].append(out)
            attn[etype] = conv.last_attention_weights

        # Residual connection for attention
        # h' = h + Dropout(message)
        h_attn: dict[str, torch.Tensor] = {}
        for ntype in _RAW_DIMS:
            if ntype not in h_dict or h_dict[ntype].numel() == 0:
                # Skip node types with no nodes
                h_attn[ntype] = h_dict.get(ntype, torch.zeros(0))
                continue
            if agg[ntype]:
                # Avoid stack+mean allocation; sum in-place style and divide once.
                # This reduces per-layer temporary tensor pressure on CPU training.
                msg = agg[ntype][0]
                for extra in agg[ntype][1:]:
                    msg = msg + extra
                msg = msg / float(len(agg[ntype]))
            else:
                msg = torch.zeros_like(h_dict[ntype])
            h_attn[ntype] = h_dict[ntype] + self.dropout(msg)

        # ===== FFN sublayer (Pre-LN) =====
        # h" = h' + Dropout(FFN(LN(h')))

        h_new: dict[str, torch.Tensor] = {}
        for ntype in _RAW_DIMS:
            if ntype not in h_attn or h_attn[ntype].numel() == 0:
                # Skip node types with no nodes
                h_new[ntype] = h_attn.get(ntype, torch.zeros(0))
                continue
            normed = self.ffn_norms[ntype](h_attn[ntype])
            h_new[ntype] = h_attn[ntype] + self.ffns[ntype](normed)

        return h_new, attn
