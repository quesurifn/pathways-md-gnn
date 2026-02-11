"""
PathwayGNN — Single heterogeneous GNN for hidden state inference.

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT THIS MODEL DOES
# ═══════════════════════════════════════════════════════════════════════════════
#
# The PathwayGNN is a HIDDEN STATE INFERENCE ENGINE for metabolic pathways.
#
# INPUT:
#   - Heterogeneous graph with node types: enzyme, metabolite, reaction
#   - Node features from canonical seed data (Km, Vmax, concentrations, etc.)
#   - Edge features encoding evidence quality and biological semantics
#
# OUTPUT:
#   - Hidden states on LEARNED edges (not physics edges):
#       • modulates_hidden: allosteric effects [-1, +1]
#       • regulates_hidden: regulatory strength [0, 1]
#       • signaling_hidden: pathway multipliers [0.5, 2.0]
#       • bridges_hidden: cofactor saturations [0, 1]
#       • transports_to_hidden: compartment transport [0, 1]
#   - Confidence scores for each prediction [0, 1]
#   - Attention weights for audit trace
#
# NOTE: SNP personalization is handled OUTSIDE the GNN via lookup tables.
# The GNN learns wild type biochemistry; SNPs are post-GNN multipliers.
# See personalization-architecture.md for the full design rationale.
#
# DOWNSTREAM USE:
#   The hidden states are fed to the RUST FLUX ENGINE, which uses them
#   as constraints in a deterministic Michaelis-Menten simulation:
#
#       Km_eff = Km × (1 - modulates_hidden)
#       v = Vmax × [S] / (Km_eff + [S])
#       (SNP multipliers are applied separately via variant_kinetics.jsonl)
#
# WHY THIS DESIGN:
#   1. GNN infers what we DON'T know (hidden states)
#   2. Rust engine encodes what we DO know (biochemistry)
#   3. Together: physics-informed inference with interpretable outputs
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

import dgl

from .config import ModelConfig, EdgeSemantic
from .graph import PHYSICS_ETYPES, LEARNED_ETYPES, EDGE_SEMANTICS
from .layers import KineticConv, LearnedConv, EdgeHead, ConfidenceHead

# ═══════════════════════════════════════════════════════════════════════════
# Raw Feature Dimensions per Node Type (must match graph.py)
# ═══════════════════════════════════════════════════════════════════════════
#
# These dimensions encode node-specific features extracted from seed data:
#
#   enzyme: 4 features
#     [vmax, km, confidence, variant_modifier]
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
_RAW_DIMS = {"enzyme": 4, "metabolite": 2, "reaction": 2}


class PathwayGNN(nn.Module):
    """Single heterogeneous GNN for inverse metabolic inference.

    # Purpose

    The PathwayGNN infers hidden states (cofactor saturations, allosteric effects)
    from observable evidence (genetic variants, dietary patterns, lab markers).

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

        # -- 2. Message-passing layers (physics + learned) -------------------
        self.mp_layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.mp_layers.append(_HeteroLayer(cfg))

        # -- 3. Edge heads (one per learned edge type) -----------------------
        # Input: src_h + dst_h concatenated
        edge_input_dim = h * 2

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

        # NOTE: affects_head removed — SNP personalization is handled outside GNN.
        # See personalization-architecture.md for the wild type GNN + SNP lookup design.

        # -- 4. Confidence heads (one per edge type) -------------------------
        # Each learned edge type has its own confidence predictor.
        # Confidence is used by the Rust flux engine to blend predictions:
        #   effective = baseline + (predicted - baseline) × confidence
        self.confidence_heads = nn.ModuleDict({
            "modulates": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "regulates": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "signaling": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
            "bridges": ConfidenceHead(edge_input_dim, edge_head_dim, cfg.dropout),
        })

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

        Note: SNP personalization is handled outside the GNN via lookup tables.
        See personalization-architecture.md for the wild type GNN + SNP lookup design.

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
                h_dict[ntype] = torch.relu(proj(g.nodes[ntype].data["feat"]))
            else:
                # Create empty tensor for missing node types
                # This allows forward pass even when some node types have no nodes
                h_dict[ntype] = torch.zeros(0, self.cfg.hidden_dim,
                                           device=next(proj.parameters()).device)

        # -- 2. Message passing (physics + learned) ---------------------------
        # Track attention weights for all learned edge types (for audit trace)
        all_attn: dict[str, list[torch.Tensor]] = {
            "modulates": [], "regulates": [], "signaling": [], "bridges": [],
        }
        for layer in self.mp_layers:
            h_dict, layer_attn = layer(g, h_dict)
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

        # NOTE: affects edges removed — SNP personalization is handled outside GNN.
        # See personalization-architecture.md for the wild type GNN + SNP lookup design.

        return outputs

    def _predict_edge_hidden(
        self,
        g: dgl.DGLHeteroGraph,
        stype: str,
        etype: str,
        dtype: str,
        h_dict: dict[str, torch.Tensor],
        head: EdgeHead,
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
            return {
                f"{name}_hidden": empty,
                f"{name}_conf": empty,
                f"{name}_semantic": empty,
            }

        src, dst = g.edges(etype=canonical)
        src_h = h_dict[stype][src]
        dst_h = h_dict[dtype][dst]

        hidden = head(src_h, dst_h)
        conf = self.confidence_heads[name](src_h, dst_h)

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

        return {
            f"{name}_hidden": hidden,
            f"{name}_conf": conf,
            f"{name}_semantic": semantic,
        }


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
            sub_g = g[(stype, etype, dtype)]
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
            sub_g = g[(stype, etype, dtype)]
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
                msg = torch.stack(agg[ntype]).mean(dim=0)
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
