"""
Configuration dataclasses for the GNN Calibrator.

All magic numbers live here. Nothing is hard-coded in model code.

# Architecture Overview

Single heterogeneous GNN with hybrid message passing:
- **Physics edges**: KineticConv (Michaelis-Menten form locked)
- **Learned edges**: Scaled dot-product attention (Transformer-style)
- **Output**: Hidden state scalars on learned edges + confidence

# Hyperparameter Justification

This file documents every hyperparameter choice with scientific rationale.

## Model Architecture

| Parameter      | Value | Justification                                      |
|----------------|-------|---------------------------------------------------|
| hidden_dim     | 64    | Sufficient for ~1000 node graphs (d ≈ √N)         |
| num_layers     | 4     | Diameter coverage, avoiding oversmoothing         |
| gat_heads      | 4     | Standard for small graphs (Veličković 2018)       |
| dropout        | 0.1   | Conservative, prevents memorization               |

## Training

| Parameter      | Value | Justification                                      |
|----------------|-------|---------------------------------------------------|
| learning_rate  | 3e-4  | Adam default, robust across domains               |
| weight_decay   | 1e-5  | Light L2 regularization                           |
| batch_size     | 64    | GPU memory efficient, stable gradients            |
| num_epochs     | 200   | Early stopping will terminate earlier             |

## Loss Weights

| Parameter         | Value | Role                                           |
|-------------------|-------|------------------------------------------------|
| lambda_physics    | 10.0  | Strong physics enforcement                     |
| lambda_smooth     | 1.0   | Baseline Occam's razor                         |
| lambda_bridge     | 5.0   | Medium pathway coherence                       |
| lambda_confidence | 0.1   | Auxiliary calibration task                     |

# References

- Vaswani et al. (2017) NeurIPS. "Attention Is All You Need"
- Veličković et al. (2018) ICLR. "Graph Attention Networks"
- Kingma & Ba (2015) ICLR. "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019) ICLR. "Decoupled Weight Decay Regularization"
- Li et al. (2018) AAAI. "Deeper Insights into GCNs" (oversmoothing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Edge Semantics — Gated Propagation Control
# ═══════════════════════════════════════════════════════════════════════════
#
# Edge semantics control how regulatory/signaling edges affect metabolic
# simulation. See gnn-calibrator.md "Edge Semantics: Gated Propagation".
#
# - METABOLIC_ANCHOR: Edges that can affect simulation (Vmax, Km)
#   Examples: CREB→TPH2, NRF2→SOD, HIF1A→LDHA
#   simulation_affecting = True
#
# - LATENT_CONTEXT: Hidden state only, no direct metabolic propagation
#   Examples: ARNTL→PER1, CLOCK→CRY1 (circadian loops)
#   simulation_affecting = False
#
# - EXCLUDED: Not in GNN, provenance/reference only
#   Examples: variant_outcomes (explainability data)
#
# Enforced rule: Only METABOLIC_ANCHOR edges can have simulation_affecting=True


class EdgeSemantic(Enum):
    """Semantic classification for regulatory/signaling edges.

    Controls gated propagation from signaling/TF layers into metabolic
    enzyme predictions. See gnn-calibrator.md for full documentation.
    """

    METABOLIC_ANCHOR = "metabolic_anchor"
    """TF/signaling edges that directly affect pathway-relevant enzymes.

    Inclusion criteria:
    1. Edge must directly affect a pathway-relevant enzyme or transporter
    2. Evidence must be human-backed (PMID-sourced, not inferred)
    3. Timescale must be compatible with metabolic simulation

    Examples: CREB→TPH2, NRF2→SOD, HIF1A→LDHA, SREBP→lipogenic enzymes
    """

    LATENT_CONTEXT = "latent_context"
    """Context edges for multi-hop reasoning; hidden state only.

    These edges update node embeddings during message passing but are
    gated out before enzyme Vmax/Km predictions. Useful for capturing
    biological context without injecting noise into metabolic outputs.

    Examples: ARNTL→PER1, CLOCK→CRY1 (circadian gene loops)
    """

    EXCLUDED = "excluded"
    """Reference/provenance data only; not loaded into GNN.

    Examples: variant_outcomes.jsonl (for explainability, not training)
    """


# ═══════════════════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the single heterogeneous GNN.

    # Architecture

    1. Per-type linear projection → shared hidden_dim
    2. N Pre-LN Transformer layers:
       - KineticConv on physics edges (Michaelis-Menten form locked)
       - LearnedConv on learned edges (scaled dot-product attention)
       - FFN sublayer (4x expansion, GELU)
    3. Edge heads: per learned-edge-type MLP → hidden state scalars
    4. Attention weights retained for audit trace

    # Design Rationale

    For small metabolic graphs (~100-1000 nodes), this configuration
    balances expressiveness with data efficiency:
    - **64-dim embeddings**: Matches sqrt(N) heuristic for graph size
    - **4 layers**: Covers graph diameter without oversmoothing
    - **4 attention heads**: Standard for interpretability

    # References

    - Veličković et al. (2018) ICLR. "Graph Attention Networks"
    - Xiong et al. (2020) ICML. "On Layer Normalization in the Transformer"
    - Li et al. (2018) AAAI. "Deeper Insights into GCNs" (oversmoothing)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Dimensions
    #
    # hidden_dim = 64 is justified by:
    # 1. sqrt(N) heuristic: For N ≈ 1000 nodes, sqrt(1000) ≈ 32. We use 64
    #    for 2x safety margin.
    # 2. Power of 2: GPU-efficient tensor operations (32, 64, 128...)
    # 3. Empirical: 64 is the GNN community standard for small graphs
    #    (Kipf & Welling 2017, Veličković 2018)
    #
    # Reference: Hamilton (2020) "Graph Representation Learning" Ch. 5
    # ─────────────────────────────────────────────────────────────────────────
    node_dim: int = 64              # initial node embedding size
    hidden_dim: int = 64            # message-passing hidden size

    # ─────────────────────────────────────────────────────────────────────────
    # Layer Depth
    #
    # num_layers = 4 is chosen to balance:
    # 1. Receptive field: 4 hops covers typical pathway diameter
    # 2. Oversmoothing: >6 layers causes node features to converge
    #    (Li et al. 2018 "Deeper Insights into GCNs")
    # 3. Compute: Linear scaling, 4 is fast enough for CPU inference
    #
    # The oversmoothing phenomenon:
    #   - After k layers, node features mix with k-hop neighbors
    #   - Too many layers → all nodes converge to same embedding
    #   - Mitigation: residual connections, dropout, normalization
    #
    # Reference: Li et al. (2018) AAAI. "Deeper Insights into GCNs"
    # ─────────────────────────────────────────────────────────────────────────
    num_layers: int = 4             # number of message-passing layers

    # ─────────────────────────────────────────────────────────────────────────
    # Dropout
    #
    # dropout = 0.1 is conservative regularization:
    # 1. Prevents memorization on small datasets
    # 2. Lower than NLP standard (0.3) because graphs have more inductive bias
    # 3. Applied after: attention, FFN, input projection
    #
    # Reference: Srivastava et al. (2014) JMLR. "Dropout"
    # ─────────────────────────────────────────────────────────────────────────
    dropout: float = 0.1

    # ─────────────────────────────────────────────────────────────────────────
    # Attention Heads
    #
    # gat_heads = 4 is the standard for small graphs:
    # 1. Original GAT used 8 heads for Cora (2708 nodes)
    # 2. For smaller graphs, 4 heads is sufficient
    # 3. Each head: d_k = hidden_dim / num_heads = 64/4 = 16
    #
    # Multi-head attention allows learning diverse attention patterns:
    #   - Head 1: may focus on substrate relationships
    #   - Head 2: may focus on regulatory relationships
    #   - etc.
    #
    # Reference: Veličković et al. (2018) ICLR. "Graph Attention Networks"
    # ─────────────────────────────────────────────────────────────────────────
    gat_heads: int = 4              # multi-head attention heads
    gat_feat_drop: float = 0.1      # feature dropout (before attention)
    gat_attn_drop: float = 0.1      # attention dropout (prevents attention collapse)

    # ─────────────────────────────────────────────────────────────────────────
    # Bridge Edge Features (Evidence Quality)
    #
    # 10-dimensional feature vector encoding evidence quality:
    # [grade_A, grade_B, grade_C,        # one-hot evidence grade
    #  num_quality_norm,                 # normalized quality score
    #  claim_mech, claim_obs, claim_gen, # claim type one-hot
    #  tier_func, tier_gen, tier_obs]    # tier one-hot
    #
    # These features bias attention via Graphormer-style edge encoding:
    #   score = (Q·K^T)/√d_k + edge_bias
    #
    # Reference: Ying et al. (2021) NeurIPS. "Graphormer"
    # ─────────────────────────────────────────────────────────────────────────
    bridge_edge_feat_dim: int = 10

    # ─────────────────────────────────────────────────────────────────────────
    # Signaling Edge Features (P0 Integration)
    #
    # 21-dimensional feature vector encoding signaling crosstalk semantics:
    #   effect_direction: one-hot [increased, decreased, mixed, unknown] = 4
    #   logic_type: one-hot [activate, inhibit, feedback, feedforward, gate, other] = 6
    #   parameter_target: one-hot [Vmax, enzyme_activity, Km, other, unknown] = 5
    #   multiplier_value: continuous normalized = 1
    #   simulation_affecting: binary = 1
    #   source_strength: one-hot [A, B, C, D] = 4
    #   Total: 4 + 6 + 5 + 1 + 1 + 4 = 21
    #
    # These features allow the GNN to learn HOW signaling affects downstream
    # pathways, not just THAT signaling exists. Critical for crosstalk modeling.
    #
    # Reference: graph.py _signaling_edges() for feature extraction details.
    # ─────────────────────────────────────────────────────────────────────────
    signaling_edge_feat_dim: int = 21

    # ─────────────────────────────────────────────────────────────────────────
    # NOTE: SNP personalization is handled OUTSIDE the GNN via lookup tables.
    # The GNN learns wild type biochemistry; SNPs are post-GNN multipliers.
    # See personalization-architecture.md for the full design rationale.
    #
    # - variant_kinetics.jsonl: SNP→enzyme multipliers (used in Rust engine)
    # - edges_affects.jsonl: SNP→enzyme links (used for explainability only)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # P1: Modulates Edge Features (Quantitative Regulatory Kinetics)
    #
    # 13-dimensional feature vector encoding modulator potency from
    # regulatory_kinetics.jsonl (265 records).
    #
    # Features:
    #   normalized_measure: one-hot [EC50, IC50, Kd, Ki] = 4
    #     - EC50: Half-maximal effective concentration (functional assay)
    #     - IC50: Half-maximal inhibitory concentration
    #     - Kd: Dissociation constant (binding affinity)
    #     - Ki: Inhibition constant (competitive inhibition)
    #
    #   action: one-hot [ACTIVATE, BIND, MODULATE, INHIBIT] = 4
    #     - ACTIVATE: Agonist increasing activity
    #     - BIND: Binding without clear functional effect
    #     - MODULATE: Allosteric modulation
    #     - INHIBIT: Antagonist/inhibitor
    #
    #   normalized_value: continuous pEC50/pKd scale = 1
    #     - Converted to log scale: pEC50 = -log10(EC50_M)
    #     - Higher values = more potent (tighter binding)
    #     - Range ~3-12 for biological ligands
    #     - Normalized to [0, 1] by: (pValue - 3) / 9
    #
    #   source_strength: one-hot [A, B, C, D] = 4
    #     - A: RCT/meta-analysis (rare for kinetics)
    #     - B: Prospective cohort
    #     - C: Human in vitro
    #     - D: Animal/cell line
    #
    # Total: 4 + 4 + 1 + 4 = 13 dimensions
    #
    # Scientific rationale:
    # - Lower EC50 = more potent modulator = stronger effect on Km
    # - The GNN uses these features to weight attention:
    #   score = (Q·K^T)/√d_k + W_e · edge_feat
    # - This lets high-affinity modulators dominate message passing
    #
    # References:
    # - Cheng & Prusoff (1973) BBRC "Relationship between inhibition constant"
    # - Swinney (2011) Nat Rev Drug Discov "Biochemical mechanisms of drug action"
    # ─────────────────────────────────────────────────────────────────────────
    modulates_edge_feat_dim: int = 13

    # ─────────────────────────────────────────────────────────────────────────
    # P1: Transport Edge Features (Compartmentalization)
    #
    # Transport edges link metabolites across compartments (systemic↔brain,
    # cytosol↔vesicle). The GNN learns transport multipliers ∈ [0, 2] that
    # modulate flux across compartment boundaries.
    #
    # Features (14 dimensions):
    #   - direction: one-hot [systemic_to_brain, brain_to_systemic,
    #                 cytosol_to_vesicle, vesicle_to_cytosol, bidirectional] = 5
    #   - is_bbb: binary (1 if BBB crossing, 0 if intracellular) = 1
    #   - logbb: continuous (normalized log brain:blood ratio) = 1
    #   - permeability: one-hot [BBB+, BBB-] = 2
    #   - has_km: binary (1 if magnitude/Km available) = 1
    #   - source_strength: one-hot [A, B, C, D] = 4
    #
    # Total: 5 + 1 + 1 + 2 + 1 + 4 = 14
    # ─────────────────────────────────────────────────────────────────────────
    transport_edge_feat_dim: int = 14

    # ─────────────────────────────────────────────────────────────────────────
    # Edge Head Hidden Dimension
    #
    # edge_head_hidden_dim = 32 is half of hidden_dim:
    # 1. Edge heads are simple MLPs (input → hidden → output)
    # 2. Input: concat(src_h, dst_h) = 128 dims
    # 3. Hidden: 32 dims (compression bottleneck)
    # 4. Output: 1 scalar (hidden state)
    #
    # The bottleneck forces learning compact edge representations.
    # ─────────────────────────────────────────────────────────────────────────
    edge_head_hidden_dim: int = 32  # MLP hidden dim in edge heads

    # ─────────────────────────────────────────────────────────────────────────
    # Output Ranges per Learned Edge Type
    #
    # Each edge type predicts a scalar in a biologically meaningful range:
    #
    # modulates: [-1, 1] — Allosteric effect on enzyme Km
    #   -1 = full inhibition (Km → ∞)
    #    0 = no effect
    #   +1 = full activation (Km → 0)
    #
    # regulates: [0, 1] — Regulatory strength (enzyme → enzyme)
    #    0 = no regulation
    #    1 = full regulation
    #
    # signaling: [0, 2] — Signal multiplier (metabolite → metabolite)
    #    0 = signal blocked
    #    1 = signal preserved
    #    2 = signal amplified
    #
    # bridges: [0, 1] — Cofactor saturation level
    #    0 = fully depleted (enzyme starved)
    #    1 = fully saturated (enzyme at Vmax)
    #
    # These ranges are enforced by tanh/sigmoid activations in EdgeHead.
    # ─────────────────────────────────────────────────────────────────────────
    modulates_range: tuple[float, float] = (-1.0, 1.0)    # effect strength
    regulates_range: tuple[float, float] = (0.0, 1.0)     # regulatory strength
    signaling_range: tuple[float, float] = (0.0, 2.0)     # signal multiplier
    bridge_range: tuple[float, float] = (0.0, 1.0)        # cofactor saturation

    # NOTE: affects_range removed — SNP personalization is handled outside GNN.
    # See personalization-architecture.md for the wild type GNN + SNP lookup design.

    # ─────────────────────────────────────────────────────────────────────────
    # Training Hyperparameters
    #
    # learning_rate = 3e-4:
    #   The "magic number" for Adam optimizer. Robust across domains.
    #   Reference: Kingma & Ba (2015) ICLR. "Adam"
    #
    # weight_decay = 1e-5:
    #   Light L2 regularization. We use AdamW (decoupled weight decay).
    #   Reference: Loshchilov & Hutter (2019) ICLR. "Decoupled Weight Decay"
    #
    # batch_size = 64:
    #   Good balance of GPU utilization and gradient noise.
    #   Larger batches → faster but worse generalization (Keskar 2017).
    #   Reference: Keskar et al. (2017) ICLR. "On Large-Batch Training"
    #
    # num_epochs = 200:
    #   Upper bound. Early stopping will terminate when val loss plateaus.
    # ─────────────────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 200

    # ─────────────────────────────────────────────────────────────────────────
    # Loss Weights (Lagrange Multipliers)
    #
    # These weights balance the multi-objective loss:
    #   L = L_pred + λ_physics·L_physics + λ_smooth·L_smooth + ...
    #
    # lambda_physics = 10.0:
    #   Strong enforcement of mass-balance constraints.
    #   Physics should dominate when predictions violate conservation laws.
    #
    # lambda_smooth = 1.0:
    #   Baseline weight for Occam's razor (prefer small hidden states).
    #   Equal to L_pred weight, so both contribute equally by default.
    #
    # lambda_bridge = 5.0:
    #   Medium weight for pathway coherence.
    #   Bridges between same pathway pair should agree.
    #
    # lambda_confidence = 0.1:
    #   Low weight for confidence calibration (auxiliary task).
    #   Calibration is important but shouldn't dominate main task.
    #
    # Reference: Multi-task learning (Caruana 1997, Ruder 2017)
    # ─────────────────────────────────────────────────────────────────────────
    lambda_physics: float = 10.0    # mass-balance consistency
    lambda_smooth: float = 1.0      # prefer small hidden states
    lambda_bridge: float = 5.0      # bridge coherence
    lambda_confidence: float = 0.1  # confidence calibration


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
#
# Canonical Seed Data Coverage
# ============================
#
# The seed data in data-experiments/data-experiments/seed/canonical/ contains
# curated biochemical knowledge from PubMed, KEGG, Reactome, BRENDA, etc.
#
# CURRENTLY USED:
# ---------------
# Core:     enzymes, metabolites, reactions, pathways, pathway_components
# Edges:    catalyzes_reaction, reaction_participants, cofactor, modulates,
#           regulates, signaling_crosstalk, bridges, transports_to
# Kinetics: kinetics.jsonl, regulatory_kinetics.jsonl, inhibitors.jsonl
#
# SNP PERSONALIZATION (NOT IN GNN - see personalization-architecture.md):
# -----------------------------------------------------------------------
#   - edges_affects.jsonl: Used for explainability only (links SNPs to enzymes)
#   - variant_kinetics.jsonl: SNP→enzyme multipliers (used in Rust engine)
#
# NOT YET INTEGRATED:
# -------------------
# P2 MEDIUM:
#   - regulatory_dynamics.jsonl (454): Time constants (tau, KOFF)
#   - receptor_expression_context.jsonl (1,478): Tissue-specific expression
#   - substances.jsonl (1,104): Signaling molecule metadata
#   - variant_outcomes.jsonl (610): Clinical phenotypes
#
# P3 LOW:
#   - edges_structure.jsonl (145): Protein complexes
#   - substance_effects.jsonl (25,137): Literature claims
#   - seed_guardrails.jsonl: Data quality validation rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Paths to seed JSONL files.

    # Data Sources

    The canonical seed directory contains curated biochemical data organized as:

    ```
    canonical/
    ├── core/           # Entity definitions (nodes)
    ├── edges/          # Relationship definitions (edges)
    ├── kinetics/       # Kinetic parameters (Vmax, Km, EC50)
    └── governance/     # Validation rules
    ```

    # Currently Loaded

    - **Core entities**: enzymes, metabolites, reactions (→ node features)
    - **Physics edges**: catalyzes, participants, cofactor (→ KineticConv)
    - **Learned edges**: modulates, regulates, signaling, bridges (→ LearnedConv)

    # Not Yet Loaded (TODO)

    See the P0/P1/P2/P3 priority list in the module-level comment above.
    """

    seed_root: Path = Path(
        "data-experiments/data-experiments/seed/canonical"
    )

    # =========================================================================
    # CORE ENTITIES (nodes)
    # =========================================================================

    @property
    def enzymes_path(self) -> Path:
        """Enzyme node definitions with baseline kinetics."""
        return self.seed_root / "core" / "enzymes.jsonl"

    @property
    def metabolites_path(self) -> Path:
        """Metabolite node definitions with baseline concentrations."""
        return self.seed_root / "core" / "metabolites.jsonl"

    @property
    def reactions_path(self) -> Path:
        """Reaction node definitions with reversibility flags."""
        return self.seed_root / "core" / "reactions.jsonl"

    @property
    def pathways_path(self) -> Path:
        """Pathway definitions for grouping components."""
        return self.seed_root / "core" / "pathways.jsonl"

    @property
    def pathway_components_path(self) -> Path:
        """Symbol → pathway mapping for component lookup."""
        return self.seed_root / "core" / "pathway_components.jsonl"

    # =========================================================================
    # KINETICS (Rust flux engine uses these directly)
    # =========================================================================

    @property
    def kinetics_path(self) -> Path:
        """Vmax/Km parameters for enzymatic reactions."""
        return self.seed_root / "kinetics" / "kinetics.jsonl"

    @property
    def variant_kinetics_path(self) -> Path:
        """SNP-specific kinetic modifiers (Vmax/Km multipliers)."""
        return self.seed_root / "kinetics" / "variant_kinetics.jsonl"

    # =========================================================================
    # PHYSICS-INFORMED EDGES (use KineticConv — encode biochemistry)
    # =========================================================================

    @property
    def edges_catalyzes_path(self) -> Path:
        """enzyme → reaction: Which enzyme catalyzes which reaction."""
        return self.seed_root / "edges" / "edges_catalyzes_reaction.jsonl"

    @property
    def edges_reaction_participants_path(self) -> Path:
        """metabolite ↔ reaction: Substrates and products."""
        return self.seed_root / "edges" / "edges_reaction_participants.jsonl"

    @property
    def edges_cofactor_path(self) -> Path:
        """metabolite → enzyme: Cofactor requirements (SAMe, NAD+, etc.)."""
        return self.seed_root / "edges" / "edges_cofactor.jsonl"

    # =========================================================================
    # LEARNED EDGES (use LearnedConv — GNN infers hidden states)
    # =========================================================================

    @property
    def edges_modulates_path(self) -> Path:
        """metabolite → enzyme: Allosteric modulation (Km effects)."""
        return self.seed_root / "edges" / "edges_modulates.jsonl"

    @property
    def edges_regulates_path(self) -> Path:
        """enzyme → enzyme: Transcriptional regulation."""
        return self.seed_root / "edges" / "edges_regulates.jsonl"

    @property
    def edges_signaling_path(self) -> Path:
        """metabolite → metabolite: Cross-pathway signaling (crosstalk)."""
        return self.seed_root / "edges" / "edges_signaling_crosstalk.jsonl"

    @property
    def bridges_path(self) -> Path:
        """metabolite → enzyme: Bridge saturation (cofactor availability)."""
        return self.seed_root / "edges" / "bridges.jsonl"

    # =========================================================================
    # SNP PERSONALIZATION — NOT IN GNN (explainability only)
    #
    # SNP personalization is handled OUTSIDE the GNN via lookup tables.
    # The GNN learns wild type biochemistry; SNPs are post-GNN multipliers.
    # See personalization-architecture.md for the full design rationale.
    # =========================================================================

    @property
    def edges_affects_path(self) -> Path:
        """SNP → enzyme: Genetic variant effects on enzyme function.

        ⚠️ NOT LOADED INTO GNN GRAPH — used for explainability only.

        Contains 2,490 records linking SNPs to enzymes for user-facing
        explanations (e.g., "Your rs4680 affects COMT activity").

        Fields:
        - snp_id, enzyme_id: Which SNP affects which enzyme
        - effect: risk_increase, baseline, unknown
        - mechanism: promoter_activity, coding_variant, etc.

        See personalization-architecture.md for why SNPs are not in the GNN.
        """
        return self.seed_root / "edges" / "edges_affects.jsonl"

    # =========================================================================
    # INTEGRATED — P1 Transport/Compartmentalization
    # =========================================================================

    @property
    def edges_transport_path(self) -> Path:
        """Compartment transport (systemic ↔ brain, etc.).

        ✅ INTEGRATED: Creates transports_to edges with 14-dim features.

        Contains 372 records with:
        - transporter_id, metabolite_id
        - from_compartment, to_compartment
        - direction: systemic_to_brain, etc.
        """
        return self.seed_root / "edges" / "edges_transport.jsonl"

    @property
    def edges_bbb_path(self) -> Path:
        """Blood-brain barrier permeability.

        Contains 103 records with:
        - systemic_metabolite_id → brain_metabolite_id
        - logbb: Blood-brain partition coefficient
        - permeability: BBB- (blocked), BBB+ (crosses)

        TODO: Use for brain compartment modeling.
        """
        return self.seed_root / "edges" / "edges_bbb.jsonl"

    @property
    def regulatory_kinetics_path(self) -> Path:
        """EC50/IC50 values for modulators.

        Contains 265 records with:
        - substance_id → target_id
        - normalized_measure: EC50
        - normalized_value: e.g., 9.33 nM
        - mechanism: agonist/antagonist

        TODO: Use for quantitative modulation in GNN.
        """
        return self.seed_root / "kinetics" / "regulatory_kinetics.jsonl"

    @property
    def inhibitors_path(self) -> Path:
        """Competitive and allosteric inhibitors.

        P1 COMPLETE: Integrated into modulates edges.

        Contains 110,317 records with:
        - enzyme_id: Target enzyme being inhibited
        - inhibitor_metabolite_id: Metabolite that inhibits (5,231 records mapped)
        - ki: Inhibition constant (31,779 records with values)
        - source_strength: Evidence quality (A/B/C/D)

        Records with inhibitor_metabolite_id are added as modulates edges
        with Ki-based features (normalized_measure=Ki, action=INHIBIT).
        """
        return self.seed_root / "kinetics" / "inhibitors.jsonl"

    # =========================================================================
    # NOT YET INTEGRATED — P2 MEDIUM PRIORITY
    # =========================================================================

    @property
    def regulatory_dynamics_path(self) -> Path:
        """Time constants for receptor dynamics.

        Contains 454 records with:
        - parameter: resensitization_tau, KOFF, etc.
        - value, units: Numeric time constants
        - substance_id → target_id

        TODO: Use for time-dependent simulation.
        """
        return self.seed_root / "kinetics" / "regulatory_dynamics.jsonl"

    @property
    def receptor_expression_path(self) -> Path:
        """Tissue-specific receptor expression.

        Contains 1,478 records with:
        - receptor_id: e.g., NR3C1, HTR3A
        - tissue, cell_type: brain, liver, etc.
        - expression_measure: RNA_nTPM
        - value: Expression level

        TODO: Use for tissue-specific effect scaling.
        """
        return self.seed_root / "core" / "receptor_expression_context.jsonl"

    @property
    def substances_path(self) -> Path:
        """Signaling molecule metadata (ChEBI IDs, types).

        Contains 1,104 records with substance metadata.

        TODO: Use for enhanced node features.
        """
        return self.seed_root / "core" / "substances.jsonl"

    @property
    def variant_outcomes_path(self) -> Path:
        """Clinical phenotypes associated with variants.

        Contains 610 records with:
        - rsid, enzyme_id
        - phenotype: Clinical outcome description
        - p_value, metric

        TODO: Use for phenotype prediction supervision.
        """
        return self.seed_root / "kinetics" / "variant_outcomes.jsonl"

