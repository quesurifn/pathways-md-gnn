"""
Graph construction — load seed JSONL into a DGL heterograph.

# Overview

This module transforms curated biochemical seed data (JSONL files) into a
heterogeneous graph for GNN-based metabolic inference. The graph encodes:

- **Nodes**: enzymes, metabolites, reactions
- **Physics edges**: Known biochemical relationships (catalysis, substrates, etc.)
- **Learned edges**: Regulatory/signaling relationships where GNN infers hidden states

# Node Types

| Type | Source File | Features |
|------|-------------|----------|
| enzyme | enzymes.jsonl | baseline_km, baseline_vmax, kcat, source_grade |
| metabolite | metabolites.jsonl | baseline_vol, source_grade |
| reaction | reactions.jsonl | reversible_flag, source_grade |

# Edge Types

## Physics-Informed Edges (KineticConv)

These encode known biochemical relationships. The GNN uses physics-informed
message passing that respects Michaelis-Menten form.

| Edge Type | Relation | Source File |
|-----------|----------|-------------|
| catalyzes | enzyme → reaction | edges_catalyzes_reaction.jsonl |
| substrate_of | metabolite → reaction | edges_reaction_participants.jsonl |
| produces | reaction → metabolite | edges_reaction_participants.jsonl |
| cofactor_for | metabolite → enzyme | edges_cofactor.jsonl |

## Learned Edges (LearnedConv with Attention)

These represent regulatory/signaling relationships where the GNN learns
hidden states (saturations, effects, multipliers) via attention.

| Edge Type | Relation | Hidden State | Source File |
|-----------|----------|--------------|-------------|
| modulates | metabolite → enzyme | Km modifier ∈ [-1, 1] | edges_modulates.jsonl |
| regulates | enzyme → enzyme | Expression effect ∈ [0, 2] | edges_regulates.jsonl |
| signaling | metabolite → metabolite | Signal multiplier ∈ [0, 2] | edges_signaling_crosstalk.jsonl |
| bridges | metabolite → enzyme | Cofactor saturation ∈ [0, 1] | bridges.jsonl |

# Canonical Seed Data Audit

## ✅ CURRENTLY USED

| File | Records | Purpose |
|------|---------|---------|
| core/enzymes.jsonl | ~200 | Enzyme nodes + features |
| core/metabolites.jsonl | ~300 | Metabolite nodes + features |
| core/reactions.jsonl | ~400 | Reaction nodes + features |
| core/pathways.jsonl | ~50 | Pathway metadata (informational) |
| core/pathway_components.jsonl | ~500 | Symbol→pathway mapping |
| edges/bridges.jsonl | ~100 | Bridge edges (metabolite→enzyme) |
| edges/edges_catalyzes_reaction.jsonl | ~400 | Physics edges |
| edges/edges_reaction_participants.jsonl | ~800 | Physics edges |
| edges/edges_cofactor.jsonl | ~200 | Physics edges |
| edges/edges_modulates.jsonl | ~300 | Learned edges |
| edges/edges_regulates.jsonl | ~50 | Learned edges |
| edges/edges_signaling_crosstalk.jsonl | 150 | Learned edges |
| kinetics/kinetics.jsonl | ~400 | Vmax/Km parameters (Rust) |
| kinetics/variant_kinetics.jsonl | ~200 | SNP kinetic effects (Rust personalization layer) |

## ✅ P0 COMPLETE — Signaling Edge Features (IMPLEMENTED)

| File | Records | Implementation | Status |
|------|---------|----------------|--------|
| edges_signaling_crosstalk | 150 | 21-dim edge features (effect_direction, logic_type, parameter_target, etc.) | ✅ DONE |

## ⚠️ SNP Personalization — NOT IN GNN (by design)

SNP personalization is handled OUTSIDE the GNN via lookup tables in the Rust flux engine.
The GNN learns wild type biochemistry; SNPs are post-GNN multipliers.
See personalization-architecture.md for the full design rationale.

| File | Records | Usage | NOT in GNN |
|------|---------|-------|------------|
| edges/edges_affects.jsonl | 2,490 | Explainability only (links SNPs to enzymes) | Lookup only |
| kinetics/variant_kinetics.jsonl | 195 | SNP→enzyme multipliers (Vmax, Km) | Rust engine |

## ✅ P1 COMPLETE — Regulatory Kinetics (IMPLEMENTED)

| File | Records | Implementation | Status |
|------|---------|----------------|--------|
| kinetics/regulatory_kinetics.jsonl | 265 | modulates edges now have 13-dim features (EC50/IC50/Kd/Ki, action, source_strength) | ✅ DONE |

## ✅ P1 COMPLETE — Transport/Compartmentalization (IMPLEMENTED)

| File | Records | Implementation | Status |
|------|---------|----------------|--------|
| edges/edges_transport.jsonl | 372 | transports_to edges with 14-dim features | ✅ DONE |
| edges/edges_bbb.jsonl | 103 | Merged into transports_to edges with BBB features | ✅ DONE |

## ✅ P1 COMPLETE — Competitive Inhibitors (IMPLEMENTED)

| File | Records | Implementation | Status |
|------|---------|----------------|--------|
| kinetics/inhibitors.jsonl | 110,317 | Merged into modulates edges (5,231 with metabolite mapping) | ✅ DONE |

The inhibitors.jsonl data is integrated into modulates edges. Records with
`inhibitor_metabolite_id` create new metabolite→enzyme edges with:
- normalized_measure: Ki (always for inhibitors)
- action: INHIBIT (always for inhibitors)
- pValue: Computed from Ki value (µM → nM → pKi scale)
- source_strength: From source record (typically D for BRENDA)

## ❌ NOT USED — REMAINING GAPS

### MEDIUM PRIORITY (P2) — Temporal & Tissue Dynamics

| File | Records | What's Missing | Impact |
|------|---------|----------------|--------|
| kinetics/regulatory_dynamics.jsonl | 454 | Time constants (tau, KOFF) | Temporal simulation |
| core/receptor_expression_context.jsonl | 1,478 | Tissue-specific expression | Tissue-specific effects |
| core/substances.jsonl | 1,104 | Signaling molecule metadata | Enhanced node features |
| kinetics/variant_outcomes.jsonl | 610 | Clinical phenotypes | Phenotype prediction |

### LOW PRIORITY (P3) — Enhancements

| File | Records | What's Missing | Impact |
|------|---------|----------------|--------|
| edges/edges_structure.jsonl | 145 | Protein complexes | Structural interactions |
| edges/substance_effects.jsonl | 25,137 | Literature claims | Weak supervision |
| governance/seed_guardrails.jsonl | 2 | Validation rules | Data quality gates |

# Scientific References

## Graph Neural Networks for Biochemistry

- Ying et al. (2021) NeurIPS. "Do Transformers Really Perform Bad for Graph
  Representation?" — Graphormer, edge features as attention bias
- Rampášek et al. (2022) NeurIPS. "Recipe for a General, Powerful, Scalable
  Graph Transformer" — GPS, hybrid local/global message passing
- Stokes et al. (2020) Cell 180:688-702. "A Deep Learning Approach to
  Antibiotic Discovery" — GNN for molecular property prediction

## Heterogeneous Graphs

- Schlichtkrull et al. (2018) ESWC. "Modeling Relational Data with Graph
  Convolutional Networks" — R-GCN for typed edges
- Wang et al. (2019) WWW. "Heterogeneous Graph Attention Network" — HAN
  for multi-relational graphs

## Metabolic Network Modeling

- Palsson (2015) "Systems Biology: Constraint-based Reconstruction and
  Analysis" — Constraint-based metabolic modeling
- Thiele & Palsson (2010) Nat Protoc 5:93-121. "A protocol for generating
  a high-quality genome-scale metabolic reconstruction"
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dgl
import torch

from .config import DataConfig, EdgeSemantic

logger = logging.getLogger(__name__)

# ---- Edge type constants (used by layers.py / model.py) --------------------

PHYSICS_ETYPES = [
    ("enzyme", "catalyzes", "reaction"),
    ("metabolite", "substrate_of", "reaction"),
    ("reaction", "produces", "metabolite"),
    ("metabolite", "cofactor_for", "enzyme"),
]

LEARNED_ETYPES = [
    ("metabolite", "modulates", "enzyme"),
    ("enzyme", "regulates", "enzyme"),
    ("metabolite", "signaling", "metabolite"),  # signaling_crosstalk
    ("metabolite", "bridges", "enzyme"),  # cofactor/substrate → enzyme bridge
    ("metabolite", "transports_to", "metabolite"),  # P1: Compartment transport
]

ALL_ETYPES = PHYSICS_ETYPES + LEARNED_ETYPES

# ---------------------------------------------------------------------------
# Edge Semantics — Gated Propagation Control
# ---------------------------------------------------------------------------
#
# Maps edge types to their semantic classification for gated propagation.
# See gnn-calibrator.md "Edge Semantics: Gated Propagation" for details.
#
# - METABOLIC_ANCHOR: Can affect simulation (Vmax, Km)
# - LATENT_CONTEXT: Hidden state only, no metabolic propagation
#
# Physics edges are always METABOLIC_ANCHOR (they define the core simulation).
# Learned edges are classified based on their biological relevance to
# metabolic enzyme activity.

EDGE_SEMANTICS: dict[str, EdgeSemantic] = {
    # Physics edges — always metabolic anchor (core simulation)
    "catalyzes": EdgeSemantic.METABOLIC_ANCHOR,
    "substrate_of": EdgeSemantic.METABOLIC_ANCHOR,
    "produces": EdgeSemantic.METABOLIC_ANCHOR,
    "cofactor_for": EdgeSemantic.METABOLIC_ANCHOR,
    # Learned edges — classified by metabolic relevance
    "modulates": EdgeSemantic.METABOLIC_ANCHOR,      # Direct enzyme modulation
    "signaling": EdgeSemantic.METABOLIC_ANCHOR,      # Crosstalk affects enzyme activity
    "bridges": EdgeSemantic.METABOLIC_ANCHOR,        # Cofactor saturation → enzyme
    "transports_to": EdgeSemantic.METABOLIC_ANCHOR,  # Compartment transport
    # Regulates: LATENT_CONTEXT by default (circadian TF→gene loops)
    # Per-edge overrides are in REGULATES_ANCHOR_EDGES below.
    "regulates": EdgeSemantic.LATENT_CONTEXT,
}

# ---------------------------------------------------------------------------
# Curated METABOLIC_ANCHOR regulates edges
# ---------------------------------------------------------------------------
#
# These TF→enzyme edges have direct metabolic relevance and should affect
# the simulation (Vmax/Km). All other regulates edges remain LATENT_CONTEXT
# (context for multi-hop reasoning, but gated out at output stage).
#
# Inclusion criteria (from gnn-calibrator.md):
#   1. Edge must directly affect a pathway-relevant enzyme or transporter
#   2. Evidence must be human-backed (PMID-sourced, not inferred)
#   3. Timescale must be compatible with metabolic simulation
#
# Format: frozenset of (tf_id, target_gene_id) tuples

REGULATES_ANCHOR_EDGES: frozenset[tuple[str, str]] = frozenset({
    # AHR (Aryl Hydrocarbon Receptor) → Xenobiotic/drug-metabolizing enzymes
    # AHR regulates Phase I/II metabolism enzymes - critical for drug response
    ("AHR", "CYP1A1"),   # Cytochrome P450 - drug/toxin metabolism
    ("AHR", "CYP1A2"),   # Cytochrome P450 - caffeine, theophylline metabolism
    ("AHR", "CYP1B1"),   # Cytochrome P450 - estrogen metabolism
    ("AHR", "CYP2B6"),   # Cytochrome P450 - drug metabolism
    ("AHR", "UGT1A1"),   # UDP-glucuronosyltransferase - bilirubin conjugation
    ("AHR", "AFMID"),    # Arylformamidase - kynurenine pathway
    ("AHR", "ALDH3A1"),  # Aldehyde dehydrogenase - aldehyde detoxification

    # NR3C1 (Glucocorticoid Receptor) → Metabolic enzymes
    ("NR3C1", "H6PD"),   # Hexose-6-phosphate dehydrogenase - cortisol metabolism

    # NR1H4 (FXR - Farnesoid X Receptor) → Bile acid metabolism enzymes
    ("NR1H4", "FABP6"),  # Fatty acid binding protein - bile acid transport
    ("NR1H4", "UGT2B4"), # UDP-glucuronosyltransferase - bile acid conjugation

    # CLOCK → Metabolically-relevant targets (not just circadian genes)
    ("CLOCK", "NAMPT"),  # Nicotinamide phosphoribosyltransferase - NAD+ biosynthesis
})

# ---------------------------------------------------------------------------
# Edge Feature Dimensions (for LearnedConv)
# ---------------------------------------------------------------------------
#
# Signaling edge features (21 dimensions):
#   - effect_direction: one-hot [increased, decreased, mixed, unknown] = 4
#   - logic_type: one-hot [activate, inhibit, feedback, feedforward, gate, other] = 6
#   - parameter_target: one-hot [Vmax, enzyme_activity, Km, other, unknown] = 5
#   - multiplier_value: continuous (0 if missing) = 1
#   - simulation_affecting: binary = 1
#   - source_strength: one-hot [A, B, C, D] = 4
#   Total: 4 + 6 + 5 + 1 + 1 + 4 = 21
SIGNALING_EDGE_FEAT_DIM = 21

# P1: Transport edge features (14 dimensions):
#
# Encodes compartment transport (edges_transport.jsonl, edges_bbb.jsonl).
# Transport edges link metabolites across compartments (systemic↔brain,
# cytosol↔vesicle) for compartmentalization modeling.
#
# Features:
#   - direction: one-hot [systemic_to_brain, brain_to_systemic,
#                 cytosol_to_vesicle, vesicle_to_cytosol, bidirectional] = 5
#   - is_bbb: binary (1 if BBB crossing, 0 if intracellular) = 1
#   - logbb: continuous (normalized log brain:blood ratio) = 1
#   - permeability: one-hot [BBB+, BBB-] = 2
#   - has_km: binary (1 if magnitude/Km available) = 1
#   - source_strength: one-hot [A, B, C, D] = 4
#
# Total: 5 + 1 + 1 + 2 + 1 + 4 = 14
TRANSPORT_EDGE_FEAT_DIM = 14


# ---- Graph metadata (exposed for audit/debug) ------------------------------

@dataclass
class GraphMeta:
    """Metadata about the constructed graph for debugging and audit.

    # Node Type ID Maps

    Each node type has a string→int map and reverse int→string map.
    These are used for:
    1. Looking up node indices during graph construction
    2. Converting predictions back to human-readable IDs for output

    # Edge Mappings

    For learned edges that produce outputs consumed by Rust:
    - bridge_idx_to_id: Maps bridge edge indices to bridge_id strings
    - modulates_idx_to_src/dst: Maps modulates edge indices to metabolite/enzyme IDs
    - transport_idx_to_pair: Maps transport edge indices to (src, dst) metabolite IDs

    # Scientific Rationale

    Exposing ID mappings enables:
    - **Auditability**: Trace predictions back to source data
    - **Interpretability**: Explain which bridges, modulators drove effects
    - **Integration**: Match GNN outputs to Rust flux engine inputs

    Note: SNP personalization is handled OUTSIDE the GNN via lookup tables.
    See personalization-architecture.md for details.
    """

    # ID maps: string → int index
    enz_map: dict[str, int] = field(default_factory=dict)
    met_map: dict[str, int] = field(default_factory=dict)
    rxn_map: dict[str, int] = field(default_factory=dict)

    # Reverse maps: int index → string ID
    enz_idx_to_id: dict[int, str] = field(default_factory=dict)
    met_idx_to_id: dict[int, str] = field(default_factory=dict)

    # pathway → domain lookup (informational, not used for routing)
    pathway_to_domain: dict[str, str] = field(default_factory=dict)

    # symbol → pathway lookup (from pathway_components)
    symbol_to_pathway: dict[str, str] = field(default_factory=dict)

    # bridge_edge_idx → bridge_id mapping (for output to Rust)
    bridge_idx_to_id: dict[int, str] = field(default_factory=dict)
    # bridge_id → bridge record (for audit/debug)
    bridges: dict[str, dict] = field(default_factory=dict)

    # modulates edge mappings (for output to Rust)
    # edge_idx → (metabolite_id, enzyme_id)
    modulates_idx_to_src: dict[int, str] = field(default_factory=dict)
    modulates_idx_to_dst: dict[int, str] = field(default_factory=dict)

    # P1: transport edge mappings (for compartmentalization output)
    # edge_idx → (src_metabolite_id, dst_metabolite_id)
    transport_idx_to_pair: dict[int, tuple[str, str]] = field(default_factory=dict)


# ---- Helpers ---------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Parse a numeric string, returning *default* on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _id_map(records: list[dict], key: str = "id") -> dict[str, int]:
    """Build {string_id: integer_index} from a list of dicts."""
    return {r[key]: i for i, r in enumerate(records)}


# ---- Edge builders ---------------------------------------------------------

def _filtered_pairs(
    records: list[dict],
    src_key: str,
    dst_key: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Return (src_indices, dst_indices), dropping rows with unknown IDs."""
    srcs, dsts = [], []
    for r in records:
        s = src_map.get(r.get(src_key, ""))
        d = dst_map.get(r.get(dst_key, ""))
        if s is not None and d is not None:
            srcs.append(s)
            dsts.append(d)
    return srcs, dsts


def _filtered_pairs_with_ids(
    records: list[dict],
    src_key: str,
    dst_key: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> tuple[list[int], list[int], dict[int, str], dict[int, str]]:
    """Return (src_indices, dst_indices, idx_to_src_id, idx_to_dst_id).

    Like _filtered_pairs but also returns edge_idx → original string ID mappings.
    Used for modulates edges where we need to map back to metabolite_enzyme keys.
    """
    srcs, dsts = [], []
    idx_to_src: dict[int, str] = {}
    idx_to_dst: dict[int, str] = {}

    for r in records:
        src_id = r.get(src_key, "")
        dst_id = r.get(dst_key, "")
        s = src_map.get(src_id)
        d = dst_map.get(dst_id)
        if s is not None and d is not None:
            edge_idx = len(srcs)
            srcs.append(s)
            dsts.append(d)
            idx_to_src[edge_idx] = src_id
            idx_to_dst[edge_idx] = dst_id

    return srcs, dsts, idx_to_src, idx_to_dst


# ---- Public API ------------------------------------------------------------

def build_graph(cfg: DataConfig | None = None) -> tuple[dgl.DGLHeteroGraph, GraphMeta]:
    """Build the full heterograph from seed JSONL.

    Returns
    -------
    tuple[dgl.DGLHeteroGraph, GraphMeta]
        Graph with node types: enzyme, metabolite, reaction.
        GraphMeta contains ID maps for debugging and audit.
    """
    cfg = cfg or DataConfig()

    # -- Load entities -------------------------------------------------------
    enzymes = _load_jsonl(cfg.enzymes_path)
    metabolites = _load_jsonl(cfg.metabolites_path)
    reactions = _load_jsonl(cfg.reactions_path)

    enz_map = _id_map(enzymes)
    met_map = _id_map(metabolites)
    rxn_map = _id_map(reactions)

    logger.info(
        "Entities: %d enzymes, %d metabolites, %d reactions",
        len(enz_map), len(met_map), len(rxn_map),
    )

    # -- Load pathway data (informational, not used for routing) -------------
    pathways = _load_jsonl(cfg.pathways_path)
    pathway_components = _load_jsonl(cfg.pathway_components_path)

    # pathway name → domain (for audit/debugging only)
    pathway_to_domain: dict[str, str] = {}
    for pw in pathways:
        pw_name = pw.get("name", "")
        domain = pw.get("domain", "unknown")
        if pw_name:
            pathway_to_domain[pw_name] = domain

    # symbol → pathway (from pathway_components)
    symbol_to_pathway: dict[str, str] = {}
    for pc in pathway_components:
        sym = pc.get("symbol", "")
        pw = pc.get("pathway", "")
        if sym and pw:
            symbol_to_pathway[sym] = pw

    logger.info("Loaded %d pathways, %d pathway_components", len(pathways), len(pathway_components))

    # -- Load edge files -----------------------------------------------------
    cat_recs = _load_jsonl(cfg.edges_catalyzes_path)
    part_recs = _load_jsonl(cfg.edges_reaction_participants_path)
    cof_recs = _load_jsonl(cfg.edges_cofactor_path)
    mod_recs = _load_jsonl(cfg.edges_modulates_path)
    reg_recs = _load_jsonl(cfg.edges_regulates_path)
    sig_recs = _load_jsonl(cfg.edges_signaling_path)
    bridge_recs = _load_jsonl(cfg.bridges_path)
    # P1: Load regulatory kinetics for quantitative modulation features
    reg_kinetics_recs = _load_jsonl(cfg.regulatory_kinetics_path)
    # P1: Load transport/BBB for compartmentalization
    transport_recs = _load_jsonl(cfg.edges_transport_path)
    bbb_recs = _load_jsonl(cfg.edges_bbb_path)
    # P1: Load inhibitors for competitive inhibition edges
    inhibitor_recs = _load_jsonl(cfg.inhibitors_path)
    # NOTE: edges_affects.jsonl is NOT loaded here - SNP personalization is
    # handled outside the GNN via lookup tables. See personalization-architecture.md.

    # -- Physics-informed edges ----------------------------------------------
    cat_s, cat_d = _filtered_pairs(cat_recs, "enzyme_id", "reaction_id", enz_map, rxn_map)

    sub_s, sub_d, prod_s, prod_d = [], [], [], []
    for r in part_recs:
        mi = met_map.get(r.get("metabolite_id", ""))
        ri = rxn_map.get(r.get("reaction_id", ""))
        if mi is None or ri is None:
            continue
        if r.get("side") == "reactant":
            sub_s.append(mi); sub_d.append(ri)
        else:  # product (or unknown — treat as product)
            prod_s.append(ri); prod_d.append(mi)

    cof_s, cof_d = _filtered_pairs(cof_recs, "cofactor_metabolite_id", "enzyme_id", met_map, enz_map)

    # -- Learned edges -------------------------------------------------------
    # P1: Modulates edges now include 13-dim regulatory kinetics features
    # for quantitative attention biasing based on EC50/IC50/Kd/Ki values.
    # Also includes competitive inhibitors from inhibitors.jsonl.
    mod_s, mod_d, mod_idx_to_src, mod_idx_to_dst, mod_edge_feats = _modulates_edges(
        mod_recs, reg_kinetics_recs, inhibitor_recs, met_map, enz_map
    )
    # Regulates: use _filtered_pairs_with_ids so we can tag per-edge semantics
    reg_s, reg_d, reg_idx_to_src, reg_idx_to_dst = _filtered_pairs_with_ids(
        reg_recs, "tf_id", "target_gene_id", enz_map, enz_map
    )

    # Signaling: signal_substance → bridge_mediator (via target_bridge_id lookup)
    # P0: Now includes edge features for quantitative crosstalk modeling
    sig_s, sig_d, sig_edge_feats = _signaling_edges(sig_recs, bridge_recs, met_map)

    # Bridges: mediator (metabolite) → target enzyme (inferred saturation)
    br_s, br_d, bridge_idx_to_id, bridge_meta, br_edge_feats = _bridge_edges(
        bridge_recs, met_map, enz_map
    )

    # P1: Transport edges for compartmentalization (systemic↔brain, cytosol↔vesicle)
    trans_s, trans_d, trans_idx_to_pair, trans_edge_feats = _transport_edges(
        transport_recs, bbb_recs, met_map
    )

    # -- Assemble heterograph ------------------------------------------------
    # Note: SNP personalization is handled outside the GNN via lookup tables.
    # P1: Now includes transport edges for compartmentalization.
    graph_data: dict[tuple, tuple] = {
        # Physics-informed edges
        ("enzyme", "catalyzes", "reaction"):       (cat_s, cat_d),
        ("metabolite", "substrate_of", "reaction"): (sub_s, sub_d),
        ("reaction", "produces", "metabolite"):     (prod_s, prod_d),
        ("metabolite", "cofactor_for", "enzyme"):   (cof_s, cof_d),
        # Learned edges
        ("metabolite", "modulates", "enzyme"):      (mod_s, mod_d),
        ("enzyme", "regulates", "enzyme"):          (reg_s, reg_d),
        ("metabolite", "signaling", "metabolite"):  (sig_s, sig_d),
        ("metabolite", "bridges", "enzyme"):        (br_s, br_d),
        # P1: Compartmentalization
        ("metabolite", "transports_to", "metabolite"): (trans_s, trans_d),
    }

    num_nodes = {
        "enzyme": len(enz_map),
        "metabolite": len(met_map),
        "reaction": len(rxn_map),
    }
    g = dgl.heterograph(graph_data, num_nodes_per_type=num_nodes)

    # -- Attach edge features -------------------------------------------------
    # Bridge edges: evidence-based features (10 dims)
    g.edges[("metabolite", "bridges", "enzyme")].data["feat"] = br_edge_feats
    # Signaling edges: crosstalk features (21 dims)
    g.edges[("metabolite", "signaling", "metabolite")].data["feat"] = sig_edge_feats
    # P1: Modulates edges: regulatory kinetics features (13 dims)
    g.edges[("metabolite", "modulates", "enzyme")].data["feat"] = mod_edge_feats
    # P1: Transport edges: compartment transport features (14 dims)
    g.edges[("metabolite", "transports_to", "metabolite")].data["feat"] = trans_edge_feats

    # -- Attach edge semantics (gated propagation control) ---------------------
    # Each edge gets a semantic label: METABOLIC_ANCHOR or LATENT_CONTEXT.
    # METABOLIC_ANCHOR edges can affect simulation (Vmax, Km).
    # LATENT_CONTEXT edges update hidden states but don't propagate to outputs.
    #
    # Semantic is stored as integer: 0 = METABOLIC_ANCHOR, 1 = LATENT_CONTEXT
    # This enables efficient masking during forward pass.
    SEMANTIC_TO_INT = {
        EdgeSemantic.METABOLIC_ANCHOR: 0,
        EdgeSemantic.LATENT_CONTEXT: 1,
    }
    anchor_count = 0
    for stype, etype, dtype in ALL_ETYPES:
        num_edges = g.num_edges((stype, etype, dtype))
        if num_edges > 0:
            if etype == "regulates":
                # Per-edge semantic for regulates based on REGULATES_ANCHOR_EDGES
                semantic_arr = []
                for i in range(num_edges):
                    tf_id = reg_idx_to_src.get(i, "")
                    target_id = reg_idx_to_dst.get(i, "")
                    if (tf_id, target_id) in REGULATES_ANCHOR_EDGES:
                        semantic_arr.append(SEMANTIC_TO_INT[EdgeSemantic.METABOLIC_ANCHOR])
                        anchor_count += 1
                    else:
                        semantic_arr.append(SEMANTIC_TO_INT[EdgeSemantic.LATENT_CONTEXT])
                g.edges[(stype, etype, dtype)].data["semantic"] = torch.tensor(
                    semantic_arr, dtype=torch.long
                )
            else:
                # Default semantic for non-regulates edges
                semantic = EDGE_SEMANTICS.get(etype, EdgeSemantic.LATENT_CONTEXT)
                semantic_int = SEMANTIC_TO_INT[semantic]
                g.edges[(stype, etype, dtype)].data["semantic"] = torch.full(
                    (num_edges,), semantic_int, dtype=torch.long
                )

    for etype, (s, _) in graph_data.items():
        if etype[1] == "regulates":
            logger.info(
                "  %s: %d edges (%d metabolic_anchor, %d latent_context)",
                etype[1], len(s), anchor_count, len(s) - anchor_count
            )
        else:
            semantic = EDGE_SEMANTICS.get(etype[1], EdgeSemantic.LATENT_CONTEXT)
            logger.info("  %s: %d edges (%s)", etype[1], len(s), semantic.value)

    # -- Build kinetics lookup for enzyme features ---------------------------
    kinetics = _load_jsonl(cfg.kinetics_path)
    kin_lookup: dict[str, dict[str, float]] = defaultdict(dict)
    for k in kinetics:
        eid = k.get("enzyme_id", "")
        pname = k.get("param_name", "")
        val = _safe_float(k.get("value"), 0.0)
        if eid and pname and val > 0:
            kin_lookup[eid][pname] = val

    # -- Attach node features ------------------------------------------------
    GRADE_MAP = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}

    # Enzyme: [baseline_km, baseline_vmax, kcat, source_grade]
    enz_feat = torch.zeros(len(enz_map), 4)
    for enz in enzymes:
        i = enz_map[enz["id"]]
        enz_feat[i, 0] = _safe_float(enz.get("baseline_km"))
        enz_feat[i, 1] = _safe_float(enz.get("baseline_vmax"))
        kin = kin_lookup.get(enz["id"], {})
        enz_feat[i, 2] = kin.get("kcat", 0.0)
        enz_feat[i, 3] = GRADE_MAP.get(enz.get("source_strength", ""), 0.25)
    g.nodes["enzyme"].data["feat"] = enz_feat

    # Metabolite: [baseline_vol, source_grade]
    met_feat = torch.zeros(len(met_map), 2)
    for met in metabolites:
        i = met_map[met["id"]]
        met_feat[i, 0] = _safe_float(met.get("baseline_vol"))
        met_feat[i, 1] = GRADE_MAP.get(met.get("source_strength", ""), 0.25)
    g.nodes["metabolite"].data["feat"] = met_feat

    # Reaction: [reversible_flag, source_grade]
    rxn_feat = torch.zeros(len(rxn_map), 2)
    for rxn in reactions:
        i = rxn_map[rxn["id"]]
        rxn_feat[i, 0] = 1.0 if rxn.get("reversible") == "true" else 0.0
        rxn_feat[i, 1] = GRADE_MAP.get(rxn.get("source_strength", ""), 0.25)
    g.nodes["reaction"].data["feat"] = rxn_feat

    # Note: SNP personalization is handled outside the GNN via lookup tables.
    # See personalization-architecture.md for the wild type GNN + SNP lookup design.

    # -- Build reverse ID maps -----------------------------------------------
    enz_idx_to_id = {v: k for k, v in enz_map.items()}
    met_idx_to_id = {v: k for k, v in met_map.items()}

    # -- Build metadata ------------------------------------------------------
    # P1: Now includes transport edge maps for compartmentalization
    meta = GraphMeta(
        enz_map=enz_map,
        met_map=met_map,
        rxn_map=rxn_map,
        enz_idx_to_id=enz_idx_to_id,
        met_idx_to_id=met_idx_to_id,
        pathway_to_domain=pathway_to_domain,
        symbol_to_pathway=symbol_to_pathway,
        bridge_idx_to_id=bridge_idx_to_id,
        bridges=bridge_meta,
        modulates_idx_to_src=mod_idx_to_src,
        modulates_idx_to_dst=mod_idx_to_dst,
        transport_idx_to_pair=trans_idx_to_pair,  # P1: edge idx → (src, dst) metabolite IDs
    )

    return g, meta


def _signaling_edges(
    sig_recs: list[dict],
    bridge_recs: list[dict],
    met_map: dict[str, int],
) -> tuple[list[int], list[int], torch.Tensor]:
    """Create metabolite→metabolite edges from signaling_crosstalk with features.

    # P0: Signaling Edge Features

    Previously we only created edges but ignored the rich feature data.
    Now we extract all features to enable quantitative crosstalk modeling:

    - **effect_direction**: How the signal affects target (increased/decreased/mixed/unknown)
    - **logic_type**: Regulatory logic (activate/inhibit/feedback/feedforward/gate)
    - **parameter_target**: Which kinetic parameter is affected (Vmax/Km/enzyme_activity)
    - **multiplier_value**: Quantitative effect magnitude when available
    - **simulation_affecting**: Whether this crosstalk should affect simulation
    - **source_strength**: Evidence quality (A/B/C/D)

    # Feature Encoding (21 dimensions)

    | Feature | Encoding | Dims |
    |---------|----------|------|
    | effect_direction | one-hot [increased, decreased, mixed, unknown] | 4 |
    | logic_type | one-hot [activate, inhibit, feedback, feedforward, gate, other] | 6 |
    | parameter_target | one-hot [Vmax, enzyme_activity, Km, other, unknown] | 5 |
    | multiplier_value | continuous (normalized) | 1 |
    | simulation_affecting | binary | 1 |
    | source_strength | one-hot [A, B, C, D] | 4 |
    | **Total** | | **21** |

    # Scientific Rationale

    Edge features as attention bias (Graphormer-style, Ying et al. 2021):
    - Rich edge semantics inform attention weights
    - Effect direction tells GNN whether to propagate increase/decrease
    - Logic type encodes regulatory structure (feedback loops, gates)
    - Parameter target tells which kinetic parameter is modulated

    Returns
    -------
    tuple of:
        srcs: list[int] — source metabolite indices
        dsts: list[int] — destination metabolite indices
        edge_feats: torch.Tensor — (num_edges, 21) signaling features
    """
    # Build bridge_id → mediator_id lookup
    bridge_to_mediator: dict[str, str] = {}
    for br in bridge_recs:
        bid = br.get("bridge_id", "")
        mid = br.get("mediator_id", "")
        if bid and mid:
            bridge_to_mediator[bid] = mid

    # Feature vocabularies (order matters for one-hot encoding)
    EFFECT_DIRS = ["increased", "decreased", "mixed", "unknown"]
    LOGIC_TYPES = ["activate", "inhibit", "feedback", "feedforward", "gate", "other"]
    PARAM_TARGETS = ["Vmax", "enzyme_activity", "Km", "other", "unknown"]
    GRADES = ["A", "B", "C", "D"]

    srcs, dsts = [], []
    edge_feat_list: list[list[float]] = []

    for sig in sig_recs:
        sig_sub = sig.get("signal_substance_id", "")
        target_bid = sig.get("target_bridge_id", "")

        src_idx = met_map.get(sig_sub)
        mediator = bridge_to_mediator.get(target_bid, "")
        dst_idx = met_map.get(mediator)

        if src_idx is not None and dst_idx is not None and src_idx != dst_idx:
            srcs.append(src_idx)
            dsts.append(dst_idx)

            # -- Extract features --
            # effect_direction one-hot (4 dims)
            eff_dir = sig.get("effect_direction", "unknown")
            eff_dir_vec = [1.0 if eff_dir == v else 0.0 for v in EFFECT_DIRS]

            # logic_type one-hot (6 dims)
            logic = sig.get("logic_type", "other")
            logic_vec = [1.0 if logic == v else 0.0 for v in LOGIC_TYPES]

            # parameter_target one-hot (5 dims)
            param = sig.get("parameter_target", "unknown")
            param_vec = [1.0 if param == v else 0.0 for v in PARAM_TARGETS]

            # multiplier_value continuous (1 dim)
            mult_str = sig.get("multiplier_value", "")
            multiplier = _safe_float(mult_str, 0.0)
            # Normalize: most multipliers are in [0, 2] range
            mult_norm = min(multiplier, 5.0) / 5.0  # Cap at 5, normalize to [0,1]

            # simulation_affecting binary (1 dim)
            sim_aff = 1.0 if sig.get("simulation_affecting", False) else 0.0

            # source_strength one-hot (4 dims)
            grade = sig.get("source_strength", "D")
            grade_vec = [1.0 if grade == v else 0.0 for v in GRADES]

            # Combine: 4 + 6 + 5 + 1 + 1 + 4 = 21
            feat = eff_dir_vec + logic_vec + param_vec + [mult_norm, sim_aff] + grade_vec
            edge_feat_list.append(feat)

    # Convert to tensor
    if edge_feat_list:
        edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        edge_feats = torch.zeros((0, SIGNALING_EDGE_FEAT_DIM), dtype=torch.float32)

    logger.info(
        "Built %d signaling edges (metabolite→metabolite) with %d-dim features",
        len(srcs), edge_feats.shape[1] if edge_feats.numel() > 0 else SIGNALING_EDGE_FEAT_DIM
    )
    return srcs, dsts, edge_feats


# ---------------------------------------------------------------------------
# P1: Modulates Edges with Regulatory Kinetics Features (13 dimensions)
# ---------------------------------------------------------------------------

# Feature dimension constant
MODULATES_EDGE_FEAT_DIM = 13


def _modulates_edges(
    mod_recs: list[dict],
    reg_kinetics_recs: list[dict],
    inhibitor_recs: list[dict],
    met_map: dict[str, int],
    enz_map: dict[str, int],
) -> tuple[list[int], list[int], dict[int, str], dict[int, str], torch.Tensor]:
    """Create metabolite→enzyme modulates edges with regulatory kinetics features.

    # P1: Quantitative Modulation via EC50/IC50/Kd/Ki + Competitive Inhibitors

    Previously we created modulates edges without features. Now we enrich them
    with data from regulatory_kinetics.jsonl AND inhibitors.jsonl:

    - **regulatory_kinetics.jsonl**: EC50/IC50/Kd/Ki values for modulators
    - **inhibitors.jsonl**: Ki values for competitive/allosteric inhibitors

    Combined, this enables quantitative attention biasing:

    - **normalized_measure**: Type of affinity measurement (EC50/IC50/Kd/Ki)
    - **action**: Direction of modulation (ACTIVATE/BIND/MODULATE/INHIBIT)
    - **normalized_value**: Potency in nM (lower = more potent)
    - **source_strength**: Evidence quality (A/B/C/D)

    # Feature Encoding (13 dimensions)

    | Feature | Encoding | Dims |
    |---------|----------|------|
    | normalized_measure | one-hot [EC50, IC50, Kd, Ki] | 4 |
    | action | one-hot [ACTIVATE, BIND, MODULATE, INHIBIT] | 4 |
    | pValue | continuous pEC50 scale, normalized [0,1] | 1 |
    | source_strength | one-hot [A, B, C, D] | 4 |
    | **Total** | | **13** |

    # pValue Normalization

    We convert concentration to -log10 scale (pEC50/pIC50/pKd/pKi):
        pValue = -log10(concentration_M)
               = -log10(concentration_nM × 1e-9)
               = 9 - log10(concentration_nM)

    Then normalize to [0, 1]:
        norm_pValue = (pValue - 3) / 9

    This maps:
        - 1 mM (weak) → pValue=3 → norm=0.0
        - 1 nM (strong) → pValue=9 → norm=0.67
        - 1 pM (very strong) → pValue=12 → norm=1.0

    # Scientific Rationale

    Edge features as attention bias (Graphormer-style):
    - High-affinity modulators (low EC50, high pValue) should dominate
    - Action type tells GNN whether effect is activation or inhibition
    - Measure type indicates assay reliability (binding vs functional)

    References:
    - Cheng & Prusoff (1973) BBRC "Relationship between inhibition constant"
    - Swinney (2011) Nat Rev Drug Discov "Biochemical mechanisms of drug action"

    Returns
    -------
    tuple of:
        srcs: list[int] — source metabolite indices
        dsts: list[int] — destination enzyme indices
        idx_to_src: dict[int, str] — edge_idx → metabolite_id
        idx_to_dst: dict[int, str] — edge_idx → enzyme_id
        edge_feats: torch.Tensor — (num_edges, 13) regulatory kinetics features
    """
    # Build lookup from (substance_id, target_id) → regulatory kinetics record
    # Multiple records may exist for the same pair; we take the first one with
    # a valid normalized_value (highest quality).
    kinetics_lookup: dict[tuple[str, str], dict] = {}
    for rec in reg_kinetics_recs:
        key = (rec.get("substance_id", ""), rec.get("target_id", ""))
        if key[0] and key[1]:
            # Only store if we don't already have one OR if this has a better value
            existing = kinetics_lookup.get(key)
            if existing is None:
                kinetics_lookup[key] = rec
            elif not existing.get("normalized_value") and rec.get("normalized_value"):
                # Prefer records with actual values
                kinetics_lookup[key] = rec

    # Feature vocabularies (order matters for one-hot encoding)
    MEASURES = ["EC50", "IC50", "Kd", "Ki"]
    ACTIONS = ["ACTIVATE", "BIND", "MODULATE", "INHIBIT"]
    GRADES = ["A", "B", "C", "D"]

    srcs, dsts = [], []
    idx_to_src: dict[int, str] = {}
    idx_to_dst: dict[int, str] = {}
    edge_feat_list: list[list[float]] = []

    for rec in mod_recs:
        src_id = rec.get("substance_id", "")
        dst_id = rec.get("enzyme_id", "")

        src_idx = met_map.get(src_id)
        dst_idx = enz_map.get(dst_id)

        if src_idx is None or dst_idx is None:
            continue

        edge_idx = len(srcs)
        srcs.append(src_idx)
        dsts.append(dst_idx)
        idx_to_src[edge_idx] = src_id
        idx_to_dst[edge_idx] = dst_id

        # Look up regulatory kinetics for this edge
        kin = kinetics_lookup.get((src_id, dst_id), {})

        # -- Extract features --
        # normalized_measure one-hot (4 dims)
        measure = kin.get("normalized_measure", "")
        measure_vec = [1.0 if measure == v else 0.0 for v in MEASURES]

        # action one-hot (4 dims) - fall back to modulates record if not in kinetics
        action = kin.get("action", "") or rec.get("action", "")
        action_vec = [1.0 if action == v else 0.0 for v in ACTIONS]

        # pValue continuous (1 dim)
        # Convert normalized_value (nM) to pValue scale
        value_str = kin.get("normalized_value", "")
        pvalue_norm = 0.0  # default for missing values
        if value_str:
            try:
                conc_nm = float(value_str)
                if conc_nm > 0:
                    import math
                    # pValue = 9 - log10(conc_nM)
                    pvalue = 9.0 - math.log10(conc_nm)
                    # Normalize to [0, 1]: (pValue - 3) / 9
                    # Clamp to [0, 1] for safety
                    pvalue_norm = max(0.0, min(1.0, (pvalue - 3.0) / 9.0))
            except (ValueError, TypeError):
                pass

        # source_strength one-hot (4 dims)
        grade = kin.get("source_strength", "") or rec.get("source_strength", "D")
        grade_vec = [1.0 if grade == v else 0.0 for v in GRADES]

        # Combine all features (13 total)
        edge_feat = measure_vec + action_vec + [pvalue_norm] + grade_vec
        edge_feat_list.append(edge_feat)

    mod_count = len(srcs)

    # ---------------------------------------------------------------------------
    # P1: Also process inhibitors.jsonl for competitive inhibitors
    # ---------------------------------------------------------------------------
    # Inhibitors with inhibitor_metabolite_id can be added as modulates edges.
    # They use Ki as the measure and INHIBIT as the action.

    def _parse_ki_value(ki_str: str) -> float | None:
        """Parse Ki string like '4.2 {alpha-Ketoisocaproate}' to float."""
        if not ki_str:
            return None
        # Extract numeric part before any {annotation}
        import re
        match = re.match(r"^\s*([0-9.]+)", ki_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    # Track edges we've already added to avoid duplicates
    existing_edges: set[tuple[str, str]] = set()
    for i, src_id in idx_to_src.items():
        dst_id = idx_to_dst.get(i, "")
        existing_edges.add((src_id, dst_id))

    inhibitor_added = 0
    for rec in inhibitor_recs:
        # Only process records with inhibitor_metabolite_id mapping
        src_id = rec.get("inhibitor_metabolite_id", "")
        dst_id = rec.get("enzyme_id", "")

        if not src_id or not dst_id:
            continue

        # Check if this edge already exists
        if (src_id, dst_id) in existing_edges:
            continue

        src_idx = met_map.get(src_id)
        dst_idx = enz_map.get(dst_id)

        if src_idx is None or dst_idx is None:
            continue

        # Add edge
        edge_idx = len(srcs)
        srcs.append(src_idx)
        dsts.append(dst_idx)
        idx_to_src[edge_idx] = src_id
        idx_to_dst[edge_idx] = dst_id
        existing_edges.add((src_id, dst_id))

        # -- Features for inhibitor --
        # normalized_measure: Ki (inhibitors always use Ki)
        measure_vec = [0.0, 0.0, 0.0, 1.0]  # Ki is index 3

        # action: INHIBIT (inhibitors are always inhibitors)
        action_vec = [0.0, 0.0, 0.0, 1.0]  # INHIBIT is index 3

        # pValue from Ki (assumed in µM, convert to nM)
        pvalue_norm = 0.0
        ki_value = _parse_ki_value(rec.get("ki", ""))
        if ki_value is not None and ki_value > 0:
            # Ki is typically in µM for BRENDA data
            conc_nm = ki_value * 1000.0  # µM → nM
            pvalue = 9.0 - math.log10(conc_nm)
            pvalue_norm = max(0.0, min(1.0, (pvalue - 3.0) / 9.0))

        # source_strength (most BRENDA data is D-grade)
        grade = rec.get("source_strength", "D")
        grade_vec = [1.0 if grade == v else 0.0 for v in GRADES]

        edge_feat = measure_vec + action_vec + [pvalue_norm] + grade_vec
        edge_feat_list.append(edge_feat)
        inhibitor_added += 1

    # Convert to tensor
    if edge_feat_list:
        edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        edge_feats = torch.zeros((0, MODULATES_EDGE_FEAT_DIM), dtype=torch.float32)

    logger.info(
        "Built %d modulates edges (metabolite→enzyme) with %d-dim features "
        "(%d from edges_modulates, %d from inhibitors)",
        len(srcs), edge_feats.shape[1] if edge_feats.numel() > 0 else MODULATES_EDGE_FEAT_DIM,
        mod_count, inhibitor_added
    )
    return srcs, dsts, idx_to_src, idx_to_dst, edge_feats


def _transport_edges(
    transport_recs: list[dict],
    bbb_recs: list[dict],
    met_map: dict[str, int],
) -> tuple[list[int], list[int], dict[int, tuple[str, str]], torch.Tensor]:
    """Create metabolite→metabolite transport edges for compartmentalization.

    # P1: Compartment Transport via Transport Edges

    Transport edges link metabolites across compartments:
    - systemic ↔ brain (BBB crossing)
    - cytosol ↔ vesicle (intracellular transport)

    The GNN learns transport multipliers ∈ [0, 2] that modulate flux
    across compartment boundaries. Values < 1 indicate restricted transport
    (e.g., BBB-), values > 1 indicate facilitated transport (e.g., LAT1).

    # Feature Encoding (14 dimensions)

    | Feature | Encoding | Dims |
    |---------|----------|------|
    | direction | one-hot [systemic_to_brain, brain_to_systemic, cytosol_to_vesicle, vesicle_to_cytosol, bidirectional] | 5 |
    | is_bbb | binary (BBB vs intracellular) | 1 |
    | logbb | continuous, normalized to [-1, 1] | 1 |
    | permeability | one-hot [BBB+, BBB-] | 2 |
    | has_km | binary (magnitude/Km available) | 1 |
    | source_strength | one-hot [A, B, C, D] | 4 |
    | **Total** | | **14** |

    # Data Sources

    - edges_transport.jsonl (372 records): Compartment transport with direction,
      transporter_id, magnitude (Km values)
    - edges_bbb.jsonl (103 records): BBB-specific data with logbb, permeability

    Returns
    -------
    tuple of:
        srcs: list[int] — source metabolite indices
        dsts: list[int] — destination metabolite indices
        idx_to_pair: dict[int, tuple[str, str]] — edge_idx → (src_met_id, dst_met_id)
        edge_feats: torch.Tensor — (num_edges, 14) transport features
    """
    # Build BBB lookup: systemic_id → BBB record
    bbb_lookup: dict[str, dict] = {}
    for rec in bbb_recs:
        sys_id = rec.get("systemic_metabolite_id", "")
        if sys_id:
            bbb_lookup[sys_id] = rec

    # Feature vocabularies
    DIRECTIONS = [
        "systemic_to_brain", "brain_to_systemic",
        "cytosol_to_vesicle", "vesicle_to_cytosol", "bidirectional"
    ]
    GRADES = ["A", "B", "C", "D"]

    srcs, dsts = [], []
    idx_to_pair: dict[int, tuple[str, str]] = {}
    edge_feat_list: list[list[float]] = []

    # Process transport records
    for rec in transport_recs:
        met_id = rec.get("metabolite_id", "")
        direction = rec.get("direction", "bidirectional")
        from_comp = rec.get("from_compartment", "")
        to_comp = rec.get("to_compartment", "")

        # Determine source and destination metabolite IDs
        # Convention: brain metabolites have _BRAIN suffix
        if from_comp == "systemic" and to_comp == "brain":
            src_id = met_id
            dst_id = f"{met_id}_BRAIN"
            is_bbb = True
        elif from_comp == "brain" and to_comp == "systemic":
            src_id = f"{met_id}_BRAIN"
            dst_id = met_id
            is_bbb = True
        elif from_comp == "cytosol" and to_comp == "vesicle":
            src_id = met_id
            dst_id = met_id  # Same node for intracellular (edge encodes direction)
            is_bbb = False
        elif from_comp == "vesicle" and to_comp == "cytosol":
            src_id = met_id
            dst_id = met_id
            is_bbb = False
        else:
            # Unknown compartment pair — skip
            continue

        src_idx = met_map.get(src_id)
        dst_idx = met_map.get(dst_id)

        if src_idx is None or dst_idx is None:
            # Skip if metabolite not in graph
            continue

        edge_idx = len(srcs)
        srcs.append(src_idx)
        dsts.append(dst_idx)
        idx_to_pair[edge_idx] = (src_id, dst_id)

        # Build features

        # direction one-hot (5 dims)
        dir_vec = [1.0 if direction == d else 0.0 for d in DIRECTIONS]

        # is_bbb (1 dim)
        is_bbb_vec = [1.0 if is_bbb else 0.0]

        # logbb (1 dim) — normalized to [-1, 1] range
        # logBB typically ranges from -2 to +1 (negative = poor penetration)
        logbb = 0.0
        if is_bbb and met_id in bbb_lookup:
            logbb_str = bbb_lookup[met_id].get("logbb", "")
            try:
                logbb_raw = float(logbb_str) if logbb_str else 0.0
                # Normalize: -2 → -1, 0 → 0, +1 → 0.5
                logbb = max(-1.0, min(1.0, logbb_raw / 2.0))
            except (ValueError, TypeError):
                logbb = 0.0
        logbb_vec = [logbb]

        # permeability one-hot (2 dims)
        perm = ""
        if is_bbb and met_id in bbb_lookup:
            perm = bbb_lookup[met_id].get("permeability", "")
        perm_vec = [
            1.0 if perm == "BBB+" else 0.0,
            1.0 if perm == "BBB-" else 0.0
        ]

        # has_km (1 dim) — magnitude field contains Km values
        magnitude = rec.get("magnitude", "")
        has_km = 1.0 if magnitude and magnitude.strip() else 0.0
        has_km_vec = [has_km]

        # source_strength one-hot (4 dims)
        grade = rec.get("source_strength", "D")
        grade_vec = [1.0 if grade == g else 0.0 for g in GRADES]

        # Combine all features (14 total)
        edge_feat = dir_vec + is_bbb_vec + logbb_vec + perm_vec + has_km_vec + grade_vec
        edge_feat_list.append(edge_feat)

    # Convert to tensor
    if edge_feat_list:
        edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        edge_feats = torch.zeros((0, TRANSPORT_EDGE_FEAT_DIM), dtype=torch.float32)

    logger.info(
        "Built %d transport edges (metabolite→metabolite) with %d-dim features",
        len(srcs), edge_feats.shape[1] if edge_feats.numel() > 0 else TRANSPORT_EDGE_FEAT_DIM
    )
    return srcs, dsts, idx_to_pair, edge_feats


def _bridge_edges(
    bridge_recs: list[dict],
    met_map: dict[str, int],
    enz_map: dict[str, int],
) -> tuple[list[int], list[int], dict[int, str], dict[str, dict], torch.Tensor]:
    """Create metabolite→enzyme bridge edges for GNN inference.

    Each bridge record becomes one edge: mediator (metabolite) → target enzyme.
    The GNN infers a hidden state (saturation level) for each bridge.

    Returns
    -------
    tuple of:
        srcs: list[int] — source metabolite indices
        dsts: list[int] — destination enzyme indices
        idx_to_id: dict[int, str] — edge_idx → bridge_id mapping
        bridges: dict[str, dict] — bridge_id → full bridge record
        edge_feats: torch.Tensor — (num_edges, feat_dim) evidence features
    """
    srcs, dsts = [], []
    idx_to_id: dict[int, str] = {}
    bridges: dict[str, dict] = {}
    edge_feat_list: list[list[float]] = []

    for br in bridge_recs:
        # Skip inactive bridges (D-grade bridges are marked inactive)
        if not br.get("active", True):
            continue

        bridge_id = br.get("bridge_id", "")
        mediator_id = br.get("mediator_id", "")

        # Extract target enzyme from bridge_id (e.g., "SAMe_HNMT" → "HNMT")
        target_enzyme = bridge_id.rsplit("_", 1)[-1] if "_" in bridge_id else ""

        med_idx = met_map.get(mediator_id)
        enz_idx = enz_map.get(target_enzyme)

        if med_idx is None:
            logger.debug("Bridge %s: mediator %s not in metabolites", bridge_id, mediator_id)
            continue
        if enz_idx is None:
            logger.debug("Bridge %s: enzyme %s not in enzymes", bridge_id, target_enzyme)
            continue

        edge_idx = len(srcs)
        srcs.append(med_idx)
        dsts.append(enz_idx)
        idx_to_id[edge_idx] = bridge_id
        bridges[bridge_id] = {
            "mediator_id": mediator_id,
            "mediator_type": br.get("mediator_type", ""),
            "source_pathway": br.get("source_pathway", ""),
            "target_pathway": br.get("target_pathway", ""),
            "target_enzyme": target_enzyme,
        }

        # -- Compute evidence features (let GNN learn the weights) -------------
        # Source strength one-hot: [A, B, C] — D excluded by active filter
        grade = br.get("source_strength", "")
        grade_A = 1.0 if grade == "A" else 0.0
        grade_B = 1.0 if grade == "B" else 0.0
        grade_C = 1.0 if grade == "C" else 0.0

        # Count quality sources (A/B/C only)
        sources = br.get("sources", [])
        num_quality = sum(
            1 for s in sources if s.get("source_strength") in ("A", "B", "C")
        )
        # Normalize to [0, 1] range (cap at 5 sources)
        num_quality_norm = min(num_quality, 5) / 5.0

        # Claim type one-hot: [mechanistic, observational, genetic]
        claim = br.get("bridge_claim_type", "")
        claim_mech = 1.0 if claim == "mechanistic" else 0.0
        claim_obs = 1.0 if claim == "observational" else 0.0
        claim_gen = 1.0 if claim == "genetic" else 0.0

        # Evidence tier one-hot: [functional, genetic, observational]
        tier = br.get("evidence_tier", "")
        tier_func = 1.0 if tier == "functional" else 0.0
        tier_gen = 1.0 if tier == "genetic" else 0.0
        tier_obs = 1.0 if tier == "observational" else 0.0

        # Feature vector: 10 dimensions
        # [grade_A, grade_B, grade_C, num_quality_norm,
        #  claim_mech, claim_obs, claim_gen,
        #  tier_func, tier_gen, tier_obs]
        edge_feat_list.append([
            grade_A, grade_B, grade_C, num_quality_norm,
            claim_mech, claim_obs, claim_gen,
            tier_func, tier_gen, tier_obs,
        ])

    # Convert to tensor
    if edge_feat_list:
        edge_feats = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        edge_feats = torch.zeros((0, 10), dtype=torch.float32)

    logger.info("Built %d bridge edges (metabolite→enzyme) with %d-dim evidence features",
                len(srcs), edge_feats.shape[1] if edge_feats.numel() > 0 else 0)
    return srcs, dsts, idx_to_id, bridges, edge_feats


# NOTE: _affects_edges() function was removed per personalization-architecture.md.
# SNP personalization is now handled outside the GNN via lookup tables in the
# Rust flux engine. The GNN learns wild type biochemistry; SNPs are post-GNN
# multipliers applied via variant_kinetics.jsonl. edges_affects.jsonl is used
# only for explainability (linking SNPs to enzymes for user-facing explanations).
