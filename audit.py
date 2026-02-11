"""
AuditTrace — built-in auditability for every GNN inference pass.

Not a bolt-on explainability tool.  Just collecting what the model
already computes: hidden states, confidence scores, GAT attention weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EdgeAttribution:
    """A single edge with its inferred hidden state and confidence."""

    edge_type: str
    edge_idx: int
    hidden_value: float
    confidence: float


@dataclass
class AttentionSummary:
    """Per-edge-type attention weight summary from GAT layers.

    These attention weights explain which cross-pathway signals
    the GNN used to make its predictions.
    """

    edge_type: str
    num_edges: int
    mean_attention: float
    max_attention: float
    top_edge_indices: list[int]  # Edges with highest attention


@dataclass
class AuditTrace:
    """Full audit trail for one calibration inference.

    Constructed during the forward pass — no post-hoc explainability needed.

    Fields
    ------
    calibration_id : str
        Unique ID for this calibration run.
    top_edges : list[EdgeAttribution]
        Highest-confidence edges that drove the prediction.
    attention_summary : list[AttentionSummary]
        Per-edge-type GAT attention weight summary.
    provenance : list[dict]
        Chain from symptom → edge → enzyme → variant → PMID.
    """

    calibration_id: str = ""
    top_edges: list[EdgeAttribution] = field(default_factory=list)
    attention_summary: list[AttentionSummary] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "calibration_id": self.calibration_id,
            "top_edges": [
                {
                    "edge_type": e.edge_type,
                    "edge_idx": e.edge_idx,
                    "hidden_value": round(e.hidden_value, 6),
                    "confidence": round(e.confidence, 4),
                }
                for e in self.top_edges
            ],
            "attention_summary": [
                {
                    "edge_type": a.edge_type,
                    "num_edges": a.num_edges,
                    "mean_attention": round(a.mean_attention, 4),
                    "max_attention": round(a.max_attention, 4),
                    "top_edge_indices": a.top_edge_indices[:10],
                }
                for a in self.attention_summary
            ],
            "provenance": self.provenance,
        }

