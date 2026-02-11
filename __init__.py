"""
PathwaysMD GNN Calibrator — Inverse metabolic inference engine.

Single heterogeneous GNN that infers hidden states on learned edges
(modulates, regulates, signaling, bridges) from wild-type vs SNP-perturbed flux
deltas + user-reported symptoms + genotype.

Architecture:
    - Input projection: per-node-type linear → shared hidden_dim
    - Message passing: N layers with:
        * KineticConv on physics edges (Michaelis-Menten form locked)
        * GATConv on learned edges (attention weights retained for audit)
    - Edge heads: per learned-edge-type MLP → hidden state scalars
    - Built-in auditability via GAT attention weights + confidence scores
"""

__version__ = "0.3.0"

from .calibrate import calibrate  # noqa: F401 — public API
from .audit import AuditTrace  # noqa: F401
from .model import PathwayGNN  # noqa: F401

