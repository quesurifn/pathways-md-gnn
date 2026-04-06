#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from config import ModelConfig
from observation_features import OBSERVATION_FEATURE_DIM, OBSERVATION_FEATURE_NAMES

CONTEXT_CHANNELS: tuple[str, ...] = (
    "flux_delta",
    "symptom_count",
    "symptom_severity_hash",
    "symptom_category_signal",
    "lifestyle_enzyme_effects",
    "lifestyle_burden",
    "intervention_exposure",
    "genotype_burden",
)


def default_model_config() -> dict[str, Any]:
    config = ModelConfig()
    return {
        "node_dim": config.node_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "gat_heads": config.gat_heads,
        "edge_head_hidden_dim": config.edge_head_hidden_dim,
    }


def default_feature_contract() -> dict[str, Any]:
    return {
        "context_feature_dim": 8,
        "observation_feature_dim": OBSERVATION_FEATURE_DIM,
        "observation_feature_names": list(OBSERVATION_FEATURE_NAMES),
        "channels": list(CONTEXT_CHANNELS),
    }


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_weight_map(raw: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    if not raw:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, value in raw.items():
        items: list[dict[str, Any]] = []
        if isinstance(value, list):
            for entry in value:
                if not isinstance(entry, dict):
                    continue
                target_id = str(entry.get("id", "")).strip()
                weight = float(entry.get("weight", 0.0))
                if target_id:
                    items.append({"id": target_id, "weight": weight})
        if items:
            out[str(key).strip().upper()] = items
    return out


def normalize_bias_map(raw: dict[str, Any] | None) -> dict[str, float]:
    if not raw:
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        normalized = str(key).strip()
        if normalized:
            out[normalized] = float(value)
    return out


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    return {
        "manifest": {
            "kind": "pathways_gnn_bundle",
            "version": 2,
            "serving_contract": config.get("serving_contract", args.serving_contract),
            "feature_schema_version": config.get("feature_schema_version", args.feature_schema_version),
            "graph_schema_version": config.get("graph_schema_version", args.graph_schema_version),
            "model": config.get("model", args.model),
            "artifact_version": config.get("artifact_version", args.artifact_version),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "assets": {
                "scoring_artifact": "scoring_artifact.json",
                "model_config": "model_config.json",
                "graph_meta": "graph_meta.json",
                "feature_contract": "feature_contract.json",
            },
        },
        "scoring_artifact": {
            "node_biases": normalize_bias_map(config.get("node_biases")),
            "recommendation_biases": normalize_bias_map(config.get("recommendation_biases")),
            "symptom_node_weights": normalize_weight_map(config.get("symptom_node_weights")),
            "symptom_recommendation_weights": normalize_weight_map(config.get("symptom_recommendation_weights")),
            "substance_node_weights": normalize_weight_map(config.get("substance_node_weights")),
            "substance_recommendation_weights": normalize_weight_map(config.get("substance_recommendation_weights")),
            "focus_symptom_multiplier": float(config.get("focus_symptom_multiplier", args.focus_symptom_multiplier)),
            "engine_recommendation_multiplier": float(
                config.get("engine_recommendation_multiplier", args.engine_recommendation_multiplier)
            ),
        },
        "model_config": config.get("model_config", default_model_config()),
        "graph_meta": config.get("graph_meta", {"bridge_idx_to_id": []}),
        "feature_contract": config.get("feature_contract", default_feature_contract()),
    }


def write_bundle(out: Path, artifact: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(artifact["manifest"], indent=2, sort_keys=True) + "\n")
        bundle.writestr(
            "scoring_artifact.json",
            json.dumps(artifact["scoring_artifact"], indent=2, sort_keys=True) + "\n",
        )
        bundle.writestr("model_config.json", json.dumps(artifact["model_config"], indent=2, sort_keys=True) + "\n")
        bundle.writestr("graph_meta.json", json.dumps(artifact["graph_meta"], indent=2, sort_keys=True) + "\n")
        bundle.writestr(
            "feature_contract.json",
            json.dumps(artifact["feature_contract"], indent=2, sort_keys=True) + "\n",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Rust-serving GNN artifact bundle.")
    parser.add_argument("--config", type=Path, help="JSON spec containing scoring maps and metadata")
    parser.add_argument("--checkpoint", type=Path, help="Optional training checkpoint path for lineage metadata")
    parser.add_argument("--out", type=Path, required=True, help="Output artifact bundle ZIP path")
    parser.add_argument("--model", default="pathways_gnn_v1")
    parser.add_argument("--artifact-version", default="v1")
    parser.add_argument("--serving-contract", default="msgpack-scoring/v2")
    parser.add_argument("--feature-schema-version", default="v1")
    parser.add_argument("--graph-schema-version", default="v1")
    parser.add_argument("--focus-symptom-multiplier", type=float, default=1.25)
    parser.add_argument("--engine-recommendation-multiplier", type=float, default=0.5)
    args = parser.parse_args()

    artifact = build_artifact(args)
    write_bundle(args.out, artifact)
    print(args.out)


if __name__ == "__main__":
    main()
