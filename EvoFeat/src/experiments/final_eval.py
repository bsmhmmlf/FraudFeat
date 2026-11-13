from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from system.evolution.real_iv import compute_real_iv
from system.paths import PROJECT_ROOT
from system.utils import load_config


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _public_panel_ids(path: Path) -> set[str]:
    import pandas as pd
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "unique_id" not in frame.columns:
        raise ValueError(f"Evaluation panel must contain unique_id: {path}")
    return {str(value) for value in frame["unique_id"].tolist()}


def _features_sha256(features: list[dict]) -> str:
    """Fingerprint only feature definitions that affect public effectiveness evaluation."""
    definitions = [{
        "feature_name": str(feature.get("feature_name") or f"feature_{index}"),
        "pool_index": int(feature.get("pool_index", index)),
        "expected_value_type": str(feature.get("expected_value_type") or ""),
        "gremlin_plan": feature.get("plan") or feature.get("gremlin_plan") or {},
    } for index, feature in enumerate(features)]
    encoded = json.dumps(definitions, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_public_run(run_dir: str | Path, data_dir: str | Path,
                        evaluation_panel: str | Path | None = None,
                        limit: int = 100, force: bool = False) -> dict:
    """Evaluate the public run through the exact shared FraudFeat implementations."""
    from system.execution.parquet_backend import (
        fraudfeat_module, load_panel_records, public_ids,
    )
    run_dir = Path(run_dir).resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    evaluation_panel = (Path(evaluation_panel) if evaluation_panel
                        else data_dir / "splits" / "evaluation_panel.parquet")
    features_path = run_dir / "features.json"
    payload = _read_json(features_path, [])
    features = payload.get("features") if isinstance(payload, dict) else payload
    if not isinstance(features, list) or not features:
        raise ValueError(f"No features found in {features_path}")
    selected = features[:limit] if limit > 0 else features
    evaluation_features = fraudfeat_module("pool").prepare_features(features)
    evaluation_selected = evaluation_features[:limit] if limit > 0 else evaluation_features
    summary_run = _read_json(run_dir / "run_summary.json", {})
    offline_panel = Path(
        summary_run.get("offline_panel") or data_dir / "splits" / "offline_panel.parquet"
    ).expanduser()
    if not offline_panel.is_file():
        raise FileNotFoundError(f"Offline evaluation panel not found: {offline_panel}")
    targets_value = summary_run.get("targets_path")
    if targets_value:
        targets_path = Path(targets_value).expanduser()
    else:
        seed = summary_run.get("random_seed")
        if seed is None:
            raise ValueError(
                "run_summary.json must record targets_path or random_seed for public evaluation")
        targets_path = data_dir / "splits" / f"target_streams_s{seed}.json"
    if not targets_path.is_file():
        raise FileNotFoundError(f"Discovery target file not found: {targets_path}")
    discovery_ids = set(public_ids(targets_path))
    if not discovery_ids:
        raise ValueError(f"Discovery target file is empty: {targets_path}")
    if not evaluation_panel.is_file():
        raise FileNotFoundError(f"Final evaluation panel not found: {evaluation_panel}")
    offline_ids = _public_panel_ids(offline_panel)
    final_ids = _public_panel_ids(evaluation_panel)
    if not offline_ids:
        raise ValueError(f"Offline evaluation panel is empty: {offline_panel}")
    if not final_ids:
        raise ValueError(f"Final evaluation panel is empty: {evaluation_panel}")
    overlap = offline_ids & final_ids
    if overlap:
        raise ValueError(f"D_off and D overlap on {len(overlap)} accounts")
    overlap = discovery_ids & offline_ids
    if overlap:
        raise ValueError(f"Discovery targets and D_off overlap on {len(overlap)} accounts")
    if discovery_ids & final_ids:
        raise ValueError("Final evaluation panel overlaps discovery targets")
    records = load_panel_records(data_dir, evaluation_panel,
                                 exclude=sorted(discovery_ids))
    eval_ids = [row["target_id"] for row in records]
    labels = {row["target_id"]: int(row["label"]) for row in records}
    panel_sha256 = hashlib.sha256(json.dumps(
        [(uid, labels[uid]) for uid in eval_ids], separators=(",", ":")).encode("utf-8")).hexdigest()
    effectiveness = fraudfeat_module("effectiveness")
    effectiveness_path = run_dir / "effectiveness.jsonl"
    summary_path = run_dir / "effectiveness_summary.json"
    features_sha256 = _features_sha256(evaluation_selected)
    rows = None
    if not force and effectiveness_path.is_file() and summary_path.is_file():
        cached = [json.loads(line) for line in effectiveness_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        cached_summary = _read_json(summary_path, {})
        expected_keys = [str(feature.get("feature_name") or f"feature_{index}")
                         + f"#{int(feature.get('pool_index', index))}"
                         for index, feature in enumerate(evaluation_selected)]
        if (cached_summary.get("panel_sha256") == panel_sha256
                and cached_summary.get("features_sha256") == features_sha256
                and len(cached) == len(evaluation_selected)
                and [str(row.get("entry_key")) for row in cached] == expected_keys):
            rows = cached
    if rows is None:
        store = fraudfeat_module("graph_store").GraphStore(data_dir, load_labels=False)
        rows = effectiveness.evaluate_all(store, evaluation_selected, eval_ids, labels=labels,
                                          amount_unit="fen")
    by_key = {str(row.get("entry_key")): row for row in rows}
    for index, feature in enumerate(selected):
        pool_index = int(feature.get("pool_index", index))
        key = f"{feature.get('feature_name') or f'feature_{index}'}#{pool_index}"
        result = by_key.get(key) or rows[index]
        feature["eval_iv"] = result.get("iv") if result.get("valid") else None
        feature["evaluation"] = {
            key_name: result.get(key_name) for key_name in
            ("valid", "exec_ok", "non_empty", "n_ok", "n_error", "n_zero_hit",
             "coverage", "n_bins", "missing_rate")
        }
        feature["admission_index"] = int(feature.get("admission_index", index))
    if isinstance(payload, dict):
        payload["features"] = features
        _write_json(features_path, payload)
    else:
        _write_json(features_path, features)
    with effectiveness_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    eval_summary = effectiveness.summarize(rows, pool_limit=limit)
    eval_summary.update({
        "features_in": len(features), "eval_size": len(eval_ids),
        "data_dir": str(data_dir), "evaluation_panel": str(evaluation_panel),
        "offline_panel": str(offline_panel), "targets_path": str(targets_path),
        "amount_unit": "fen", "id_list_limit": 300,
        "discovery_targets_excluded": len(discovery_ids),
        "panel_sha256": panel_sha256, "features_sha256": features_sha256,
    })
    _write_json(summary_path, eval_summary)
    report = effectiveness.render_report(
        rows, eval_summary, data_dir=str(data_dir), features_path=str(features_path),
        eval_size=len(eval_ids), excluded=len(discovery_ids), amount_unit="fen")
    (run_dir / "effectiveness_report.txt").write_text(report, encoding="utf-8")
    diversity = fraudfeat_module("diversity")
    schema_index = diversity.SchemaIndex.load(data_dir)
    iv = {str(row["entry_key"]): row["iv"] for row in rows
          if row.get("valid") and row.get("iv") is not None}
    diversity_summary = diversity.summarize(features, schema_index, pool_limit=limit, iv=iv)
    _write_json(run_dir / "diversity_summary.json", diversity_summary)
    (run_dir / "diversity_report.txt").write_text(
        diversity.render_report(diversity_summary, features_path=str(features_path)),
        encoding="utf-8")
    _write_json(run_dir / "final_eval.json", {
        "backend": "fraudfeat", "evaluation_panel": str(evaluation_panel),
        "amount_unit": "fen", "id_list_limit": 300,
        "features": {str(row["entry_key"]): {"eval_iv": row.get("iv"),
                                             "valid": row.get("valid")}
                     for row in rows},
    })
    return {"run_dir": str(run_dir), "evaluated": eval_summary["valid"],
            "failed": eval_summary["invalid"],
            "pool_size": len(selected), "eval_size": len(eval_ids),
            "backend": "fraudfeat"}


def evaluate_run(run_dir: str | Path, limit: int = 100, force: bool = False) -> dict:
    """Compute legacy internal-selection IV, not benchmark-final effectiveness.

    Benchmark-comparable reports must use :func:`evaluate_public_run`, which applies
    the shared Gremlin execution and validity gates on a released fixed panel.
    """
    run_dir = Path(run_dir).resolve()
    features_path = run_dir / "features.json"
    payload = _read_json(features_path, [])
    features = payload.get("features") if isinstance(payload, dict) else payload
    if not isinstance(features, list) or not features:
        raise ValueError(f"No features found in {features_path}")
    base_cfg = dict(load_config("iv.yaml"))
    final_cfg = dict(base_cfg)
    final_cfg.update(base_cfg.get("final_eval") or {})
    final_cfg["results_dir"] = str(run_dir / "final_iv_values")
    final_cfg["use_run_env"] = False
    offline_path = Path(base_cfg.get("samples_path") or "")
    final_path = Path(final_cfg.get("samples_path") or "")
    if not offline_path.is_absolute():
        offline_path = PROJECT_ROOT / offline_path
    if not final_path.is_absolute():
        final_path = PROJECT_ROOT / final_path
    offline_rows = _read_json(offline_path, [])
    final_rows = _read_json(final_path, [])
    offline_ids = {str(row.get("target_id")) for row in offline_rows if isinstance(row, dict)}
    final_ids = {str(row.get("target_id")) for row in final_rows if isinstance(row, dict)}
    overlap = offline_ids & final_ids
    if overlap:
        raise ValueError(f"D_off and D overlap on {len(overlap)} accounts")
    selected = features[:limit] if limit > 0 else features
    checkpoint_path = run_dir / "final_eval.json"
    checkpoint = _read_json(checkpoint_path, {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    completed = 0
    failed = 0
    for index, row in enumerate(selected):
        feature_name = str(row.get("feature_name") or f"feature_{index}")
        if not force and row.get("eval_iv") is not None:
            continue
        from system.orchestrator.loop import _candidate_from_export
        candidate = _candidate_from_export(row)
        value = compute_real_iv(candidate, cfg=final_cfg)
        row["eval_iv"] = value
        row["admission_index"] = int(row.get("admission_index", index))
        checkpoint[feature_name] = {"admission_index": index, "eval_iv": value}
        if value is None:
            failed += 1
        else:
            completed += 1
        _write_json(checkpoint_path, checkpoint)
        if isinstance(payload, dict):
            payload["features"] = features
            _write_json(features_path, payload)
        else:
            _write_json(features_path, features)
    return {"run_dir": str(run_dir), "evaluated": completed, "failed": failed,
            "pool_size": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data", default=None, metavar="DIR",
                        help="Released FraudFeat data directory for shared public evaluation")
    parser.add_argument("--evaluation-panel", default=None,
                        help="Fixed final panel (default <data>/splits/evaluation_panel.parquet)")
    args = parser.parse_args()
    if args.data:
        result = evaluate_public_run(args.run_dir, args.data, args.evaluation_panel,
                                     limit=args.limit, force=args.force)
    else:
        result = evaluate_run(args.run_dir, limit=args.limit, force=args.force)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
