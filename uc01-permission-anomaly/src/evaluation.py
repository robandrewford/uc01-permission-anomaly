"""Evaluation metrics for UC-01 Permission Anomaly Detection.

Implements metrics per spec:
- Primary: precision_at_100 (target 0.7)
- Secondary: recall_at_threshold (target 0.85), auc_roc (target 0.9)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from typing import Dict, Tuple, Optional


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute precision when reviewing top-k anomalies.
    
    Per spec: "SOC analysts can review ~100 alerts/day; 70%+ actionable rate"
    
    Args:
        y_true: Ground truth labels (1=anomaly, 0=normal)
        scores: Anomaly scores (higher = more anomalous)
        k: Number of top-scoring samples to evaluate
    
    Returns:
        Precision in top-k samples
    """
    # Get indices of top-k scores
    top_k_idx = np.argsort(scores)[-k:]
    
    # Calculate precision
    top_k_true = y_true[top_k_idx]
    return np.mean(top_k_true)


def recall_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """Compute recall at a given score threshold.
    
    Per spec: Target 0.85 recall
    
    Args:
        y_true: Ground truth labels
        scores: Anomaly scores
        threshold: Score threshold for classifying as anomaly
    
    Returns:
        Recall at threshold
    """
    y_pred = (scores >= threshold).astype(int)
    return recall_score(y_true, y_pred, zero_division=0)


def false_positive_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> float:
    """Compute false positive rate at threshold.
    
    Per spec: Target <= 0.05 (max 5% of normal users flagged)
    
    Args:
        y_true: Ground truth labels
        scores: Anomaly scores
        threshold: Score threshold
    
    Returns:
        False positive rate
    """
    y_pred = (scores >= threshold).astype(int)
    
    # FPR = FP / (FP + TN) = FP / total_negatives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def calibrate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.95
) -> float:
    """Calibrate threshold to achieve target recall.
    
    Per spec: "calibrate threshold at 95% recall for known anomaly patterns"
    
    Args:
        y_true: Ground truth labels
        scores: Anomaly scores
        target_recall: Target recall to achieve
    
    Returns:
        Threshold that achieves target recall
    """
    # Sort scores descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_true = y_true[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Find threshold that achieves target recall
    total_positives = np.sum(y_true)
    if total_positives == 0:
        return np.median(scores)
    
    cumsum_positives = np.cumsum(sorted_true)
    recall_at_each = cumsum_positives / total_positives
    
    # Find first index where recall >= target
    idx = np.searchsorted(recall_at_each, target_recall)
    if idx >= len(sorted_scores):
        idx = len(sorted_scores) - 1
    
    return sorted_scores[idx]


def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 50.0
) -> Dict[str, float]:
    """Compute all evaluation metrics from spec.
    
    Args:
        y_true: Ground truth labels
        scores: Anomaly scores (0-100 scale)
        threshold: Score threshold for anomaly classification
    
    Returns:
        Dictionary of all metrics
    """
    y_pred = (scores >= threshold).astype(int)
    
    # Primary metric
    p_at_100 = precision_at_k(y_true, scores, k=100)
    p_at_500 = precision_at_k(y_true, scores, k=500)
    
    # Secondary metrics
    recall = recall_at_threshold(y_true, scores, threshold)
    fpr = false_positive_rate(y_true, scores, threshold)
    
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5  # If only one class present
    
    # Additional metrics
    anomaly_rate = np.mean(y_pred)
    mean_score = np.mean(scores)
    
    return {
        'precision_at_100': p_at_100,
        'precision_at_500': p_at_500,
        'recall_at_threshold': recall,
        'auc_roc': auc,
        'false_positive_rate': fpr,
        'anomaly_rate': anomaly_rate,
        'mean_anomaly_score': mean_score,
        'threshold': threshold
    }


def evaluate_against_targets(metrics: Dict[str, float]) -> Dict[str, bool]:
    """Check if metrics meet spec targets.
    
    Targets from spec:
    - precision_at_100 >= 0.7
    - recall_at_threshold >= 0.85
    - auc_roc >= 0.9
    - false_positive_rate <= 0.05
    """
    targets = {
        'precision_at_100': ('>=', 0.7),
        'recall_at_threshold': ('>=', 0.85),
        'auc_roc': ('>=', 0.9),
        'false_positive_rate': ('<=', 0.05)
    }
    
    results = {}
    for metric, (op, target) in targets.items():
        if metric in metrics:
            if op == '>=':
                results[metric] = metrics[metric] >= target
            else:
                results[metric] = metrics[metric] <= target
    
    return results


def format_metrics_report(
    metrics: Dict[str, float],
    targets_met: Dict[str, bool] = None
) -> str:
    """Format metrics as a readable report."""
    lines = ["=" * 50, "Evaluation Metrics Report", "=" * 50]
    
    for name, value in metrics.items():
        status = ""
        if targets_met and name in targets_met:
            status = " ✅" if targets_met[name] else " ❌"
        
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}{status}")
        else:
            lines.append(f"  {name}: {value}{status}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def save_metrics_report(
    metrics: Dict[str, float],
    output_dir: str,
    run_info: Dict = None
) -> None:
    """Save metrics report as JSON and Markdown.
    
    Args:
        metrics: Dictionary of metric values
        output_dir: Directory to save reports
        run_info: Optional dict with mlflow_run_id, timestamp, etc.
    """
    import json
    from pathlib import Path
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate against targets
    targets_met = evaluate_against_targets(metrics)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Build full report data
    report_data = {
        "metrics": convert_to_native(metrics),
        "targets_met": convert_to_native(targets_met),
        "generated_at": datetime.now().isoformat(),
    }
    if run_info:
        report_data["run_info"] = convert_to_native(run_info)
    
    # Save JSON
    json_path = output_path / "metrics_report.json"
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Save Markdown
    md_path = output_path / "metrics_report.md"
    md_content = _generate_markdown_report(metrics, targets_met, run_info)
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"  Saved metrics_report.json -> {json_path}")
    print(f"  Saved metrics_report.md -> {md_path}")


def _generate_markdown_report(
    metrics: Dict[str, float],
    targets_met: Dict[str, bool],
    run_info: Dict = None
) -> str:
    """Generate markdown metrics report."""
    from datetime import datetime
    
    lines = [
        "# Model Evaluation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    
    if run_info:
        lines.extend([
            f"**MLflow Run ID**: `{run_info.get('mlflow_run_id', 'N/A')}`",
            f"**Model Path**: `{run_info.get('model_path', 'N/A')}`",
        ])
    
    lines.extend([
        "",
        "## Metrics vs Spec Targets",
        "",
        "| Metric | Value | Target | Status |",
        "|--------|-------|--------|--------|",
    ])
    
    # Spec targets
    targets = {
        'precision_at_100': ('≥ 0.70', 0.7, '>='),
        'recall_at_threshold': ('≥ 0.85', 0.85, '>='),
        'auc_roc': ('≥ 0.90', 0.9, '>='),
        'false_positive_rate': ('≤ 0.05', 0.05, '<='),
    }
    
    for metric_name, (target_str, _, _) in targets.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            met = targets_met.get(metric_name, False)
            status = "✅" if met else "❌"
            lines.append(f"| {metric_name} | {value:.4f} | {target_str} | {status} |")
    
    lines.extend([
        "",
        "## All Metrics",
        "",
    ])
    
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- **{name}**: {value:.4f}")
        else:
            lines.append(f"- **{name}**: {value}")
    
    return "\n".join(lines)
