"""Stage 5: Evaluate trained fraud model and produce reports."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_params


def _write_confusion_matrix_svg(path: Path, cm: list[list[int]]) -> None:
    """Write a simple SVG confusion matrix visualization."""
    width = 420
    height = 320
    cell = 120
    start_x = 110
    start_y = 70
    labels = [[str(cm[0][0]), str(cm[0][1])], [str(cm[1][0]), str(cm[1][1])]]
    colors = [["#dbeafe", "#bfdbfe"], ["#93c5fd", "#60a5fa"]]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="32" font-family="Arial" font-size="20" fill="#111827">Confusion Matrix - SENTINEL Fraud Detection</text>',
        '<text x="52" y="120" font-family="Arial" font-size="14" fill="#374151" transform="rotate(-90 52 120)">Actual</text>',
        '<text x="215" y="292" font-family="Arial" font-size="14" fill="#374151">Predicted</text>',
    ]

    for row in range(2):
        for col in range(2):
            x = start_x + col * cell
            y = start_y + row * cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{colors[row][col]}" stroke="#1f2937" stroke-width="1"/>')
            svg.append(f'<text x="{x + cell / 2}" y="{y + cell / 2 + 8}" text-anchor="middle" font-family="Arial" font-size="24" fill="#111827">{labels[row][col]}</text>')

    svg.append('<text x="170" y="60" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Predicted 0</text>')
    svg.append('<text x="290" y="60" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Predicted 1</text>')
    svg.append('<text x="80" y="140" text-anchor="end" font-family="Arial" font-size="14" fill="#374151">Actual 0</text>')
    svg.append('<text x="80" y="260" text-anchor="end" font-family="Arial" font-size="14" fill="#374151">Actual 1</text>')
    svg.append('</svg>')

    path.write_text("\n".join(svg), encoding="utf-8")


def _write_roc_curve_svg(path: Path, fpr: list[float], tpr: list[float], auc_roc: float) -> None:
    """Write a simple SVG ROC curve visualization."""
    width = 420
    height = 320
    margin = 45
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def scale_x(value: float) -> float:
        return margin + value * plot_w

    def scale_y(value: float) -> float:
        return height - margin - value * plot_h

    points = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in zip(fpr, tpr))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="32" font-family="Arial" font-size="20" fill="#111827">ROC Curve - SENTINEL Fraud Detection</text>',
        f'<text x="20" y="54" font-family="Arial" font-size="12" fill="#374151">AUC = {auc_roc:.4f}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#9ca3af" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#9ca3af" stroke-width="1"/>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{margin}" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,5"/>',
        '<text x="190" y="306" font-family="Arial" font-size="14" fill="#374151">False Positive Rate</text>',
        '<text x="18" y="170" font-family="Arial" font-size="14" fill="#374151" transform="rotate(-90 18 170)">True Positive Rate</text>',
        '</svg>',
    ]

    path.write_text("\n".join(svg), encoding="utf-8")


def _write_unavailable_roc_curve_svg(path: Path) -> None:
    """Write a placeholder SVG when ROC is undefined for a single-class target."""
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="420" height="320" viewBox="0 0 420 320">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="32" font-family="Arial" font-size="20" fill="#111827">ROC Curve - SENTINEL Fraud Detection</text>',
        '<text x="20" y="70" font-family="Arial" font-size="14" fill="#374151">ROC is undefined because y_test contains a single class.</text>',
        '<line x1="45" y1="275" x2="375" y2="45" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,5"/>',
        '<text x="190" y="306" font-family="Arial" font-size="14" fill="#374151">False Positive Rate</text>',
        '<text x="18" y="170" font-family="Arial" font-size="14" fill="#374151" transform="rotate(-90 18 170)">True Positive Rate</text>',
        '</svg>',
    ]

    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    """Run Stage 5 evaluation flow."""
    params = load_params()
    data_cfg = params.get("data", {})
    cfg = params.get("evaluate", {})

    model_path = Path(cfg.get("model_path", "models/fraud_model.pkl"))
    metrics_path = Path(cfg.get("metrics_path", "metrics/scores.json"))
    reports_dir = Path(cfg.get("reports_dir", "reports/"))

    x_test_path = Path(data_cfg.get("x_test_path", "data/processed/X_test.csv"))
    y_test_path = Path(data_cfg.get("y_test_path", "data/processed/y_test.csv"))

    if not model_path.exists():
        raise FileNotFoundError(f"[Evaluate] Model not found: {model_path}")
    if not x_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("[Evaluate] Missing X_test.csv or y_test.csv in data/processed/")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["is_fraud"].astype(int)

    # Align just in case of schema drift
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    y_test_has_both_classes = y_test.nunique() > 1

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    auc_roc_val = float(roc_auc_score(y_test, y_prob)) if y_test_has_both_classes else 0.0
    avg_prec_val = float(average_precision_score(y_test, y_prob)) if y_test_has_both_classes else 0.0

    # Compute ROC curve data for UI
    fpr_list, tpr_list = [], []
    if y_test_has_both_classes:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
        # Subsample to keep JSON size reasonable
        step = max(1, len(fpr_arr) // 200)
        fpr_list = [round(float(v), 6) for v in fpr_arr[::step]]
        tpr_list = [round(float(v), 6) for v in tpr_arr[::step]]
        # Ensure endpoint is included
        if fpr_list[-1] != round(float(fpr_arr[-1]), 6):
            fpr_list.append(round(float(fpr_arr[-1]), 6))
            tpr_list.append(round(float(tpr_arr[-1]), 6))

    # Feature importance from model
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        for feat, imp in zip(feature_columns, model.feature_importances_):
            feature_importance[feat] = round(float(imp), 6)

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": auc_roc_val,
        "average_precision": avg_prec_val,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "support_neg": tn + fp,
        "support_pos": fn + tp,
        "fpr": fpr_list,
        "tpr": tpr_list,
        "feature_importance": feature_importance,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _write_confusion_matrix_svg(reports_dir / "confusion_matrix.svg", cm.tolist())

    if y_test_has_both_classes:
        _write_roc_curve_svg(reports_dir / "roc_curve.svg", fpr_list, tpr_list, metrics["auc_roc"])
    else:
        _write_unavailable_roc_curve_svg(reports_dir / "roc_curve.svg")

    print("[Evaluate] Classification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"[Evaluate] Metrics saved to {metrics_path}")
    print(f"[Evaluate] Reports saved to {reports_dir}")


if __name__ == "__main__":
    main()
