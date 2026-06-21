#!/usr/bin/env python
"""
Evaluate saved GenoPHI CatBoost models on a merged feature table.

Runs ensemble prediction (median across run_*/best_model.pkl), computes global
binary classification metrics, per-strain hit@k / precision@k, and saves plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from genophi.mmseqs2_clustering import _read_table, _resolve_table_path
from genophi.workflows.prediction_workflow import calculate_median_predictions, predict_interactions

HIT_K_VALUES = (1, 2, 3, 4, 5)
DEFAULT_PHENOTYPE_COLUMN = "interaction"
DEFAULT_STRAIN_COLUMN = "strain"
DEFAULT_PHAGE_COLUMN = "phage"
PROBABILITY_THRESHOLD = 0.5


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_merged_feature_table(path: str) -> pd.DataFrame:
    resolved = _resolve_table_path(path)
    logging.info("Loading merged feature table from %s", resolved)
    df = _read_table(resolved)
    if df.empty:
        raise ValueError(f"Feature table is empty: {resolved}")
    return df


def validate_columns(
    df: pd.DataFrame,
    phenotype_column: str,
    strain_column: str,
    phage_column: str,
) -> None:
    missing = [
        col
        for col in (phenotype_column, strain_column, phage_column)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns in feature table: {missing}")


def prepare_prediction_frame(
    df: pd.DataFrame,
    phenotype_column: str,
    strain_column: str,
    phage_column: str,
    extra_drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split labels from features; drop metadata not used by the model."""
    y_true = df[phenotype_column].copy()
    drop_cols = {phenotype_column, strain_column, phage_column}
    if extra_drop_columns:
        drop_cols.update(extra_drop_columns)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    predict_df = df[[strain_column, phage_column] + feature_cols].copy()
    return predict_df, y_true


def compute_global_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute standard binary classification metrics at a fixed threshold."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    ppv = precision_score(y_true_arr, y_pred_arr, zero_division=0)

    metrics: dict[str, float | int | None] = {
        "n_samples": int(len(y_true_arr)),
        "n_positive_true": int(y_true_arr.sum()),
        "n_negative_true": int((y_true_arr == 0).sum()),
        "n_predicted_positive": int(y_pred_arr.sum()),
        "n_predicted_negative": int((y_pred_arr == 0).sum()),
        "pct_predicted_positive": float(100.0 * y_pred_arr.mean()),
        "pct_true_positive": float(100.0 * y_true_arr.mean()),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(ppv),
        "ppv": float(ppv),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "sensitivity": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "confusion_tp": int(tp),
        "confusion_fp": int(fp),
        "confusion_tn": int(tn),
        "confusion_fn": int(fn),
    }

    if len(np.unique(y_true_arr)) < 2:
        logging.warning("Only one class in ground truth; AUC metrics set to null.")
        metrics["auc_roc"] = None
        metrics["auc_prc"] = None
        metrics["average_precision"] = None
    else:
        metrics["auc_roc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["auc_prc"] = float(average_precision_score(y_true_arr, y_prob_arr))
        metrics["average_precision"] = metrics["auc_prc"]

    return metrics


def compute_per_strain_ranking_metrics(
    eval_df: pd.DataFrame,
    strain_column: str,
    phenotype_column: str,
    confidence_column: str = "Confidence",
    k_values: tuple[int, ...] = HIT_K_VALUES,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute hit@k and precision@k averaged over eligible strains.

    hit@k: fraction of strains (with >=1 true positive) where >=1 true positive
           appears in the top-k phages ranked by predicted probability.

    precision@k: mean over strains with >=k total true positives of
                 (true positives in top-k) / k.
    """
    per_strain_rows: list[dict] = []
    hit_accumulators = {k: [] for k in k_values}
    precision_accumulators = {k: [] for k in k_values}

    for strain_id, group in eval_df.groupby(strain_column):
        group_sorted = group.sort_values(confidence_column, ascending=False)
        total_positives = int(group_sorted[phenotype_column].sum())

        row: dict = {
            strain_column: strain_id,
            "n_pairs": len(group_sorted),
            "n_true_positive": total_positives,
        }

        for k in k_values:
            top_k = group_sorted.head(k)
            positives_in_top_k = int(top_k[phenotype_column].sum())

            hit_value = int(positives_in_top_k >= 1)
            precision_value = positives_in_top_k / k

            row[f"hit@{k}"] = hit_value
            row[f"precision@{k}"] = precision_value

            if total_positives >= 1:
                hit_accumulators[k].append(hit_value)
            if total_positives >= k:
                precision_accumulators[k].append(precision_value)

        per_strain_rows.append(row)

    summary: dict[str, float] = {}
    for k in k_values:
        hit_vals = hit_accumulators[k]
        prec_vals = precision_accumulators[k]
        summary[f"hit@{k}"] = float(np.mean(hit_vals)) if hit_vals else float("nan")
        summary[f"hit@{k}_n_strains"] = len(hit_vals)
        summary[f"precision@{k}"] = float(np.mean(prec_vals)) if prec_vals else float("nan")
        summary[f"precision@{k}_n_strains"] = len(prec_vals)

    per_strain_df = pd.DataFrame(per_strain_rows)
    return summary, per_strain_df


def plot_probability_distribution(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
    threshold: float = PROBABILITY_THRESHOLD,
) -> None:
    """Histogram of predicted probabilities, optionally split by true class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_prob, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(threshold, color="crimson", linestyle="--", label=f"threshold={threshold}")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of predicted probabilities")
    axes[0].legend()

    axes[1].hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="True negative", color="tab:blue")
    axes[1].hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="True positive", color="tab:orange")
    axes[1].axvline(threshold, color="crimson", linestyle="--", label=f"threshold={threshold}")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted probabilities by true label")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Saved probability distribution plot to %s", output_path)


def save_metrics_report(
    output_dir: Path,
    global_metrics: dict,
    ranking_metrics: dict,
) -> None:
    combined = {**global_metrics, **ranking_metrics}
    json_path = output_dir / "evaluation_metrics.json"
    with json_path.open("w") as f:
        json.dump(combined, f, indent=2)

    csv_path = output_dir / "evaluation_metrics.csv"
    pd.DataFrame([combined]).to_csv(csv_path, index=False)
    logging.info("Saved metrics to %s and %s", json_path, csv_path)


def run_evaluation(
    feature_table: str,
    model_dir: str,
    output_dir: str,
    phenotype_column: str = DEFAULT_PHENOTYPE_COLUMN,
    strain_column: str = DEFAULT_STRAIN_COLUMN,
    phage_column: str = DEFAULT_PHAGE_COLUMN,
    threads: int = 4,
    threshold: float = PROBABILITY_THRESHOLD,
    k_values: tuple[int, ...] = HIT_K_VALUES,
) -> pd.DataFrame:
    """Run full evaluation pipeline and return per-pair predictions with labels."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_merged_feature_table(feature_table)
    validate_columns(df, phenotype_column, strain_column, phage_column)

    predict_df, _ = prepare_prediction_frame(
        df, phenotype_column, strain_column, phage_column
    )

    logging.info("Running predictions with models from %s", model_dir)
    all_preds = predict_interactions(
        model_dir,
        predict_df,
        single_strain_mode=False,
        strain_source=strain_column,
        phage_source=phage_column,
        threads=threads,
    )
    all_preds.to_csv(out / "all_run_predictions.csv", index=False)

    median_preds = calculate_median_predictions(
        all_preds,
        single_strain_mode=False,
        strain_source=strain_column,
        phage_source=phage_column,
    )
    median_preds["Final_Prediction"] = (median_preds["Confidence"] >= threshold).astype(int)
    median_preds.to_csv(out / "median_predictions.csv", index=False)

    eval_df = median_preds.merge(
        df[[strain_column, phage_column, phenotype_column]],
        on=[strain_column, phage_column],
        how="inner",
    )
    if len(eval_df) != len(median_preds):
        logging.warning(
            "Merged evaluation rows (%d) differ from predictions (%d); check for duplicate keys.",
            len(eval_df),
            len(median_preds),
        )

    eval_df.to_csv(out / "evaluation_pairs.csv", index=False)

    y_prob = eval_df["Confidence"].to_numpy()
    y_pred = eval_df["Final_Prediction"].to_numpy()
    y_true_arr = eval_df[phenotype_column].to_numpy()

    global_metrics = compute_global_metrics(
        pd.Series(y_true_arr), y_pred, y_prob
    )
    global_metrics["probability_threshold"] = threshold

    ranking_metrics, per_strain_df = compute_per_strain_ranking_metrics(
        eval_df,
        strain_column=strain_column,
        phenotype_column=phenotype_column,
        k_values=k_values,
    )
    per_strain_df.to_csv(out / "per_strain_ranking_metrics.csv", index=False)

    save_metrics_report(out, global_metrics, ranking_metrics)
    plot_probability_distribution(
        y_prob, y_true_arr, out / "predicted_probability_distribution.png", threshold
    )

    _log_metrics_summary(global_metrics, ranking_metrics, k_values)
    return eval_df


def _log_metrics_summary(
    global_metrics: dict,
    ranking_metrics: dict,
    k_values: tuple[int, ...],
) -> None:
    logging.info("=== Global metrics ===")
    for key in (
        "n_samples",
        "pct_true_positive",
        "pct_predicted_positive",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "npv",
        "f1",
        "mcc",
        "auc_roc",
        "auc_prc",
    ):
        if key in global_metrics:
            logging.info("  %s: %s", key, global_metrics[key])

    logging.info("=== Per-strain ranking metrics (averages) ===")
    for k in k_values:
        logging.info(
            "  hit@%d: %.4f (n=%s) | precision@%d: %.4f (n=%s)",
            k,
            ranking_metrics.get(f"hit@{k}", float("nan")),
            ranking_metrics.get(f"hit@{k}_n_strains", 0),
            k,
            ranking_metrics.get(f"precision@{k}", float("nan")),
            ranking_metrics.get(f"precision@{k}_n_strains", 0),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved GenoPHI models on a merged feature table with ground-truth labels. "
            "Computes global classification metrics, hit@k, precision@k per strain, and plots."
        ),
    )
    parser.add_argument(
        "--feature_table",
        "-i",
        required=True,
        help="Merged CSV/parquet with strain, phage, phenotype, and feature columns",
    )
    parser.add_argument(
        "--model_dir",
        "-m",
        required=True,
        help="Directory with run_*/best_model.pkl (e.g. modeling_results/cutoff_10/)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Directory for predictions, metrics, and plots",
    )
    parser.add_argument(
        "--phenotype_column",
        default=DEFAULT_PHENOTYPE_COLUMN,
        help=f"Ground-truth label column (default: {DEFAULT_PHENOTYPE_COLUMN})",
    )
    parser.add_argument(
        "--strain_column",
        default=DEFAULT_STRAIN_COLUMN,
        help=f"Strain identifier column (default: {DEFAULT_STRAIN_COLUMN})",
    )
    parser.add_argument(
        "--phage_column",
        default=DEFAULT_PHAGE_COLUMN,
        help=f"Phage identifier column (default: {DEFAULT_PHAGE_COLUMN})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads for CatBoost prediction (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PROBABILITY_THRESHOLD,
        help=f"Probability threshold for binary prediction (default: {PROBABILITY_THRESHOLD})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        run_evaluation(
            feature_table=args.feature_table,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            phenotype_column=args.phenotype_column,
            strain_column=args.strain_column,
            phage_column=args.phage_column,
            threads=args.threads,
            threshold=args.threshold,
        )
    except Exception as exc:
        logging.error("Evaluation failed: %s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
