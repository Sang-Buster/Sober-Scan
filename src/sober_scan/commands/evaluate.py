"""``sober-scan evaluate ...`` \u2014 honest cross-subject evaluation.

The single ``baseline`` subcommand runs leave-one-subject-out
cross-validation on a ``data/testing_data``-style corpus for one of the
known Tier 1 baselines (majority / handcrafted / imagenet) and prints a
metrics table. This is the *only* evaluation surface that produces
defensible numbers; the older ``train``-time accuracy reports use
leaky splits.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import typer

from sober_scan.corpus import IntoxicationCorpus
from sober_scan.evaluation import LOSOReport, evaluate_baseline
from sober_scan.models.baselines import (
    HandcraftedFeaturesLR,
    ImageNetFrozenLR,
    LOSOTrainedCNN,
    MajorityClassBaseline,
    SiameseDelta,
)

eval_app = typer.Typer(
    name="evaluate",
    help="Honest cross-subject evaluation (leave-one-subject-out CV).",
    no_args_is_help=True,
)


class BaselineName(str, Enum):
    """The baselines registered with ``evaluate baseline``."""

    MAJORITY = "majority"
    HANDCRAFTED = "handcrafted"
    IMAGENET = "imagenet"
    LOSO_CNN = "loso-cnn"
    SIAMESE = "siamese"


_BASELINE_FACTORIES: Dict[BaselineName, Callable] = {
    BaselineName.MAJORITY: MajorityClassBaseline,
    BaselineName.HANDCRAFTED: HandcraftedFeaturesLR,
    BaselineName.IMAGENET: ImageNetFrozenLR,
    BaselineName.LOSO_CNN: LOSOTrainedCNN,
    BaselineName.SIAMESE: SiameseDelta,
}


def _format_report(name: str, report: LOSOReport, threshold: float) -> str:
    auc_str = (
        f"{report.pooled_auc:.4f}" if report.pooled_auc is not None else "N/A"
    )
    lines = [
        f"== LOSO evaluation: {name} ==",
        f"threshold:                BAC >= {threshold}",
        f"folds:                    {len(report.per_fold)}",
        f"pooled accuracy:          {report.pooled_accuracy:.4f}",
        f"pooled balanced accuracy: {report.pooled_balanced_accuracy:.4f}",
        f"pooled ROC-AUC:           {auc_str}",
        "",
        "per-fold:",
        f"  {'subject':<8} {'n':<4} {'acc':<7} {'true':<10} {'pred':<10}",
    ]
    for fold in report.per_fold:
        acc = (fold.y_true == fold.y_pred).mean()
        sob = int((fold.y_true == 0).sum())
        dru = int((fold.y_true == 1).sum())
        sob_pred = int((fold.y_pred == 0).sum())
        dru_pred = int((fold.y_pred == 1).sum())
        lines.append(
            f"  {fold.held_out_subject:<8} {len(fold.y_true):<4} "
            f"{acc:<7.4f} {f'{sob}s/{dru}d':<10} {f'{sob_pred}s/{dru_pred}d':<10}"
        )
    return "\n".join(lines)


def _report_to_dict(report: LOSOReport, threshold: float) -> dict:
    return {
        "threshold": threshold,
        "pooled_accuracy": report.pooled_accuracy,
        "pooled_balanced_accuracy": report.pooled_balanced_accuracy,
        "pooled_auc": report.pooled_auc,
        "per_fold": [
            {
                "held_out_subject": f.held_out_subject,
                "n_test": len(f.y_true),
                "accuracy": float((f.y_true == f.y_pred).mean()),
                "y_true": f.y_true.tolist(),
                "y_pred": f.y_pred.tolist(),
                "y_score": f.y_score.tolist(),
            }
            for f in report.per_fold
        ],
    }


def _format_threshold_sweep(
    name: str,
    rows: List[tuple],
) -> str:
    """Render a compact multi-threshold summary table.

    ``rows`` is a list of ``(threshold, n_drunk, report)`` tuples.
    """
    lines = [
        f"== LOSO evaluation: {name} \u2014 threshold sweep ==",
        "",
        f"  {'threshold':<10} {'n_drunk':<8} {'acc':<8} {'bal_acc':<9} {'AUC':<8}",
        f"  {'-' * 9:<10} {'-' * 7:<8} {'-' * 7:<8} {'-' * 8:<9} {'-' * 7:<8}",
    ]
    for threshold, n_drunk, report in rows:
        auc_str = (
            f"{report.pooled_auc:.4f}" if report.pooled_auc is not None else "N/A"
        )
        lines.append(
            f"  {threshold:<10.3f} {n_drunk:<8d} "
            f"{report.pooled_accuracy:<8.4f} "
            f"{report.pooled_balanced_accuracy:<9.4f} "
            f"{auc_str:<8}"
        )
    return "\n".join(lines)


@eval_app.command("baseline", no_args_is_help=True)
def evaluate_baseline_command(
    name: BaselineName = typer.Argument(
        ...,
        help="Which baseline to run.",
    ),
    data: Path = typer.Option(
        Path("data/testing_data"),
        "--data",
        "-d",
        help="Directory of labeled photos in P{n}_bd/ad_BBB_*.jpg format.",
    ),
    threshold: List[float] = typer.Option(
        [0.05],
        "--threshold",
        "-t",
        help=(
            "BAC threshold for the binary drunk-vs-sober label. "
            "Pass multiple times for a threshold sweep, "
            "e.g. -t 0.03 -t 0.05 -t 0.08 -t 0.10."
        ),
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Optional path to dump per-fold predictions and metrics as JSON.",
    ),
) -> None:
    """Run leave-one-subject-out CV for a baseline and print metrics.

    Single ``--threshold`` prints the full per-fold breakdown. Multiple
    ``--threshold`` values render a compact sweep table instead.
    """
    if not data.exists():
        typer.echo(f"Error: data folder not found: {data}")
        raise typer.Exit(code=1)

    # Backfill ambiguous (timestamp-only) photos from sibling Px.txt session
    # notes before filtering. If no .txt files exist this is a no-op.
    corpus = (
        IntoxicationCorpus.from_folder(data)
        .with_session_notes_backfill(notes_dir=data)
        .with_known_bac()
    )
    if len(corpus) == 0:
        typer.echo(f"Error: no labeled photos found under {data}")
        raise typer.Exit(code=1)

    factory = _BASELINE_FACTORIES[name]
    # Sort thresholds for predictable table ordering and de-dup.
    thresholds = sorted(set(threshold))

    if len(thresholds) == 1:
        only = thresholds[0]
        report = evaluate_baseline(corpus, factory, threshold=only)
        typer.echo(_format_report(name.value, report, only))

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(_report_to_dict(report, only), indent=2)
            )
            typer.echo(f"\nWrote JSON to {output_json}")
        return

    # Multi-threshold sweep.
    rows = []
    json_entries = []
    for t in thresholds:
        labels = corpus.binary_labels(threshold=t)
        n_drunk = sum(labels)
        report = evaluate_baseline(corpus, factory, threshold=t)
        rows.append((t, n_drunk, report))
        json_entries.append(_report_to_dict(report, t))

    typer.echo(_format_threshold_sweep(name.value, rows))

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(json_entries, indent=2))
        typer.echo(f"\nWrote JSON sweep to {output_json}")
