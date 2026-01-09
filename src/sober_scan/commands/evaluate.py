"""Command for evaluating trained models and generating reports."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader

from sober_scan.commands.train import (
    DetectionType,
    IntoxicationDataset,
    extract_features_from_folder,
    plot_confusion_matrix,
    plot_roc_curve,
)
from sober_scan.config import MODEL_DIR
from sober_scan.models.cnn import IntoxicationDetector as CNNDetector
from sober_scan.models.knn import DrowsinessDetector as KNNDetector
from sober_scan.models.nb import DrowsinessDetector as NBDetector
from sober_scan.models.rf import DrowsinessDetector as RFDetector
from sober_scan.models.svm import DrowsinessDetector as SVMDetector
from sober_scan.utils import setup_logger

# Create Typer app for evaluation command
eval_app = typer.Typer(
    name="evaluate",
    help="Evaluate trained models and generate reports",
    no_args_is_help=True,
)


@eval_app.command("model")
def evaluate_model_command(
    model_path: Path = typer.Argument(..., help="Path to the trained model file", exists=True),
    data_folder: Path = typer.Argument(..., help="Path to test data folder", exists=True),
    detection_type: DetectionType = typer.Option(DetectionType.DROWSINESS, "--type", "-t", help="Type of detection"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save evaluation plots"),
    save_plots: bool = typer.Option(True, "--save-plots", help="Save confusion matrix and ROC curve"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Evaluate a trained model on test data and generate evaluation plots.
    
    Examples:
        # Evaluate SVM model
        sober-scan evaluate model models/drowsiness_svm.joblib data/test --type drowsiness
        
        # Evaluate CNN model
        sober-scan evaluate model models/intoxication_cnn.pt data/test --type intoxication
        
        # Save plots to custom location
        sober-scan evaluate model models/drowsiness_knn.joblib data/test -o results/
    """
    setup_logger(verbose)
    
    # Determine output directory
    if output_dir is None:
        output_dir = MODEL_DIR / "evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model type from filename
    model_file = model_path.name.lower()
    
    if detection_type == DetectionType.DROWSINESS:
        # Extract features
        typer.echo("Extracting features from test data...")
        features, labels, _ = extract_features_from_folder(data_folder, threshold=0.25)
        
        if len(features) == 0:
            typer.echo("Error: No features extracted from test data")
            raise typer.Exit(code=1)
        
        # Load appropriate model
        if "svm" in model_file:
            model = SVMDetector()
            model_name = "svm"
        elif "knn" in model_file:
            model = KNNDetector()
            model_name = "knn"
        elif "nb" in model_file:
            model = NBDetector()
            model_name = "nb"
        elif "rf" in model_file:
            model = RFDetector()
            model_name = "rf"
        else:
            typer.echo("Could not determine model type from filename")
            raise typer.Exit(code=1)
        
        # Load model
        model.load(model_path)
        typer.echo(f"Loaded {model_name.upper()} model from {model_path}")
        
        # Make predictions
        y_pred = []
        y_proba = []
        for feat in features:
            pred, prob = model.predict(feat)
            y_pred.append(1 if pred == "DROWSY" else 0)
            y_proba.append(prob)
        
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        y_true = labels
        
    else:  # Intoxication
        # Load CNN model
        model = CNNDetector()
        model.load(model_path)
        model_name = "cnn"
        
        typer.echo(f"Loaded CNN model from {model_path}")
        
        # Create dataset
        dataset = IntoxicationDataset(
            data_folder=data_folder,
            image_size=model.image_size,
            infrared_mode=model.infrared_mode,
            use_face_detection=False,
        )
        
        if len(dataset) == 0:
            typer.echo("Error: No images found in test folder")
            raise typer.Exit(code=1)
        
        # Create dataloader
        import torch
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluate
        model.model.eval()
        y_true = []
        y_pred = []
        y_proba = []
        
        with torch.no_grad():
            for inputs, labels_batch in test_loader:
                inputs = inputs.to(device)
                outputs = model.model(inputs).squeeze()
                preds = (outputs > 0.5).float()
                
                y_true.extend(labels_batch.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_proba.extend(outputs.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = None
    
    # Print results
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Evaluation Results for {model_name.upper()}")
    typer.echo("=" * 60)
    typer.echo(f"Test Data: {len(y_true)} samples")
    typer.echo(f"Accuracy: {accuracy:.4f}")
    if auc is not None:
        typer.echo(f"ROC AUC: {auc:.4f}")
    
    # Classification report
    target_names = ["Alert", "Drowsy"] if detection_type == DetectionType.DROWSINESS else ["Sober", "Drunk"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    typer.echo("\nClassification Report:")
    typer.echo(report)
    
    # Save plots
    if save_plots:
        prefix = "drowsiness" if detection_type == DetectionType.DROWSINESS else "intoxication"
        
        # Confusion matrix
        cm_path = output_dir / f"{prefix}_{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)
        typer.echo(f"\n✓ Confusion matrix saved: {cm_path}")
        
        # ROC curve
        if auc is not None:
            roc_path = output_dir / f"{prefix}_{model_name}_roc_curve.png"
            plot_roc_curve(y_true, y_proba, roc_path)
            typer.echo(f"✓ ROC curve saved: {roc_path}")
    
    typer.echo("\nEvaluation complete!")


@eval_app.command("all")
def evaluate_all_models_command(
    data_folder: Path = typer.Argument(..., help="Path to test data folder", exists=True),
    detection_type: DetectionType = typer.Option(DetectionType.DROWSINESS, "--type", "-t", help="Type of detection"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save evaluation plots"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Evaluate all trained models of a given type.
    
    Examples:
        # Evaluate all drowsiness models
        sober-scan evaluate all data/test --type drowsiness
        
        # Evaluate intoxication models
        sober-scan evaluate all data/test --type intoxication -o results/
    """
    setup_logger(verbose)
    
    if output_dir is None:
        output_dir = models_dir / "evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which models to evaluate
    if detection_type == DetectionType.DROWSINESS:
        model_files = [
            ("svm", models_dir / "drowsiness_svm.joblib"),
            ("knn", models_dir / "drowsiness_knn.joblib"),
            ("nb", models_dir / "drowsiness_nb.joblib"),
            ("rf", models_dir / "drowsiness_rf.joblib"),
        ]
    else:
        model_files = [
            ("cnn", models_dir / "intoxication_cnn.pt"),
        ]
    
    # Evaluate each model
    for model_name, model_path in model_files:
        if not model_path.exists():
            typer.echo(f"⚠️  Skipping {model_name}: Model not found at {model_path}")
            continue
        
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Evaluating {model_name.upper()} model...")
        typer.echo(f"{'='*60}")
        
        # Use the single model evaluation
        try:
            from typer.testing import CliRunner
            runner = CliRunner()
            result = runner.invoke(
                eval_app,
                [
                    "model",
                    str(model_path),
                    str(data_folder),
                    "--type",
                    detection_type.value,
                    "--output",
                    str(output_dir),
                ]
            )
            typer.echo(result.output)
        except Exception as e:
            typer.echo(f"Error evaluating {model_name}: {e}")
    
    typer.echo(f"\n✅ All evaluations complete! Plots saved to: {output_dir}")
