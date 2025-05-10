"""Command for detecting intoxication from images."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import typer

from sober_scan.deep_models import DeepModelHandler
from sober_scan.feature_extraction import detect_face_and_landmarks, extract_features
from sober_scan.traditional_methods import TraditionalModel
from sober_scan.utils import (
    draw_intoxication_result,
    draw_landmarks,
    load_image,
    logger,
    plot_features,
    save_image,
    setup_logger,
)


class ModelType(str, Enum):
    """Supported model types for intoxication detection."""

    TRADITIONAL_SVM = "svm"
    TRADITIONAL_RF = "random_forest"
    DEEP_CNN = "cnn"
    DEEP_GNN = "gnn"


def detect_image_command(
    image_path: Path = typer.Argument(..., help="Path to the input image", exists=True),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save the output image with results"
    ),
    model_type: ModelType = typer.Option(
        ModelType.TRADITIONAL_SVM, "--model", "-m", help="Model type to use for detection"
    ),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Custom path to the model file"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Visualize facial landmarks and features"),
    save_features: bool = typer.Option(False, "--save-features", help="Save extracted features as CSV"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Detect intoxication from a facial image.

    This command processes a facial image to detect signs of alcohol intoxication,
    extracting features like skin redness, eye aspect ratio, and facial geometry.
    It then classifies the intoxication level using the specified model.
    """
    # Setup logger
    setup_logger(verbose)

    # Load image
    logger.info(f"Loading image from {image_path}")
    image = load_image(image_path)

    if image is None:
        typer.echo(f"Error: Failed to load image from {image_path}")
        raise typer.Exit(code=1)

    # Detect face and landmarks
    face_rect, landmarks = detect_face_and_landmarks(image)

    if face_rect is None:
        typer.echo("Error: No face detected in the image")
        raise typer.Exit(code=1)

    # Extract features
    logger.info("Extracting facial features")
    features = extract_features(image)

    if not features:
        typer.echo("Error: Failed to extract facial features")
        raise typer.Exit(code=1)

    # Report on features found
    typer.echo(f"Detected face at position: {face_rect}")
    typer.echo(f"Extracted {len(features)} facial features")

    try:
        # Initialize appropriate model based on type
        if model_type in [ModelType.TRADITIONAL_SVM, ModelType.TRADITIONAL_RF]:
            # Traditional ML approach
            model = TraditionalModel(model_type=model_type.value, model_path=str(model_path) if model_path else None)

            # Check if model is loaded properly
            if model.model is None:
                typer.echo(f"Warning: No trained model found for {model_type}.")
                typer.echo("Using default classification (results may not be accurate).")
                # Provide a fallback prediction - just use some middle values to indicate uncertainty
                bac_level = "N/A"
                confidence = 0
            else:
                bac_level, confidence = model.predict(features)

        elif model_type == ModelType.DEEP_CNN:
            # Deep CNN approach
            model = DeepModelHandler(model_type="cnn", model_path=str(model_path) if model_path else None)

            # Check if model is loaded properly
            if model.model is None:
                typer.echo(f"Warning: No trained model found for {model_type}.")
                typer.echo("Using default classification (results may not be accurate).")
                # Provide a fallback prediction
                bac_level = "N/A"
                confidence = 0
            else:
                bac_level, confidence = model.predict(image=image)

        elif model_type == ModelType.DEEP_GNN:
            # Graph NN approach (needs landmarks)
            if landmarks is None:
                typer.echo("Error: No facial landmarks detected for GNN model")
                raise typer.Exit(code=1)

            model = DeepModelHandler(model_type="gnn", model_path=str(model_path) if model_path else None)

            # Check if model is loaded properly
            if model.model is None:
                typer.echo(f"Warning: No trained model found for {model_type}.")
                typer.echo("Using default classification (results may not be accurate).")
                # Provide a fallback prediction
                bac_level = "N/A"
                confidence = 0
            else:
                bac_level, confidence = model.predict(landmarks=landmarks)

        else:
            typer.echo(f"Error: Unsupported model type: {model_type}")
            raise typer.Exit(code=1)

        # Print results
        result_str = f"Detected intoxication level: {bac_level.name} (confidence: {confidence:.2f})"
        typer.echo(result_str)

        # Visualize and save results if requested
        if visualize or output_path:
            visualization = image.copy()

            # Draw landmarks if available
            if landmarks is not None and visualize:
                visualization = draw_landmarks(visualization, landmarks)

            # Draw intoxication result
            visualization = draw_intoxication_result(visualization, bac_level, confidence)

            # Save output image
            if output_path:
                if save_image(visualization, output_path):
                    typer.echo(f"Result saved to {output_path}")
                else:
                    typer.echo(f"Error: Failed to save result to {output_path}")

            # Display image if visualize is enabled
            if visualize:
                # Convert from RGB to BGR for OpenCV display
                cv2_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                cv2.imshow("Intoxication Detection Result", cv2_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Save features if requested
        if save_features:
            features_path = (
                output_path.with_suffix(".csv")
                if output_path
                else Path(os.path.splitext(str(image_path))[0] + "_features.csv")
            )

            # Convert features to CSV format
            feature_str = ",".join([f"{k},{v}" for k, v in features.items()])

            try:
                with open(features_path, "w") as f:
                    f.write("feature,value\n")
                    f.write(feature_str)

                typer.echo(f"Features saved to {features_path}")

                # Also save feature plot
                if visualize:
                    plot_path = features_path.with_suffix(".png")
                    plot_features(features, plot_path)
                    typer.echo(f"Feature plot saved to {plot_path}")

            except Exception as e:
                typer.echo(f"Error saving features: {e}")

    except Exception as e:
        typer.echo(f"Error during detection: {e}")
        if verbose:
            # In verbose mode, print the full exception details
            import traceback

            traceback.print_exc()
        else:
            typer.echo("Run with --verbose for more details.")
        raise typer.Exit(code=1)

    # Return success
    return
