"""Command for detecting drowsiness and intoxication from images."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer

from sober_scan.config import MODEL_DIR
from sober_scan.feature_extraction import (
    calculate_eye_aspect_ratio,
    detect_face_and_landmarks,
    extract_skin_redness,
)

# Import intoxication detection models
from sober_scan.models.cnn import IntoxicationDetector as CNNDetector
from sober_scan.models.knn import DrowsinessDetector as KNNDetector
from sober_scan.models.nb import DrowsinessDetector as NBDetector
from sober_scan.models.rf import DrowsinessDetector as RFDetector

# Import drowsiness detection models
from sober_scan.models.svm import DrowsinessDetector as SVMDetector
from sober_scan.utils import (
    draw_drowsiness_result,
    draw_intoxication_result,
    draw_landmarks,
    load_image,
    logger,
    save_image,
    setup_logger,
)


class ModelType(str, Enum):
    """Supported model types for detection."""

    SVM = "svm"
    RANDOM_FOREST = "rf"
    KNN = "knn"
    NAIVE_BAYES = "nb"
    CNN = "cnn"
    GNN = "gnn"


class DetectionType(str, Enum):
    """Types of detection to perform."""

    DROWSINESS = "drowsiness"
    INTOXICATION = "intoxication"


def detect_image_command(
    image_path: Path = typer.Argument(..., help="Path to the input image", exists=True),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the output image with results"),
    detection_type: DetectionType = typer.Option(DetectionType.DROWSINESS, "--type", "-t", help="Type of detection to perform"),
    model_type: str = typer.Option(
        ModelType.SVM.value,
        "--model",
        "-m",
        help="Model type to use for detection (svm, rf, knn, nb, cnn, gnn) or path to a model file",
    ),
    color_mode: bool = typer.Option(False, "--color", help="Use color images for intoxication detection (default is infrared)"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Visualize facial landmarks and features"),
    save_features: bool = typer.Option(False, "--save-features", help="Save extracted features as CSV"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Detect drowsiness or intoxication from a facial image.

    This command processes a facial image to detect signs of drowsiness or intoxication,
    extracting relevant features and classifying the state using the specified model.

    For intoxication detection, infrared images are used by default, but color images
    can be used with the --color flag.

    Examples:

        # Detect drowsiness using default SVM model

        sober-scan detect path/to/your/image.jpg


        # Detect intoxication using CNN model and save the output image

        sober-scan detect path/to/your/image.jpg --type intoxication --model cnn -o path/to/output.jpg


        # Detect drowsiness with visualization enabled

        sober-scan detect path/to/your/image.jpg --type drowsiness -v


        # Detect intoxication using a color image (instead of default infrared)

        sober-scan detect path/to/color_image.jpg --type intoxication --model cnn --color


        # Detect drowsiness using KNN model and save extracted features

        sober-scan detect path/to/your/image.jpg --type drowsiness --model knn --save-features


        # Detect intoxication using a custom model file

        sober-scan detect path/to/your/image.jpg --type intoxication --model /path/to/custom_model.pt -o result.jpg
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

    try:
        typer.echo(f"Detected face at position: {face_rect}")

        # Check if model_type is a file path
        model_path = None
        if os.path.exists(model_type):
            model_path = Path(model_type)
            # Determine model type from file extension
            if model_type.lower().endswith(".pt"):
                model_type_enum = ModelType.CNN
            elif model_type.lower().endswith(".joblib"):
                # Default to SVM for joblib files if we can't determine precisely
                model_type_enum = ModelType.SVM
            else:
                typer.echo("Warning: Unknown model file extension. Attempting to use it anyway.")
                model_type_enum = ModelType.SVM  # Default
        else:
            # Try to convert to enum
            try:
                model_type_enum = ModelType(model_type.lower())
            except ValueError:
                typer.echo(f"Error: Invalid model type or file path: {model_type}")
                typer.echo(f"Model type must be one of: {', '.join([t.value for t in ModelType])}")
                typer.echo("Or it must be a valid path to a model file.")
                raise typer.Exit(code=1)

        # Handle different detection types
        if detection_type == DetectionType.DROWSINESS:
            # Extract EAR and MAR features specifically for drowsiness detection
            ear = calculate_eye_aspect_ratio(landmarks)

            # Mouth aspect ratio: calculate using mouth landmarks (48-68)
            if landmarks is not None and len(landmarks) >= 68:
                # Horizontal distance
                mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
                # Vertical distance
                mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
                mar = mouth_height / mouth_width if mouth_width > 0 else 0
            else:
                mar = 0

            features = np.array([ear, mar])

            typer.echo(f"Eye Aspect Ratio (EAR): {ear:.4f}")
            typer.echo(f"Mouth Aspect Ratio (MAR): {mar:.4f}")

            # Initialize appropriate model based on type
            if model_type_enum == ModelType.SVM:
                model = SVMDetector()
                if not model_path:
                    model_filename = f"drowsiness_{model_type_enum.value}.joblib"
                    model_path = MODEL_DIR / model_filename
            elif model_type_enum == ModelType.RANDOM_FOREST:
                model = RFDetector()
                if not model_path:
                    model_filename = f"drowsiness_{model_type_enum.value}.joblib"
                    model_path = MODEL_DIR / model_filename
            elif model_type_enum == ModelType.KNN:
                model = KNNDetector()
                if not model_path:
                    model_filename = f"drowsiness_{model_type_enum.value}.joblib"
                    model_path = MODEL_DIR / model_filename
            elif model_type_enum == ModelType.NAIVE_BAYES:
                model = NBDetector()
                if not model_path:
                    model_filename = f"drowsiness_{model_type_enum.value}.joblib"
                    model_path = MODEL_DIR / model_filename
            elif model_type_enum in [ModelType.CNN, ModelType.GNN]:
                if not model_path:
                    typer.echo(f"Warning: {model_type_enum.value.upper()} model not implemented for drowsiness detection yet.")
                    raise typer.Exit(code=1)
                else:
                    # For custom model paths with CNN, try to use it
                    typer.echo(f"Attempting to use custom CNN model at: {model_path}")
                    model = CNNDetector()
            else:
                typer.echo(f"Error: Unsupported model type: {model_type}")
                raise typer.Exit(code=1)

            # Try to load the model from specified path
            model_loaded = False

            if model_path.exists():
                try:
                    model.load(model_path)
                    typer.echo(f"Loaded model from {model_path}")
                    model_loaded = True
                except Exception as e:
                    typer.echo(f"Error loading model: {e}")
                    typer.echo("Using default threshold-based detection")
            else:
                typer.echo(f"No trained model found at {model_path}")
                typer.echo("Using default threshold-based detection")

            # Make prediction
            prediction, confidence = model.predict(features)

            # For untrained/fallback models - only use this when model is not loaded
            if not model_loaded:
                # Use simple threshold-based detection as fallback
                ear_threshold = 0.25
                mar_threshold = 0.4  # Higher MAR can indicate drowsiness (mouth open)

                # Combined drowsiness score using both EAR and MAR
                # Lower EAR (eyes closed) and higher MAR (mouth open) indicate drowsiness
                drowsiness_score = (1 - (ear / ear_threshold)) + (mar / mar_threshold)
                drowsiness_score = drowsiness_score / 2  # Normalize to 0-1 range

                if drowsiness_score > 0.5:
                    prediction = "DROWSY"
                else:
                    prediction = "ALERT"

                # Use a default confidence when no model is available
                confidence = 0.0

            # Print results
            typer.echo(f"Drowsiness Detection Result: {prediction} (confidence: {confidence:.2f})")

            # Visualize and save results if requested
            if visualize or output_path:
                visualization = image.copy()

                # Draw landmarks if available
                if landmarks is not None and visualize:
                    visualization = draw_landmarks(visualization, landmarks)

                # Draw drowsiness result text
                visualization = draw_drowsiness_result(visualization, prediction, ear, confidence, mar=mar, face_rect=face_rect)

        elif detection_type == DetectionType.INTOXICATION:
            # Extract skin redness features for intoxication detection
            redness_metrics = extract_skin_redness(image, face_rect, landmarks)
            typer.echo(f"Face Redness: {redness_metrics['face_redness']:.4f}")
            if "forehead_redness" in redness_metrics:
                typer.echo(f"Forehead Redness: {redness_metrics['forehead_redness']:.4f}")
            if "cheeks_redness" in redness_metrics:
                typer.echo(f"Cheeks Redness: {redness_metrics['cheeks_redness']:.4f}")

            # Use CNN for intoxication detection
            if model_type_enum == ModelType.CNN:
                model = CNNDetector(infrared_mode=not color_mode)
                if not model_path:
                    model_filename = f"intoxication_{model_type_enum.value}.pt"
                    model_path = MODEL_DIR / model_filename

                # Try to load the model
                model_loaded = False

                try:
                    if model_path.exists():
                        model.load(model_path)
                        typer.echo(f"Loaded CNN model from {model_path}")
                        model_loaded = True
                    else:
                        typer.echo(f"No trained CNN model found at {model_path}")
                except Exception as e:
                    typer.echo(f"Error loading CNN model: {e}")

                # Face crop for CNN model
                if face_rect is not None:
                    x1, y1, x2, y2 = face_rect
                    # Add some margin
                    h, w = image.shape[:2]
                    margin_x = int((x2 - x1) * 0.2)
                    margin_y = int((y2 - y1) * 0.2)
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)
                    face_img = image[y1:y2, x1:x2]

                    # Use the model for prediction
                    if model_loaded:
                        # The model's predict method should return the prediction and probability directly
                        prediction, confidence = model.predict(face_img)
                    else:
                        # Simple fallback based on multiple redness metrics
                        # Calculate weighted redness score from multiple facial regions
                        face_redness = redness_metrics["face_redness"]
                        forehead_redness = redness_metrics.get("forehead_redness", face_redness)
                        cheeks_redness = redness_metrics.get("cheeks_redness", face_redness)

                        # Weight the different regions (cheeks show redness more prominently)
                        weighted_redness = face_redness * 0.3 + forehead_redness * 0.3 + cheeks_redness * 0.4

                        if weighted_redness > 0.5:
                            prediction = "INTOXICATED"
                        else:
                            prediction = "SOBER"

                            # Use a default confidence when no model is available
                            confidence = 0.0

                    typer.echo(f"Intoxication Detection Result: {prediction} (confidence: {confidence:.2f})")

                    # Visualize and save results
                    if visualize or output_path:
                        visualization = image.copy()

                        # Draw landmarks if available
                        if landmarks is not None and visualize:
                            visualization = draw_landmarks(visualization, landmarks)

                        # Draw intoxication result
                        visualization = draw_intoxication_result(
                            visualization, prediction, redness_metrics["face_redness"], confidence, face_rect=face_rect
                        )

            elif model_type_enum == ModelType.GNN:
                typer.echo("Warning: GNN model not implemented for intoxication detection yet.")
                raise typer.Exit(code=1)
            else:
                # If using a custom model file, try to load it with CNN detector
                if model_path and os.path.exists(model_path):
                    model = CNNDetector(infrared_mode=not color_mode)
                    try:
                        model.load(model_path)
                        typer.echo(f"Loaded custom model from {model_path}")

                        # Face crop for model
                        if face_rect is not None:
                            x1, y1, x2, y2 = face_rect
                            # Add some margin
                            h, w = image.shape[:2]
                            margin_x = int((x2 - x1) * 0.2)
                            margin_y = int((y2 - y1) * 0.2)
                            x1 = max(0, x1 - margin_x)
                            y1 = max(0, y1 - margin_y)
                            x2 = min(w, x2 + margin_x)
                            y2 = min(h, y2 + margin_y)
                            face_img = image[y1:y2, x1:x2]

                            prediction, confidence = model.predict(face_img)

                            # Visualize and save results
                            if visualize or output_path:
                                visualization = image.copy()

                                # Draw landmarks if available
                                if landmarks is not None and visualize:
                                    visualization = draw_landmarks(visualization, landmarks)

                                # Draw intoxication result
                                visualization = draw_intoxication_result(
                                    visualization, prediction, redness_metrics["face_redness"], confidence, face_rect=face_rect
                                )

                    except Exception as e:
                        typer.echo(f"Error loading custom model: {e}")
                        raise typer.Exit(code=1)
                else:
                    typer.echo(f"Error: Model type {model_type} not supported for intoxication detection.")
                    typer.echo("Please use --model cnn for intoxication detection or specify a valid model file path.")
                    raise typer.Exit(code=1)
        else:
            typer.echo(f"Error: Unsupported detection type: {detection_type}")
            raise typer.Exit(code=1)

        # Save output image if output_path is provided
        if output_path:
            if save_image(visualization, output_path):
                typer.echo(f"Result saved to {output_path}")
            else:
                typer.echo(f"Error: Failed to save result to {output_path}")

        # Display image if visualize is enabled
        if visualize:
            # Convert from RGB to BGR for OpenCV display
            cv2_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"{detection_type.value.capitalize()} Detection Result", cv2_img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            # Add an additional waitKey to ensure all windows are properly destroyed
            cv2.waitKey(1)

        # Save features if requested
        if save_features:
            features_path = (
                output_path.with_suffix(".csv") if output_path else Path(os.path.splitext(str(image_path))[0] + "_features.csv")
            )

            try:
                with open(features_path, "w") as f:
                    f.write("feature,value\n")

                    if detection_type == DetectionType.DROWSINESS:
                        f.write(f"eye_aspect_ratio,{ear}\n")
                        f.write(f"mouth_aspect_ratio,{mar}\n")
                    elif detection_type == DetectionType.INTOXICATION:
                        for key, value in redness_metrics.items():
                            f.write(f"{key},{value}\n")

                typer.echo(f"Features saved to {features_path}")

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
