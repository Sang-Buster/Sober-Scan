"""``sober-scan detect`` \u2014 single-image drowsiness / intoxication detection.

Drowsiness uses a direct EAR threshold rule (no model file); the
former "drowsiness model" path was a learning-on-its-own-label
tautology and has been removed. Intoxication uses the existing CNN,
with the caveat that its honest cross-subject performance is poor (see
``models/MODEL_EVALUATION_REPORT.md``).
"""

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
from sober_scan.models.cnn import IntoxicationDetector as CNNDetector
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
    """Models supported by ``sober-scan detect``.

    Drowsiness is no longer model-backed; intoxication uses the CNN.
    """

    CNN = "cnn"


class DetectionType(str, Enum):
    """Types of detection to perform."""

    DROWSINESS = "drowsiness"
    INTOXICATION = "intoxication"


_EAR_DROWSY_THRESHOLD = 0.2
_MAR_DROWSY_THRESHOLD = 0.4


def detect_image_command(
    image_path: Path = typer.Argument(..., help="Path to the input image", exists=True),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save the output image with results"
    ),
    detection_type: DetectionType = typer.Option(
        DetectionType.DROWSINESS, "--type", "-t", help="Type of detection to perform"
    ),
    model_type: str = typer.Option(
        ModelType.CNN.value,
        "--model",
        "-m",
        help="Model type for intoxication (cnn only) or path to a model file. Ignored for drowsiness.",
    ),
    color_mode: bool = typer.Option(
        False, "--color", help="Use color (RGB) images for intoxication CNN (default: grayscale)."
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Show / draw facial landmarks and results."
    ),
    save_features: bool = typer.Option(
        False, "--save-features", help="Save extracted features as CSV."
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output."),
) -> None:
    """Detect drowsiness (EAR rule) or intoxication (CNN) from a facial image.

    Examples:

        sober-scan detect path/to/image.jpg                          # drowsiness via EAR rule
        sober-scan detect path/to/image.jpg --type intoxication      # intoxication via CNN
        sober-scan detect img.jpg --type intoxication -o out.jpg     # save result
    """
    setup_logger(verbose)

    logger.info(f"Loading image from {image_path}")
    image = load_image(image_path)
    if image is None:
        typer.echo(f"Error: Failed to load image from {image_path}")
        raise typer.Exit(code=1)

    face_rect, landmarks = detect_face_and_landmarks(image)
    if face_rect is None:
        typer.echo("Error: No face detected in the image")
        raise typer.Exit(code=1)

    typer.echo(f"Detected face at position: {face_rect}")

    if detection_type == DetectionType.DROWSINESS:
        _run_drowsiness(image, landmarks, face_rect, visualize, output_path, save_features, image_path)
    elif detection_type == DetectionType.INTOXICATION:
        _run_intoxication(image, landmarks, face_rect, model_type, color_mode, visualize, output_path, save_features, image_path)
    else:
        typer.echo(f"Error: Unsupported detection type: {detection_type}")
        raise typer.Exit(code=1)


def _run_drowsiness(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    face_rect,
    visualize: bool,
    output_path: Optional[Path],
    save_features: bool,
    image_path: Path,
) -> None:
    """Apply the EAR-rule directly. No model file involved."""
    ear = calculate_eye_aspect_ratio(landmarks)

    if landmarks is not None and len(landmarks) >= 68:
        mouth_width = float(np.linalg.norm(landmarks[48] - landmarks[54]))
        mouth_height = float(np.linalg.norm(landmarks[51] - landmarks[57]))
        mar = mouth_height / mouth_width if mouth_width > 0 else 0.0
    else:
        mar = 0.0

    typer.echo(f"Eye Aspect Ratio (EAR): {ear:.4f}  (rule: drowsy iff EAR < {_EAR_DROWSY_THRESHOLD})")
    typer.echo(f"Mouth Aspect Ratio (MAR): {mar:.4f}  (rule: yawn iff MAR > {_MAR_DROWSY_THRESHOLD})")

    is_drowsy = ear < _EAR_DROWSY_THRESHOLD or mar > _MAR_DROWSY_THRESHOLD
    prediction = "DROWSY" if is_drowsy else "ALERT"
    # Confidence is a transparent function of how far the rule fired; no
    # probabilistic interpretation, just a UX number.
    confidence = (
        max(
            (_EAR_DROWSY_THRESHOLD - ear) / _EAR_DROWSY_THRESHOLD if ear < _EAR_DROWSY_THRESHOLD else 0.0,
            (mar - _MAR_DROWSY_THRESHOLD) / _MAR_DROWSY_THRESHOLD if mar > _MAR_DROWSY_THRESHOLD else 0.0,
            1.0 - (ear / _EAR_DROWSY_THRESHOLD) if not is_drowsy else 0.0,
        )
        if is_drowsy
        else 1.0 - min(ear / _EAR_DROWSY_THRESHOLD, 1.0)
    )

    typer.echo(f"Drowsiness Detection Result: {prediction} (confidence: {confidence:.2f})")

    visualization = None
    if visualize or output_path:
        visualization = image.copy()
        if landmarks is not None and visualize:
            visualization = draw_landmarks(visualization, landmarks)
        visualization = draw_drowsiness_result(
            visualization, prediction, ear, confidence, mar=mar, face_rect=face_rect
        )

    if output_path and visualization is not None:
        if save_image(visualization, output_path):
            typer.echo(f"Result saved to {output_path}")
        else:
            typer.echo(f"Error: Failed to save result to {output_path}")

    if visualize and visualization is not None:
        cv2_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imshow("Drowsiness Detection Result", cv2_img)
        cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if save_features:
        features_path = (
            output_path.with_suffix(".csv")
            if output_path
            else Path(os.path.splitext(str(image_path))[0] + "_features.csv")
        )
        try:
            with open(features_path, "w") as f:
                f.write("feature,value\n")
                f.write(f"eye_aspect_ratio,{ear}\n")
                f.write(f"mouth_aspect_ratio,{mar}\n")
            typer.echo(f"Features saved to {features_path}")
        except Exception as e:
            typer.echo(f"Error saving features: {e}")


def _run_intoxication(
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    face_rect,
    model_type: str,
    color_mode: bool,
    visualize: bool,
    output_path: Optional[Path],
    save_features: bool,
    image_path: Path,
) -> None:
    """Run the intoxication CNN over a face crop."""
    redness_metrics = extract_skin_redness(image, face_rect, landmarks)
    typer.echo(f"Face Redness: {redness_metrics.get('face_redness', 0.0):.4f}")
    if "forehead_redness" in redness_metrics:
        typer.echo(f"Forehead Redness: {redness_metrics['forehead_redness']:.4f}")
    if "cheeks_redness" in redness_metrics:
        typer.echo(f"Cheeks Redness: {redness_metrics['cheeks_redness']:.4f}")

    # Resolve model path. `model_type` may be "cnn" (default) or a file path.
    if os.path.exists(model_type):
        model_path: Path = Path(model_type)
    elif model_type.lower() == "cnn":
        model_path = MODEL_DIR / "intoxication_cnn.pt"
    else:
        typer.echo(
            f"Error: intoxication only supports --model cnn or a CNN file path; got {model_type!r}."
        )
        raise typer.Exit(code=1)

    cnn = CNNDetector(infrared_mode=not color_mode)
    if model_path.exists():
        cnn.load(model_path)
        typer.echo(f"Loaded CNN model from {model_path}")
    else:
        typer.echo(f"Error: CNN model file not found at {model_path}. Run `sober-scan model download cnn`.")
        raise typer.Exit(code=1)

    # Face crop with margin.
    x1, y1, x2, y2 = face_rect
    h, w = image.shape[:2]
    mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    face_img = image[y1:y2, x1:x2]

    prediction, confidence = cnn.predict(face_img)
    typer.echo(f"Intoxication Detection Result: {prediction} (confidence: {confidence:.2f})")
    typer.echo(
        "  Note: this model's honest cross-subject AUC is ~0.48 \u2014 do not treat the result as reliable."
    )

    visualization = None
    if visualize or output_path:
        visualization = image.copy()
        if landmarks is not None and visualize:
            visualization = draw_landmarks(visualization, landmarks)
        visualization = draw_intoxication_result(
            visualization, prediction, redness_metrics.get("face_redness", 0.0), confidence, face_rect=face_rect
        )

    if output_path and visualization is not None:
        if save_image(visualization, output_path):
            typer.echo(f"Result saved to {output_path}")
        else:
            typer.echo(f"Error: Failed to save result to {output_path}")

    if visualize and visualization is not None:
        cv2_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imshow("Intoxication Detection Result", cv2_img)
        cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if save_features:
        features_path = (
            output_path.with_suffix(".csv")
            if output_path
            else Path(os.path.splitext(str(image_path))[0] + "_features.csv")
        )
        try:
            with open(features_path, "w") as f:
                f.write("feature,value\n")
                for key, value in redness_metrics.items():
                    f.write(f"{key},{value}\n")
            typer.echo(f"Features saved to {features_path}")
        except Exception as e:
            typer.echo(f"Error saving features: {e}")
