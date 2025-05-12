"""Utility functions for I/O, visualization, and logging."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sober_scan.config import LOGGING_CONFIG

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("sober_scan")


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure and return the logger.

    Args:
        verbose: If True, set console handler to DEBUG level

    Returns:
        Configured logger instance
    """
    if verbose:
        for handler in logger.handlers:
            if handler.name == "console":
                handler.setLevel(logging.DEBUG)
    return logger


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load an image from path.

    Args:
        image_path: Path to the image file

    Returns:
        Loaded image as numpy array or None if loading fails
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image path does not exist: {image_path}")
        return None

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Convert from BGR to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, save_path: Union[str, Path], create_dirs: bool = True) -> bool:
    """Save an image to the specified path.

    Args:
        image: Image as numpy array
        save_path: Path where to save the image
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        True if saved successfully, False otherwise
    """
    save_path = Path(save_path)

    if create_dirs:
        os.makedirs(save_path.parent, exist_ok=True)

    try:
        # Convert from RGB to BGR for OpenCV
        if image.shape[2] == 3:  # Check if it's a color image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        cv2.imwrite(str(save_path), image_bgr)
        logger.debug(f"Image saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {e}")
        return False


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw facial landmarks on an image.

    Args:
        image: Input image
        landmarks: Array of (x, y) landmark coordinates

    Returns:
        Image with landmarks drawn on it
    """
    vis_img = image.copy()

    # Draw each landmark point
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(vis_img, (int(x), int(y)), 2, (0, 255, 0), -1)
        # Draw landmark number for debug purposes
        # cv2.putText(vis_img, str(i), (int(x), int(y)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return vis_img


def draw_intoxication_result(image, prediction, redness, confidence, face_rect=None):
    """Draw intoxication detection results on an image.

    Args:
        image: Input image (NumPy array)
        prediction: Prediction label ("INTOXICATED" or "SOBER")
        redness: Facial redness value
        confidence: Confidence score of the prediction
        face_rect: Optional face rectangle coordinates (x1, y1, x2, y2)

    Returns:
        Image with detection results drawn
    """
    image_copy = image.copy()
    h, w = image_copy.shape[:2]

    # Draw face rectangle if provided
    if face_rect is not None:
        x1, y1, x2, y2 = face_rect
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Determine color and text based on prediction
    if prediction == "INTOXICATED":
        color = (255, 0, 0)  # Red for intoxicated
        status_text = "INTOXICATED"
    else:
        color = (0, 255, 0)  # Green for sober
        status_text = "SOBER"

    # Create a semi-transparent overlay for the text background
    overlay = image_copy.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)

    # Add the overlay with transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_copy, f"Status: {status_text}", (10, h - 50), font, 0.8, color, 2)
    cv2.putText(image_copy, f"Redness: {redness:.2f}", (10, h - 25), font, 0.8, (255, 255, 255), 2)
    cv2.putText(image_copy, f"Confidence: {confidence:.2f}", (w - 230, h - 25), font, 0.8, (255, 255, 255), 2)

    return image_copy


def draw_drowsiness_result(image, prediction, ear, confidence, mar=None, face_rect=None):
    """Draw drowsiness detection results on an image.

    Args:
        image: Input image
        prediction: Drowsiness prediction (ALERT or DROWSY)
        ear: Eye aspect ratio
        confidence: Confidence score for the prediction
        mar: Optional mouth aspect ratio
        face_rect: Optional face rectangle coordinates (x1, y1, x2, y2)

    Returns:
        Image with results drawn
    """
    result_img = image.copy()
    height, width = result_img.shape[:2]

    # Draw face rectangle if provided
    if face_rect is not None:
        x1, y1, x2, y2 = face_rect
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a semi-transparent overlay for the results
    overlay = result_img.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80 if mar is not None else 60), (0, 0, 0), -1)

    # Set text color based on drowsiness prediction
    if prediction == "ALERT":
        text_color = (0, 255, 0)  # Green for alert
    elif prediction == "DROWSY":
        text_color = (0, 0, 255)  # Red for drowsy
    else:
        text_color = (255, 255, 255)  # White for unknown

    # Draw text
    cv2.putText(
        overlay,
        f"Status: {prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
    )
    cv2.putText(
        overlay,
        f"EAR: {ear:.2f}",
        (width // 2 - 50, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Add MAR if provided
    if mar is not None:
        cv2.putText(
            overlay,
            f"MAR: {mar:.2f}",
            (width // 2 - 50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        overlay,
        f"Conf: {confidence:.2f}",
        (width - 180, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Apply the overlay with transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)

    return result_img


def plot_features(features: Dict[str, float], save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot extracted features as a bar chart.

    Args:
        features: Dictionary of extracted features
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Sort features by value for better visualization
    sorted_features = dict(sorted(features.items(), key=lambda x: x[1]))

    plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.title("Extracted Facial Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.debug(f"Feature plot saved to {save_path}")

    plt.close()


def create_progress_bar(total: int, prefix: str = "", suffix: str = "", length: int = 50, fill: str = "█") -> None:
    """Print a text-based progress bar.

    Args:
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
        fill: Bar fill character
    """

    def update(iteration):
        percent = 100 * (iteration / float(total))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="")
        if iteration == total:
            print()

    return update
