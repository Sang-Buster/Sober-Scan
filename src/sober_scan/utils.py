"""Utility functions for I/O, visualization, and logging."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sober_scan.config import BAC_THRESHOLDS, LOGGING_CONFIG, BACLevel

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


def draw_intoxication_result(image: np.ndarray, bac_level: BACLevel, confidence: float) -> np.ndarray:
    """Draw intoxication results on an image.

    Args:
        image: Input image
        bac_level: Detected BAC level
        confidence: Confidence score of the detection

    Returns:
        Image with detection result drawn on it
    """
    result_img = image.copy()
    h, w = result_img.shape[:2]

    # Define colors for different BAC levels
    colors = {
        BACLevel.SOBER: (0, 255, 0),  # Green
        BACLevel.MILD: (0, 255, 255),  # Yellow
        BACLevel.MODERATE: (0, 165, 255),  # Orange
        BACLevel.SEVERE: (0, 0, 255),  # Red
    }

    # Status text based on BAC level
    status_text = f"{bac_level.name}: {confidence:.2f}"

    # Add a semi-transparent overlay at the bottom
    overlay = result_img.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, result_img, 0.5, 0, result_img)

    # Add the BAC level text
    cv2.putText(result_img, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[bac_level], 2)

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


def bac_probability_to_level(probability: float) -> Tuple[BACLevel, float]:
    """Convert intoxication probability to BAC level.

    Args:
        probability: Model's prediction probability (0.0-1.0)

    Returns:
        Tuple of (BAC level enum, confidence score)
    """
    # Determine BAC level based on probability thresholds
    if probability < BAC_THRESHOLDS[BACLevel.SOBER]:
        return BACLevel.SOBER, 1.0 - (probability / BAC_THRESHOLDS[BACLevel.SOBER])
    elif probability < BAC_THRESHOLDS[BACLevel.MILD]:
        range_size = BAC_THRESHOLDS[BACLevel.MILD] - BAC_THRESHOLDS[BACLevel.SOBER]
        normalized = (probability - BAC_THRESHOLDS[BACLevel.SOBER]) / range_size
        return BACLevel.MILD, normalized
    elif probability < BAC_THRESHOLDS[BACLevel.MODERATE]:
        range_size = BAC_THRESHOLDS[BACLevel.MODERATE] - BAC_THRESHOLDS[BACLevel.MILD]
        normalized = (probability - BAC_THRESHOLDS[BACLevel.MILD]) / range_size
        return BACLevel.MODERATE, normalized
    else:
        range_size = BAC_THRESHOLDS[BACLevel.SEVERE] - BAC_THRESHOLDS[BACLevel.MODERATE]
        normalized = (probability - BAC_THRESHOLDS[BACLevel.MODERATE]) / range_size
        return BACLevel.SEVERE, normalized


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
