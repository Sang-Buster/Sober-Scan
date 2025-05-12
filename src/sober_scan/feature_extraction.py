"""Functions to extract facial features for intoxication detection."""

import os
from typing import Callable, Dict, Optional, Tuple

import cv2
import dlib
import numpy as np

from sober_scan.config import FEATURE_PARAMS, MODEL_DIR
from sober_scan.utils import logger


def get_face_detector(detector_type: str = "dlib") -> Callable:
    """Get a face detector based on the specified type.

    Args:
        detector_type: Type of face detector ("dlib", "mediapipe", "opencv")

    Returns:
        Face detector function
    """
    if detector_type == "dlib":
        detector = dlib.get_frontal_face_detector()
        return lambda img: [(rect.left(), rect.top(), rect.right(), rect.bottom()) for rect in detector(img)]
    elif detector_type == "opencv":
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return lambda img: face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 1.1, 4)
    else:
        logger.warning(f"Unsupported detector type: {detector_type}. Using dlib.")
        detector = dlib.get_frontal_face_detector()
        return lambda img: [(rect.left(), rect.top(), rect.right(), rect.bottom()) for rect in detector(img)]


def get_landmark_detector(detector_type: str = "dlib") -> Callable:
    """Get a facial landmark detector based on the specified type.

    Args:
        detector_type: Type of landmark detector ("dlib")

    Returns:
        Landmark detector function

    Raises:
        FileNotFoundError: If the required model file is not found
    """
    if detector_type == "dlib":
        # Path to the pre-trained shape predictor model
        model_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(model_path):
            error_msg = (
                f"Required dlib shape predictor model not found at {model_path}. "
                "Please download the model using: "
                "'sober-scan model download dlib-shape-predictor'"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            predictor = dlib.shape_predictor(model_path)

            def detect_landmarks(img, face_rect):
                dlib_rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[2], face_rect[3])
                shape = predictor(img, dlib_rect)
                return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            return detect_landmarks
        except RuntimeError as e:
            error_msg = f"Error loading dlib shape predictor model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        error_msg = f"Unsupported landmark detector: {detector_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def detect_face_and_landmarks(image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Detect face and facial landmarks in an image.

    Args:
        image: Input image

    Returns:
        Tuple of (face_rect, landmarks) or (None, None) if detection fails
    """
    # Get detectors based on config
    face_detector = get_face_detector(FEATURE_PARAMS["face_detector"])

    # Detect faces
    faces = face_detector(image)

    if not faces:
        logger.warning("No face detected in the image")
        return None, None

    # Use the first (or largest) face
    if FEATURE_PARAMS["face_detector"] == "opencv":
        # Convert OpenCV format (x, y, w, h) to (x1, y1, x2, y2)
        face = faces[0]
        face_rect = (face[0], face[1], face[0] + face[2], face[1] + face[3])
    else:
        face_rect = faces[0]  # Already in (x1, y1, x2, y2) format

    # Detect landmarks
    try:
        # Get landmark detector
        landmark_detector = get_landmark_detector(FEATURE_PARAMS["landmark_model"])
        landmarks = landmark_detector(image, face_rect)
        return face_rect, landmarks
    except (FileNotFoundError, RuntimeError, ValueError):
        # Let the error propagate up to be handled by the command
        raise
    except Exception as e:
        logger.error(f"Error detecting landmarks: {e}")
        return face_rect, None


def calculate_eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Calculate eye aspect ratio (measure of eye openness).

    The eye aspect ratio (EAR) is calculated as:
    EAR = (dist(p2,p6) + dist(p3,p5)) / (2 * dist(p1,p4))

    where p1, p2, ..., p6 are the 6 landmarks around the eye.

    Args:
        landmarks: Array of facial landmarks (68-point format)

    Returns:
        Mean eye aspect ratio of both eyes
    """
    # Check if landmarks are available
    if landmarks is None or len(landmarks) < 68:
        return 0.0

    # Define eye landmark indices for 68-point model
    # Left eye: 36-41, Right eye: 42-47
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    def eye_ratio(eye):
        # Calculate Euclidean distances
        a = np.linalg.norm(eye[1] - eye[5])  # p2-p6
        b = np.linalg.norm(eye[2] - eye[4])  # p3-p5
        c = np.linalg.norm(eye[0] - eye[3])  # p1-p4

        # Return eye aspect ratio
        return (a + b) / (2.0 * c) if c > 0 else 0.0

    # Calculate the ratio for both eyes and return the mean
    left_ear = eye_ratio(left_eye)
    right_ear = eye_ratio(right_eye)

    return (left_ear + right_ear) / 2.0


def extract_skin_redness(image: np.ndarray, face_rect: Tuple[int, int, int, int], landmarks: np.ndarray) -> Dict[str, float]:
    """Extract skin color metrics focused on redness.

    Args:
        image: Input image (RGB)
        face_rect: Face bounding box (x1, y1, x2, y2)
        landmarks: Facial landmarks

    Returns:
        Dictionary of redness metrics for different facial regions
    """
    if landmarks is None:
        return {"face_redness": 0.0, "forehead_redness": 0.0, "cheeks_redness": 0.0}

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    results = {}

    # Analyze full face
    x1, y1, x2, y2 = face_rect
    face_roi = hsv[y1:y2, x1:x2]
    if face_roi.size > 0:
        # Extract hue and saturation channels
        h, s, v = cv2.split(face_roi)

        # Measure redness (lower hue values in HSV)
        # Red is around hue 0 or 180
        mask = ((h < 10) | (h > 170)) & (s > 50)
        redness = np.mean(s[mask]) if mask.any() else 0

        results["face_redness"] = float(redness / 255.0)
    else:
        results["face_redness"] = 0.0

    # Analyze forehead region if landmarks available
    if "forehead" in FEATURE_PARAMS["skin_regions"] and landmarks is not None:
        # Define forehead region above eyebrows
        eyebrow_y = int(min(landmarks[17:27, 1]))  # Eyebrow landmarks
        forehead_height = int((eyebrow_y - y1) * 0.7)

        if forehead_height > 0:
            forehead_roi = hsv[eyebrow_y - forehead_height : eyebrow_y, x1:x2]
            if forehead_roi.size > 0:
                h, s, v = cv2.split(forehead_roi)
                mask = ((h < 10) | (h > 170)) & (s > 50)
                redness = np.mean(s[mask]) if mask.any() else 0
                results["forehead_redness"] = float(redness / 255.0)
            else:
                results["forehead_redness"] = 0.0
        else:
            results["forehead_redness"] = 0.0

    # Analyze cheeks if landmarks available
    if "cheeks" in FEATURE_PARAMS["skin_regions"] and landmarks is not None:
        # Left cheek region
        left_cheek_center = landmarks[31]  # Approximate cheek center
        left_cheek_x = int(left_cheek_center[0] - 30)
        left_cheek_y = int(left_cheek_center[1])
        left_cheek_roi = hsv[left_cheek_y : left_cheek_y + 30, max(0, left_cheek_x) : left_cheek_x + 30]

        # Right cheek region
        right_cheek_center = landmarks[35]  # Approximate cheek center
        right_cheek_x = int(right_cheek_center[0])
        right_cheek_y = int(right_cheek_center[1])
        right_cheek_roi = hsv[right_cheek_y : right_cheek_y + 30, right_cheek_x : right_cheek_x + 30]

        # Calculate redness
        h1, s1, v1 = cv2.split(left_cheek_roi) if left_cheek_roi.size > 0 else (np.array([]), np.array([]), np.array([]))
        h2, s2, v2 = cv2.split(right_cheek_roi) if right_cheek_roi.size > 0 else (np.array([]), np.array([]), np.array([]))

        # Combine cheek data
        h = np.concatenate([h1.flatten(), h2.flatten()])
        s = np.concatenate([s1.flatten(), s2.flatten()])

        if h.size > 0 and s.size > 0:
            mask = ((h < 10) | (h > 170)) & (s > 50)
            redness = np.mean(s[mask]) if mask.any() else 0
            results["cheeks_redness"] = float(redness / 255.0)
        else:
            results["cheeks_redness"] = 0.0

    return results


def calculate_landmark_distances(landmarks: np.ndarray) -> Dict[str, float]:
    """Calculate distances between key facial landmarks.

    Args:
        landmarks: Array of facial landmarks

    Returns:
        Dictionary of landmark distances
    """
    if landmarks is None or len(landmarks) < 68:
        return {}

    distances = {}

    # Calculate distances between configured landmark pairs
    for i, (idx1, idx2) in enumerate(FEATURE_PARAMS["landmark_distances"]):
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            dist = np.linalg.norm(landmarks[idx1] - landmarks[idx2])
            distances[f"landmark_dist_{idx1}_{idx2}"] = float(dist)

    # Calculate eye aspect ratio
    ear = calculate_eye_aspect_ratio(landmarks)
    distances["eye_aspect_ratio"] = ear

    # Add eyebrow height (distance from eye to eyebrow)
    left_eye_y = landmarks[37][1]  # Upper left eye
    left_brow_y = landmarks[19][1]  # Left eyebrow
    right_eye_y = landmarks[44][1]  # Upper right eye
    right_brow_y = landmarks[24][1]  # Right eyebrow

    # Eyebrow-to-eye distances
    distances["left_brow_eye_dist"] = float(abs(left_eye_y - left_brow_y))
    distances["right_brow_eye_dist"] = float(abs(right_eye_y - right_brow_y))

    # Mouth shape metrics
    mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
    mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
    distances["mouth_aspect_ratio"] = float(mouth_height / mouth_width if mouth_width > 0 else 0)

    return distances


def extract_features(image: np.ndarray) -> Dict[str, float]:
    """Extract all features from a facial image for intoxication detection.

    Args:
        image: Input image

    Returns:
        Dictionary of extracted features

    Raises:
        FileNotFoundError: If required model files are not found
        RuntimeError: If there's an error in the landmark detection
        ValueError: If an unsupported detector is specified
    """
    # Detect face and landmarks
    face_rect, landmarks = detect_face_and_landmarks(image)

    if face_rect is None:
        logger.warning("Could not extract features: no face detected")
        return {}

    # Extract all features
    features = {}

    # Get skin redness features
    redness_features = extract_skin_redness(image, face_rect, landmarks)
    features.update(redness_features)

    # Get landmark distance features
    if landmarks is not None:
        distance_features = calculate_landmark_distances(landmarks)
        features.update(distance_features)

    # Add face position and size as features
    if face_rect is not None:
        x1, y1, x2, y2 = face_rect
        h, w = image.shape[:2]
        features["face_rel_x"] = float((x1 + x2) / 2 / w)  # Relative x position
        features["face_rel_y"] = float((y1 + y2) / 2 / h)  # Relative y position
        features["face_rel_size"] = float((x2 - x1) * (y2 - y1) / (w * h))  # Relative size

    logger.debug(f"Extracted {len(features)} features")
    return features
