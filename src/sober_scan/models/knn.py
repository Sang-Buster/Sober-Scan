"""KNN model for drowsiness detection using eye aspect ratio (EAR) and mouth aspect ratio (MAR)."""

import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DrowsinessDetector:
    """KNN-based drowsiness detector using EAR and MAR."""

    def __init__(self):
        """Initialize the model"""
        # Increase n_neighbors to smooth decision boundaries and prevent overfitting
        # Use 'distance' weights to give less importance to far-away points
        self.model = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            p=2,  # Euclidean distance
        )
        self.scaler = StandardScaler()
        self.pipeline = make_pipeline(self.scaler, self.model)
        self.is_trained = False

    def train(self, features, labels):
        """Train the model with eye and mouth features.

        Args:
            features (numpy.ndarray): Feature array containing EAR and MAR values
            labels (numpy.ndarray): Labels for drowsiness state
        """
        self.pipeline.fit(features, labels)
        self.is_trained = True
        return self

    def predict(self, features):
        """Predict drowsiness based on features.

        Args:
            features (numpy.ndarray): Feature array containing EAR and MAR values

        Returns:
            tuple: (prediction, probability)
                prediction: "ALERT", "DROWSY", or "UNKNOWN"
                probability: Float probability of the DROWSY class (1)
        """
        if not self.is_trained:
            return "ALERT", 0.0

        # Get prediction
        features_array = np.array([features]) if not isinstance(features, list) else np.array(features)
        pred = self.pipeline.predict(features_array)[0]

        # Get probability of the positive class (DROWSY = 1)
        proba_drowsy = self.pipeline.predict_proba(features_array)[0][1]

        if pred == 0:
            return "ALERT", float(proba_drowsy)
        elif pred == 1:
            return "DROWSY", float(proba_drowsy)
        else:
            return "UNKNOWN", 0.0

    def save(self, filepath):
        """Save the model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the pipeline which includes both scaler and model
        joblib.dump(self.pipeline, filepath)
        return filepath

    def load(self, filepath):
        """Load the model from disk.

        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        return self
