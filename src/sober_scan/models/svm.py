"""SVM model for drowsiness detection using eye aspect ratio (EAR) and mouth aspect ratio (MAR)."""

import os

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class DrowsinessDetector:
    """SVM-based drowsiness detector using EAR and MAR with incremental learning support."""

    def __init__(self, use_incremental=False):
        """Initialize the model.
        
        Args:
            use_incremental: If True, use SGDClassifier for incremental learning support
        """
        self.use_incremental = use_incremental
        
        if use_incremental:
            # Use SGDClassifier with hinge loss (SVM-like) for incremental learning
            self.model = SGDClassifier(
                loss="hinge",  # SVM-like loss
                alpha=0.01,  # Regularization parameter (similar to 1/C)
                max_iter=1000,
                tol=1e-3,
                class_weight="balanced",
                random_state=42,
            )
        else:
            # Use lower C value for better regularization (prevents overfitting)
            # gamma='scale' makes the model rely less on individual training examples
            self.model = SVC(kernel="rbf", C=0.1, gamma="scale", probability=True, class_weight="balanced")
        
        self.scaler = StandardScaler()
        self.pipeline = make_pipeline(self.scaler, self.model)
        self.is_trained = False
        self.n_samples_seen = 0

    def train(self, features, labels):
        """Train the model with eye and mouth features.

        Args:
            features (numpy.ndarray): Feature array containing EAR and MAR values
            labels (numpy.ndarray): Labels for drowsiness state
        """
        self.pipeline.fit(features, labels)
        self.is_trained = True
        self.n_samples_seen += len(features)
        return self

    def partial_fit(self, features, labels):
        """Incrementally train the model with new data (only for incremental mode).

        Args:
            features (numpy.ndarray): Feature array containing EAR and MAR values
            labels (numpy.ndarray): Labels for drowsiness state
            
        Returns:
            self for chaining
            
        Raises:
            ValueError: If not in incremental mode
        """
        if not self.use_incremental:
            raise ValueError("partial_fit is only available when use_incremental=True")
        
        features_array = np.array(features) if not isinstance(features, np.ndarray) else features
        labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        
        # Ensure features are 2D
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        if not self.is_trained:
            # First time: fit the scaler and model
            self.scaler.fit(features_array)
            scaled_features = self.scaler.transform(features_array)
            self.model.partial_fit(scaled_features, labels_array, classes=[0, 1])
            self.is_trained = True
        else:
            # Incremental update: update scaler and model
            # Update scaler statistics incrementally
            self.scaler.partial_fit(features_array)
            scaled_features = self.scaler.transform(features_array)
            self.model.partial_fit(scaled_features, labels_array)
        
        self.n_samples_seen += len(features_array)
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

        # Save the pipeline and metadata
        model_data = {
            "pipeline": self.pipeline,
            "use_incremental": self.use_incremental,
            "n_samples_seen": self.n_samples_seen,
        }
        joblib.dump(model_data, filepath)
        return filepath

    def load(self, filepath):
        """Load the model from disk.

        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            # Try loading new format with metadata
            model_data = joblib.load(filepath)
            if isinstance(model_data, dict):
                self.pipeline = model_data["pipeline"]
                self.use_incremental = model_data.get("use_incremental", False)
                self.n_samples_seen = model_data.get("n_samples_seen", 0)
                self.model = self.pipeline.named_steps.get("sgdclassifier") or self.pipeline.named_steps.get("svc")
                self.scaler = self.pipeline.named_steps["standardscaler"]
            else:
                # Old format: just the pipeline
                self.pipeline = model_data
                self.model = self.pipeline.named_steps.get("sgdclassifier") or self.pipeline.named_steps.get("svc")
                self.scaler = self.pipeline.named_steps["standardscaler"]
                self.use_incremental = isinstance(self.model, SGDClassifier)
                self.n_samples_seen = 0
        except Exception:
            # Fallback to old format
            self.pipeline = joblib.load(filepath)
            self.model = self.pipeline.named_steps.get("sgdclassifier") or self.pipeline.named_steps.get("svc")
            self.scaler = self.pipeline.named_steps["standardscaler"]
            self.use_incremental = isinstance(self.model, SGDClassifier)
            self.n_samples_seen = 0
        
        self.is_trained = True
        return self
