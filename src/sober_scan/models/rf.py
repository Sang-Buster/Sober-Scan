"""Random Forest model for drowsiness detection using eye aspect ratio (EAR) and mouth aspect ratio (MAR)."""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DrowsinessDetector:
    """Random Forest-based drowsiness detector using EAR and MAR with data accumulation support."""

    def __init__(self, warm_start=True):
        """Initialize the model.
        
        Args:
            warm_start: If True, allows adding more trees incrementally
        """
        # Increase min_samples_leaf and min_samples_split to prevent overfitting
        # Reduce max_depth to create a more generalized model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            warm_start=warm_start,  # Allows incremental tree addition
        )
        self.scaler = StandardScaler()
        self.pipeline = make_pipeline(self.scaler, self.model)
        self.is_trained = False
        self.n_samples_seen = 0
        self.warm_start = warm_start
        # Store training data for incremental updates
        self.training_features = None
        self.training_labels = None

    def train(self, features, labels, store_data=True):
        """Train the model with eye and mouth features.

        Args:
            features (numpy.ndarray): Feature array containing EAR and MAR values
            labels (numpy.ndarray): Labels for drowsiness state
            store_data (bool): Whether to store training data for future incremental updates
        """
        self.pipeline.fit(features, labels)
        self.is_trained = True
        self.n_samples_seen += len(features)
        
        # Store training data for incremental updates
        if store_data:
            self.training_features = np.array(features)
            self.training_labels = np.array(labels)
        
        return self

    def update_with_new_data(self, new_features, new_labels, add_trees=False):
        """Update the model by combining old and new data.
        
        Random Forest supports two update modes:
        1. Full retrain: Rebuild all trees on combined data (slower, more accurate)
        2. Add trees: Add new trees to forest (faster, truly incremental)

        Args:
            new_features (numpy.ndarray): New feature array
            new_labels (numpy.ndarray): New labels
            add_trees (bool): If True and warm_start=True, add more trees instead of retraining
            
        Returns:
            self for chaining
            
        Raises:
            ValueError: If training data was not stored
        """
        if self.training_features is None or self.training_labels is None:
            raise ValueError(
                "No training data stored. Either train with store_data=True or use train() instead"
            )
        
        new_features_array = np.array(new_features) if not isinstance(new_features, np.ndarray) else new_features
        new_labels_array = np.array(new_labels) if not isinstance(new_labels, np.ndarray) else new_labels
        
        # Ensure features are 2D
        if new_features_array.ndim == 1:
            new_features_array = new_features_array.reshape(1, -1)
        
        # Combine old and new data
        combined_features = np.vstack([self.training_features, new_features_array])
        combined_labels = np.hstack([self.training_labels, new_labels_array])
        
        if add_trees and self.warm_start:
            # Incremental tree addition (truly incremental!)
            # Increase number of trees and retrain (adds new trees, keeps old ones)
            old_n_estimators = self.model.n_estimators
            self.model.n_estimators += 50  # Add 50 more trees
            
            # Fit with warm_start=True (only trains new trees)
            self.scaler.partial_fit(new_features_array)
            scaled_combined = self.scaler.transform(combined_features)
            self.model.fit(scaled_combined, combined_labels)
            
            print(f"Added {self.model.n_estimators - old_n_estimators} trees (total: {self.model.n_estimators})")
        else:
            # Full retrain on combined data
            self.pipeline.fit(combined_features, combined_labels)
        
        # Update stored data
        self.training_features = combined_features
        self.training_labels = combined_labels
        self.n_samples_seen = len(combined_features)
        
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
            "n_samples_seen": self.n_samples_seen,
            "training_features": self.training_features,
            "training_labels": self.training_labels,
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
                self.n_samples_seen = model_data.get("n_samples_seen", 0)
                self.training_features = model_data.get("training_features", None)
                self.training_labels = model_data.get("training_labels", None)
                self.model = self.pipeline.named_steps["randomforestclassifier"]
                self.scaler = self.pipeline.named_steps["standardscaler"]
            else:
                # Old format: just the pipeline
                self.pipeline = model_data
                self.model = self.pipeline.named_steps["randomforestclassifier"]
                self.scaler = self.pipeline.named_steps["standardscaler"]
                self.n_samples_seen = 0
                self.training_features = None
                self.training_labels = None
        except Exception:
            # Fallback to old format
            self.pipeline = joblib.load(filepath)
            self.model = self.pipeline.named_steps["randomforestclassifier"]
            self.scaler = self.pipeline.named_steps["standardscaler"]
            self.n_samples_seen = 0
            self.training_features = None
            self.training_labels = None
        
        self.is_trained = True
        return self
