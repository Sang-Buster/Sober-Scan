"""Traditional machine learning methods for intoxication detection."""

import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sober_scan.config import MODEL_DIR, BACLevel
from sober_scan.utils import logger


class TraditionalModel:
    """Base class for traditional ML models."""

    def __init__(self, model_type: str = "svm", model_path: Optional[str] = None):
        """Initialize the model.

        Args:
            model_type: Type of model ('svm' or 'random_forest')
            model_path: Path to a saved model or None to create a new one
        """
        self.model_type = model_type
        self.model_path = model_path if model_path else os.path.join(MODEL_DIR, f"traditional_{model_type}.joblib")
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        self._create_model()
        self._try_load_model()

    def _create_model(self) -> None:
        """Create a new model based on model_type."""
        if self.model_type == "svm":
            self.model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            logger.warning(f"Unsupported model type: {self.model_type}. Using SVM.")
            self.model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)

    def _try_load_model(self) -> bool:
        """Try to load a pre-trained model.

        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                saved_data = joblib.load(self.model_path)
                self.model = saved_data["model"]
                self.scaler = saved_data["scaler"]
                self.feature_names = saved_data["feature_names"]
                logger.info(f"Loaded model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                # When no model is found, we'll still create the empty model object
                # but will set the feature_names to a default list to avoid errors
                # This will allow the predict method to be called (returning default values)
                self.feature_names = ["face_redness", "forehead_redness", "cheeks_redness"]
                self.model = None
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.feature_names = ["face_redness", "forehead_redness", "cheeks_redness"]
            return False

    def train(self, X: Union[List[Dict[str, float]], pd.DataFrame], y: Union[List[int], np.ndarray]) -> float:
        """Train the model on the provided data.

        Args:
            X: Feature data as list of dictionaries or DataFrame
            y: Labels as list or array (0=sober, 1=mild, 2=moderate, 3=severe)

        Returns:
            Training accuracy
        """
        # Convert input features to DataFrame if needed
        if isinstance(X, list):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        # Remember feature names for prediction
        self.feature_names = list(X_df.columns)

        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2, random_state=42)

        # Fit the scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train the model
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        val_accuracy = self.model.score(X_val_scaled, y_val)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        # Generate detailed classification report
        y_pred = self.model.predict(X_val_scaled)
        logger.info(f"Classification report:\n{classification_report(y_val, y_pred)}")

        # Save the trained model
        self.save()

        return val_accuracy

    def predict(self, features: Dict[str, float]) -> Tuple[BACLevel, float]:
        """Predict BAC level from features.

        Args:
            features: Dictionary of facial features

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        if self.model is None:
            logger.warning("Model not loaded. Using default prediction.")
            # Default to MILD level with medium confidence when no model is available
            return BACLevel.MILD, 0.5

        try:
            # Convert features to DataFrame with the same columns as training data
            X = pd.DataFrame([features])

            # Handle missing features
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0.0

            # Make sure columns are in the same order as during training
            X = X[self.feature_names]

            # Scale the features
            X_scaled = self.scaler.transform(X)

            # Predict
            probabilities = self.model.predict_proba(X_scaled)[0]
            pred_class = int(self.model.predict(X_scaled)[0])
            confidence = float(probabilities[pred_class])

            # Convert to BAC level enum
            bac_level = BACLevel(pred_class)

            return bac_level, confidence
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Default to MILD level with medium confidence when prediction fails
            return BACLevel.MILD, 0.5

    def save(self) -> bool:
        """Save the model to disk.

        Returns:
            True if saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No model to save.")
            return False

        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save model, scaler, and feature names together
            saved_data = {"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names}

            joblib.dump(saved_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


def evaluate_traditional_model(
    features_path: str, model_type: str = "svm", test_size: float = 0.2, save_model: bool = True
) -> Dict[str, float]:
    """Load features, train a model, and evaluate its performance.

    Args:
        features_path: Path to CSV file with features and labels
        model_type: Type of model to train ('svm' or 'random_forest')
        test_size: Fraction of data to use for testing
        save_model: Whether to save the trained model

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Load features dataset
        data = pd.read_csv(features_path)

        # Separate features and labels
        X = data.drop("label", axis=1)
        y = data["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Create and train model
        model = TraditionalModel(model_type=model_type)
        model.train(X_train, y_train)

        # Evaluate on test set
        predictions = []
        confidences = []

        for _, features in X_test.iterrows():
            features_dict = features.to_dict()
            bac_level, confidence = model.predict(features_dict)
            predictions.append(bac_level.value)
            confidences.append(confidence)

        test_accuracy = accuracy_score(y_test, predictions)

        # Print detailed results
        report = classification_report(y_test, predictions, output_dict=True)

        # Save model if requested
        if save_model:
            model.save()

        # Return metrics
        metrics = {
            "accuracy": test_accuracy,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
        }

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
