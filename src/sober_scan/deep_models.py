"""Deep learning models for intoxication detection from facial images."""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from sober_scan.config import DEFAULT_MODELS, BACLevel
from sober_scan.utils import logger


class FaceClassifierCNN(nn.Module):
    """CNN model for facial intoxication classification."""

    def __init__(self, num_classes: int = 4, use_pretrained: bool = True):
        """Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default 4: sober, mild, moderate, severe)
            use_pretrained: Whether to use pretrained weights
        """
        super(FaceClassifierCNN, self).__init__()

        # Use a pre-trained ResNet as the backbone
        self.backbone = models.resnet50(pretrained=use_pretrained)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class FacialGraphNN(nn.Module):
    """Graph Neural Network for facial landmark-based intoxication detection."""

    def __init__(self, num_landmarks: int = 68, hidden_dim: int = 64, num_classes: int = 4):
        """Initialize the GNN model.

        Args:
            num_landmarks: Number of facial landmarks
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
        """
        super(FacialGraphNN, self).__init__()

        # Input features per landmark (x, y coordinates)
        self.input_dim = 2

        # GNN layers
        self.gnn_layer1 = GraphConvLayer(self.input_dim, hidden_dim)
        self.gnn_layer2 = GraphConvLayer(hidden_dim, hidden_dim)

        # Global pooling and classification layers
        self.fc1 = nn.Linear(hidden_dim * num_landmarks, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_landmarks, 2)
            adj_matrix: Adjacency matrix of shape (batch_size, num_landmarks, num_landmarks)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Apply GNN layers
        x = F.relu(self.gnn_layer1(x, adj_matrix))
        x = F.relu(self.gnn_layer2(x, adj_matrix))

        # Flatten and apply fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class GraphConvLayer(nn.Module):
    """Graph convolutional layer."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize the graph convolutional layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases."""
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            x: Input tensor of shape (batch_size, num_nodes, in_features)
            adj: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)

        Returns:
            Output tensor of shape (batch_size, num_nodes, out_features)
        """
        # Matrix multiplication: H = X * W
        support = torch.matmul(x, self.weight)

        # Graph convolution: output = A * H
        output = torch.matmul(adj, support)

        # Add bias
        return output + self.bias


class DeepModelHandler:
    """Handler for deep learning models."""

    def __init__(self, model_type: str = "cnn", model_path: Optional[str] = None):
        """Initialize the deep model handler.

        Args:
            model_type: Type of model ("cnn" or "gnn")
            model_path: Path to the saved model or None to use default
        """
        self.model_type = model_type
        self.model_path = model_path or DEFAULT_MODELS[model_type]["path"]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._load_model()

    def _load_model(self) -> bool:
        """Load the deep learning model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.model_type == "cnn":
                self.model = FaceClassifierCNN()
            elif self.model_type == "gnn":
                num_landmarks = DEFAULT_MODELS["gnn"]["landmarks"]
                self.model = FacialGraphNN(num_landmarks=num_landmarks)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return False

            # Check if model file exists
            if os.path.exists(self.model_path):
                # Load saved weights
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded {self.model_type} model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}. Using untrained model.")
                self.model.to(self.device)
                self.model.eval()
                return False

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess an image for CNN inference.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image tensor
        """
        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def preprocess_landmarks(self, landmarks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess landmarks for GNN inference.

        Args:
            landmarks: Array of landmark coordinates

        Returns:
            Tuple of (landmark tensor, adjacency matrix tensor)
        """
        # Normalize landmarks to [0,1] range
        landmarks = landmarks.astype(np.float32)
        min_values = np.min(landmarks, axis=0)
        max_values = np.max(landmarks, axis=0)
        normalized_landmarks = (landmarks - min_values) / (max_values - min_values + 1e-6)

        # Convert to tensor and add batch dimension
        landmark_tensor = torch.tensor(normalized_landmarks, dtype=torch.float32).unsqueeze(0)

        # Create adjacency matrix (fully connected graph for simplicity)
        num_landmarks = landmarks.shape[0]
        adj_matrix = torch.ones(1, num_landmarks, num_landmarks, dtype=torch.float32)

        # Add self-loops and normalize
        adj_matrix = adj_matrix + torch.eye(num_landmarks).unsqueeze(0)
        adj_matrix = adj_matrix / adj_matrix.sum(dim=2, keepdim=True)

        return landmark_tensor.to(self.device), adj_matrix.to(self.device)

    def predict_cnn(self, image: np.ndarray) -> Tuple[BACLevel, float]:
        """Predict intoxication from image using CNN.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return BACLevel.SOBER, 0.0

        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()

            # Get the predicted class and confidence
            pred_class = np.argmax(probabilities)
            confidence = probabilities[pred_class]

            # Convert to BAC level enum
            bac_level = BACLevel(pred_class)

            return bac_level, float(confidence)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return BACLevel.SOBER, 0.0

    def predict_gnn(self, landmarks: np.ndarray) -> Tuple[BACLevel, float]:
        """Predict intoxication from facial landmarks using GNN.

        Args:
            landmarks: Array of landmark coordinates

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return BACLevel.SOBER, 0.0

        try:
            # Preprocess landmarks
            landmark_tensor, adj_matrix = self.preprocess_landmarks(landmarks)

            # Run inference
            with torch.no_grad():
                output = self.model(landmark_tensor, adj_matrix)
                probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()

            # Get the predicted class and confidence
            pred_class = np.argmax(probabilities)
            confidence = probabilities[pred_class]

            # Convert to BAC level enum
            bac_level = BACLevel(pred_class)

            return bac_level, float(confidence)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return BACLevel.SOBER, 0.0

    def predict(
        self, image: Optional[np.ndarray] = None, landmarks: Optional[np.ndarray] = None
    ) -> Tuple[BACLevel, float]:
        """Predict intoxication using the appropriate model.

        Args:
            image: Input image (required for CNN)
            landmarks: Facial landmarks (required for GNN)

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        if self.model_type == "cnn":
            if image is None:
                logger.error("Image is required for CNN prediction.")
                return BACLevel.SOBER, 0.0
            return self.predict_cnn(image)

        elif self.model_type == "gnn":
            if landmarks is None:
                logger.error("Landmarks are required for GNN prediction.")
                return BACLevel.SOBER, 0.0
            return self.predict_gnn(landmarks)

        logger.error(f"Unsupported model type: {self.model_type}")
        return BACLevel.SOBER, 0.0

    def save_model(self, model_path: Optional[str] = None) -> bool:
        """Save the model to disk.

        Args:
            model_path: Path to save the model or None to use default

        Returns:
            True if saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No model to save.")
            return False

        try:
            save_path = model_path or self.model_path

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save model weights
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
