"""CNN model for intoxication detection."""

import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


class IntoxicationCNN(nn.Module):
    """CNN architecture for intoxication detection with transfer learning."""

    def __init__(self, in_channels=3):
        """Initialize the model with the specified input channels."""
        super(IntoxicationCNN, self).__init__()
        self.in_channels = in_channels  # 3 for RGB, 1 for infrared

        # Use MobileNetV2 - more efficient model
        self.backbone = models.mobilenet_v2(weights="DEFAULT")

        # Replace the first conv layer if needed for grayscale images
        if in_channels == 1:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )

        # Replace the classifier for binary classification
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


class IntoxicationDetector:
    """CNN-based intoxication detector."""

    def __init__(self, image_size=(224, 224), infrared_mode=True):
        """Initialize the model.

        Args:
            image_size: Input image size (height, width)
            infrared_mode: Whether to use infrared images (default) or RGB
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.infrared_mode = infrared_mode
        self.in_channels = 1 if infrared_mode else 3
        self.model = IntoxicationCNN(in_channels=self.in_channels).to(self.device)
        self.is_trained = False

    def preprocess_image(self, image):
        """Preprocess an image for the model.

        Args:
            image: Input image (NumPy array)

        Returns:
            Preprocessed tensor
        """
        # Handle None image
        if image is None:
            # Create blank image
            if self.infrared_mode:
                image = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
            else:
                image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        # Resize
        resized = cv2.resize(image, self.image_size)

        # Convert to grayscale if using infrared mode
        if self.infrared_mode and len(resized.shape) == 3 and resized.shape[2] == 3:
            processed = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            processed = processed.reshape(self.image_size[0], self.image_size[1], 1)
        else:
            processed = resized

        # Normalize
        processed = processed.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        if self.infrared_mode:
            tensor = torch.tensor(processed.reshape(1, 1, self.image_size[0], self.image_size[1]))
        else:
            # Ensure we have 3 channels
            if len(processed.shape) == 2:
                processed = np.stack([processed] * 3, axis=2)
            # Move channel dimension to the front (H,W,C) -> (C,H,W)
            processed = np.transpose(processed, (2, 0, 1))
            tensor = torch.tensor(processed.reshape(1, 3, self.image_size[0], self.image_size[1]))

        return tensor.to(self.device)

    def train(self, dataset, batch_size=16, epochs=20, learning_rate=0.0001):
        """Train the model.

        Args:
            dataset: PyTorch Dataset containing (image, label) pairs
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization

        Returns:
            Self for chaining
        """
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Count class distribution
        all_labels = [label.item() for _, label in dataset]
        class_counts = {}
        for label in all_labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        print(f"Class distribution: {class_counts}")

        # Loss and optimizer
        criterion = nn.BCELoss()

        # First freeze all parameters
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the later layers
        for i in range(-5, 0):  # Last 5 blocks
            try:
                for param in self.model.backbone.features[i].parameters():
                    param.requires_grad = True
            except IndexError:
                # In case there are not enough blocks
                pass

        # Unfreeze classifier
        for param in self.model.backbone.classifier.parameters():
            param.requires_grad = True

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, weight_decay=1e-4
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        # Early stopping
        best_val_accuracy = 0
        best_model_state = None
        patience = 5
        no_improvement = 0

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs).squeeze()

                # Handle dimensionality for single item batches
                if outputs.dim() == 0 and labels.dim() == 1:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()

                # Statistics
                train_loss += loss.item() * inputs.size(0)
                pred = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (pred == labels).sum().item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)

                    # Forward pass
                    outputs = self.model(inputs).squeeze()

                    # Handle dimensionality for single item batches
                    if outputs.dim() == 0 and labels.dim() == 1:
                        outputs = outputs.unsqueeze(0)

                    loss = criterion(outputs, labels)

                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    pred = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (pred == labels).sum().item()

            # Calculate epoch statistics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping check
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = {"model": self.model.state_dict(), "epoch": epoch, "val_acc": val_acc}
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state["model"])
            print(
                f"Loaded best model from epoch {best_model_state['epoch'] + 1} "
                f"with validation accuracy {best_model_state['val_acc']:.4f}"
            )

        self.is_trained = True
        return self

    def predict(self, image):
        """Predict intoxication based on image.

        Args:
            image: Input image (NumPy array)

        Returns:
            Tuple of (prediction label, probability)
        """
        if not self.is_trained:
            return "UNKNOWN", 0.5

        # Preprocess the image
        tensor = self.preprocess_image(image)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor).item()

        # Convert to prediction and confidence
        prediction = "INTOXICATED" if output > 0.5 else "SOBER"
        return prediction, output

    def save(self, filepath):
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        # Save model weights and configuration
        model_info = {
            "image_size": self.image_size,
            "infrared_mode": self.infrared_mode,
            "in_channels": self.in_channels,
            "is_trained": self.is_trained,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save configurations separately
        config_path = Path(filepath).with_suffix(".config.joblib")
        joblib.dump(model_info, config_path)

        # Save model weights
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Load the model from disk.

        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load configuration
        config_path = Path(filepath).with_suffix(".config.joblib")
        if os.path.exists(config_path):
            model_info = joblib.load(config_path)
            self.image_size = model_info["image_size"]
            self.infrared_mode = model_info["infrared_mode"]
            self.in_channels = model_info["in_channels"]
            self.is_trained = model_info["is_trained"]

            # Create model with correct configuration
            self.model = IntoxicationCNN(in_channels=self.in_channels).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        self.is_trained = True
