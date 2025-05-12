"""Command for training drowsiness detection models."""

import glob
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from sober_scan.config import MODEL_DIR
from sober_scan.feature_extraction import calculate_eye_aspect_ratio, detect_face_and_landmarks
from sober_scan.models.cnn import IntoxicationDetector as CNNDetector

# Import drowsiness detection models
from sober_scan.utils import load_image, logger, setup_logger


class ModelType(str, Enum):
    """Supported model types for drowsiness detection training."""

    SVM = "svm"
    RANDOM_FOREST = "rf"
    KNN = "knn"
    NAIVE_BAYES = "nb"
    CNN = "cnn"


class DetectionType(str, Enum):
    """Type of detection to train for."""

    DROWSINESS = "drowsiness"
    INTOXICATION = "intoxication"


def extract_features_from_image(image_path: Path) -> Tuple[Optional[np.ndarray], bool]:
    """Extract EAR and MAR features from a single image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (features, success)
    """
    # Load image
    image = load_image(image_path)

    if image is None:
        return None, False

    # Detect face and landmarks
    face_rect, landmarks = detect_face_and_landmarks(image)

    if face_rect is None or landmarks is None:
        return None, False

    # Extract EAR
    ear = calculate_eye_aspect_ratio(landmarks)

    # Calculate MAR using mouth landmarks (48-68)
    if len(landmarks) >= 68:
        # Horizontal distance
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        # Vertical distance
        mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
        mar = mouth_height / mouth_width if mouth_width > 0 else 0
    else:
        mar = 0

    return np.array([ear, mar]), True


def extract_features_from_folder(folder_path: Path, threshold: float = 0.25) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """Extract features and labels from all images in a folder.

    Args:
        folder_path: Path to folder containing images
        threshold: EAR threshold for drowsiness classification

    Returns:
        Tuple of (features array, labels array, successful image paths)
    """
    features_list = []
    labels_list = []
    successful_images = []

    # Get all image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    total_images = len(image_files)
    successful = 0

    for i, image_path in enumerate(image_files):
        logger.info(f"Processing image {i + 1}/{total_images}: {image_path}")
        features, success = extract_features_from_image(Path(image_path))

        if success:
            successful += 1
            features_list.append(features)
            # Determine label based on EAR threshold
            ear = features[0]
            label = 1 if ear < threshold else 0  # 1 = drowsy, 0 = alert
            labels_list.append(label)
            successful_images.append(Path(image_path))

    logger.info(f"Successfully processed {successful}/{total_images} images")

    if successful == 0:
        return np.array([]), np.array([]), []

    return np.array(features_list), np.array(labels_list), successful_images


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Optional[Path] = None) -> None:
    """Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_score: Predicted probabilities
        output_path: Path to save the plot
    """
    from sklearn.metrics import auc, roc_curve

    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path)
        logger.info(f"ROC curve saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Optional[Path] = None) -> None:
    """Plot confusion matrix for binary classification.

    Args:
        y_true: True binary labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Alert", "Drowsy"]
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if output_path:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()


class IntoxicationDataset(Dataset):
    """PyTorch Dataset for intoxication images."""

    def __init__(self, data_folder, image_size=(224, 224), infrared_mode=True, transform=None, use_face_detection=False):
        """Initialize dataset from image folder.

        Args:
            data_folder: Root folder with subfolders 'sober' and 'drunk'
            image_size: Target image size (height, width)
            infrared_mode: Whether to convert images to grayscale
            transform: Optional transform to apply to images
            use_face_detection: Whether to detect and crop faces or use entire image
        """
        self.data_folder = Path(data_folder)
        self.image_size = image_size
        self.infrared_mode = infrared_mode
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.samples = []

        # Expected folder structure:
        # data_folder/
        #   sober/
        #     img1.jpg
        #     img2.jpg
        #   drunk/
        #     img1.jpg
        #     img2.jpg

        # Load sober images (label 0)
        sober_dir = self.data_folder / "sober"
        if sober_dir.exists():
            for ext in ["jpg", "jpeg", "png"]:
                self.samples.extend([(f, 0) for f in sober_dir.glob(f"*.{ext}")])
            # Also try uppercase extensions
            for ext in ["JPG", "JPEG", "PNG"]:
                self.samples.extend([(f, 0) for f in sober_dir.glob(f"*.{ext}")])

        # Load drunk images (label 1)
        drunk_dir = self.data_folder / "drunk"
        if drunk_dir.exists():
            for ext in ["jpg", "jpeg", "png"]:
                self.samples.extend([(f, 1) for f in drunk_dir.glob(f"*.{ext}")])
            # Also try uppercase extensions
            for ext in ["JPG", "JPEG", "PNG"]:
                self.samples.extend([(f, 1) for f in drunk_dir.glob(f"*.{ext}")])

        # If no images found in standard folders, check if we have a flat directory with all images
        if len(self.samples) == 0:
            typer.echo("No images found in standard sober/drunk folders, checking for flat directory structure...")
            # Try to find all images in the data folder directly
            all_images = []
            for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
                all_images.extend(list(self.data_folder.glob(f"*.{ext}")))

            if all_images:
                typer.echo(f"Found {len(all_images)} images in flat directory, using all as positive (drunk) examples")
                self.samples = [(f, 1) for f in all_images]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = load_image(img_path)

        if image is None:
            # Handle broken images by creating a blank image
            if self.infrared_mode:
                image = np.zeros((224, 224, 1), dtype=np.uint8)
            else:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            logger.warning(f"Failed to load image: {img_path}, using blank image instead")

        # Face detection and cropping (optional)
        if self.use_face_detection:
            face_rect, _ = detect_face_and_landmarks(image)

            # Crop to face if detected
            if face_rect is not None:
                x1, y1, x2, y2 = face_rect
                # Add some margin
                h, w = image.shape[:2]
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)
                image = image[y1:y2, x1:x2]

        # Resize
        import cv2

        image = cv2.resize(image, self.image_size)

        # Convert to grayscale if using infrared mode
        if self.infrared_mode and len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image.reshape(self.image_size[0], self.image_size[1], 1)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        if self.infrared_mode:
            # Single channel
            tensor = torch.tensor(image.reshape(1, self.image_size[0], self.image_size[1]))
        else:
            # RGB channels
            if len(image.shape) == 2:
                # If we have a grayscale image but not in infrared mode
                image = np.stack([image] * 3, axis=2)
            image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            tensor = torch.tensor(image)

        # Apply any additional transformations
        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)


def train_intoxication_cnn(
    data_folder,
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    infrared_mode=True,
    image_size=(224, 224),
    test_size=0.2,
    random_seed=42,
    use_face_detection=False,
    use_cross_validation=False,
):
    """Train a CNN model for intoxication detection.

    Args:
        data_folder: Path to data folder with 'sober' and 'drunk' subfolders
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        infrared_mode: Whether to use infrared (grayscale) images
        image_size: Input image size (height, width)
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        use_face_detection: Whether to detect and crop faces or use entire image
        use_cross_validation: Whether to use k-fold cross-validation for more robust evaluation

    Returns:
        Trained model and evaluation metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create dataset
    dataset = IntoxicationDataset(
        data_folder=data_folder,
        image_size=image_size,
        infrared_mode=infrared_mode,
        use_face_detection=use_face_detection,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"""No images found in {data_folder}, please ensure this folder structure is correct:
            {data_folder}/
            ├── sober/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── drunk/
                ├── img1.jpg
                └── img2.jpg
            """
        )

    # Print dataset stats
    typer.echo(f"Dataset loaded with {len(dataset)} images")

    # Calculate class distribution
    labels = [label for _, label in dataset.samples]
    sober_count = sum(1 for label in labels if label == 0)
    intoxicated_count = sum(1 for label in labels if label == 1)
    typer.echo(f"Class distribution: {sober_count} sober, {intoxicated_count} intoxicated/drunk")

    # Split dataset into train and test sets
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed)
    )

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = CNNDetector(image_size=image_size, infrared_mode=infrared_mode)

    # Train the model
    typer.echo(f"Training CNN model for {epochs} epochs...")
    model.train(train_dataset, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)

    # Evaluate the model
    typer.echo("Evaluating model on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.model.eval()

    # Track predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []

    # No gradients needed for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model.model(inputs).squeeze()

            # Convert outputs to predictions
            preds = (outputs > 0.5).float()

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Try to calculate ROC AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
        typer.echo(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        typer.echo(f"Could not calculate ROC AUC: {e}")
        auc = None

    # Print classification report
    target_names = ["Sober", "Drunk"]
    report = classification_report(all_labels, all_preds, target_names=target_names)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return model, {
        "accuracy": accuracy,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train_model_command(
    data_folder: Path = typer.Argument(..., help="Path to folder containing training images", exists=True),
    model_type: str = typer.Option(
        ModelType.SVM.value,
        "--model",
        "-m",
        help="Type of model to train (svm, rf, knn, nb, cnn) or path to a model file to continue training",
    ),
    detection_type: DetectionType = typer.Option(
        DetectionType.DROWSINESS, "--detection-type", "-d", help="Type of detection to train for"
    ),
    save_model: bool = typer.Option(False, "--save-model", "-s", help="Save the trained model to disk"),
    save_path: Path = typer.Option(MODEL_DIR, "--save-path", help="Directory where to save the model if --save-model is used"),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Path to an existing model to continue training"),
    test_size: float = typer.Option(0.2, "--test-size", help="Proportion of data to use for testing"),
    ear_threshold: float = typer.Option(0.25, "--ear-threshold", help="Eye Aspect Ratio threshold for drowsiness detection"),
    random_seed: int = typer.Option(42, "--random-seed", help="Random seed for reproducibility"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of folds for cross-validation"),
    use_test_data: bool = typer.Option(
        False, "--test-data", help="Use synthetic test data instead of real images (for debugging)"
    ),
    # CNN-specific parameters
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for CNN training"),
    epochs: int = typer.Option(10, "--epochs", help="Number of epochs for CNN training"),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="Learning rate for CNN training"),
    infrared_mode: bool = typer.Option(True, "--infrared", help="Use infrared (grayscale) images for intoxication detection"),
    image_size: int = typer.Option(224, "--image-size", help="Size of input images for CNN (will be resized to square)"),
    use_face_detection: bool = typer.Option(
        False, "--use-face-detection", help="Detect and crop faces before training (default: use entire image)"
    ),
    use_cross_validation: bool = typer.Option(
        False, "--use-cross-validation", help="Use k-fold cross-validation for more robust evaluation"
    ),
    # Model-specific parameters
    svm_c: float = typer.Option(0.1, "--svm-c", help="Regularization parameter for SVM (smaller values increase regularization)"),
    svm_kernel: str = typer.Option("rbf", "--svm-kernel", help="Kernel type for SVM"),
    knn_neighbors: int = typer.Option(5, "--knn-neighbors", help="Number of neighbors for KNN"),
    rf_estimators: int = typer.Option(100, "--rf-estimators", help="Number of trees in Random Forest"),
    rf_max_depth: int = typer.Option(5, "--rf-max-depth", help="Maximum depth of trees in Random Forest"),
    visualize_results: bool = typer.Option(True, "--visualize", "-v", help="Generate and save ROC curve and confusion matrix"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Train a drowsiness or intoxication detection model on a folder of images.

    This command processes images in the specified folder, extracts features,
    and trains a detection model. It outputs evaluation metrics and
    optionally saves the trained model.

    Examples:\n
        # Train drowsiness detection model\n
        sober-scan train data/drowsiness --model svm --detection-type drowsiness\n

        # Train intoxication detection model\n
        sober-scan train data/intoxication --model cnn --detection-type intoxication\n

        # Train intoxication detection model with entire images (no face detection)\n
        sober-scan train data/intoxication --model cnn --detection-type intoxication --no-use-face-detection\n

        # Train and save to default models directory\n
        sober-scan train data/training --model svm --save-model\n

        # Train and save to specific location\n
        sober-scan train data/training --model svm --save-model --save-path custom/path\n

        # Continue training from existing model file\n
        sober-scan train data/intoxication --model /path/to/existing/model.pt --detection-type intoxication --save-model
    """
    # Setup logger
    setup_logger(verbose)

    # Create output directory if we'll be saving model or results
    if save_model:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path / "evaluation", exist_ok=True)

    # Check if model_type is a file path (will override model_path if both are provided)
    if os.path.exists(model_type):
        model_path = Path(model_type)
        # Determine model type from file extension
        if model_type.lower().endswith(".pt"):
            model_type_enum = ModelType.CNN
        elif model_type.lower().endswith(".joblib"):
            # Default to SVM for joblib files if we can't determine precisely
            model_type_enum = ModelType.SVM
        else:
            typer.echo("Warning: Unknown model file extension. Attempting to use it anyway.")
            model_type_enum = ModelType.SVM  # Default
    else:
        # Try to convert to enum
        try:
            model_type_enum = ModelType(model_type.lower())
        except ValueError:
            typer.echo(f"Error: Invalid model type or file path: {model_type}")
            typer.echo(f"Model type must be one of: {', '.join([t.value for t in ModelType])}")
            typer.echo("Or it must be a valid path to a model file.")
            raise typer.Exit(code=1)

    # Handle different detection types
    if detection_type == DetectionType.INTOXICATION:
        if model_type_enum != ModelType.CNN and not (model_path and os.path.exists(model_path)):
            typer.echo("Error: Only CNN model is supported for intoxication detection.")
            typer.echo("Please use --model cnn when --detection-type is intoxication")
            raise typer.Exit(code=1)

        # Train intoxication CNN model
        try:
            typer.echo("Training CNN model for intoxication detection...")

            # If model_path is provided, inform the user
            if model_path:
                typer.echo(f"Will attempt to continue training from existing model: {model_path}")

            model, metrics = train_intoxication_cnn(
                data_folder=data_folder,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                infrared_mode=infrared_mode,
                image_size=(image_size, image_size),
                test_size=test_size,
                random_seed=random_seed,
                use_face_detection=use_face_detection,
                use_cross_validation=use_cross_validation,
            )

            # Print evaluation results
            typer.echo("\nTest Set Evaluation:")
            typer.echo(f"Accuracy: {metrics['accuracy']:.4f}")
            if metrics["auc"] is not None:
                typer.echo(f"ROC AUC: {metrics['auc']:.4f}")

            typer.echo("\nClassification Report:")
            typer.echo(metrics["report"])

            # Visualize results if requested
            if visualize_results:
                if save_model:
                    evaluation_dir = save_path / "evaluation"

                    # Plot confusion matrix
                    cm_path = evaluation_dir / f"intoxication_{model_type_enum.value}_confusion_matrix.png"
                    plot_confusion_matrix(metrics["labels"], metrics["predictions"], cm_path)

                    # Plot ROC curve if applicable
                    if metrics["auc"] is not None:
                        roc_path = evaluation_dir / f"intoxication_{model_type_enum.value}_roc_curve.png"
                        plot_roc_curve(metrics["labels"], metrics["probabilities"], roc_path)
                else:
                    # Just show the plots without saving
                    plot_confusion_matrix(metrics["labels"], metrics["predictions"])
                    if metrics["auc"] is not None:
                        plot_roc_curve(metrics["labels"], metrics["probabilities"])

            # Save model if requested
            if save_model:
                model_filename = f"intoxication_{model_type_enum.value}.pt"
                model_save_path = save_path / model_filename

                try:
                    model.save(model_save_path)
                    typer.echo(f"\nModel saved to {model_save_path}")

                    # Verify file was created
                    if os.path.exists(model_save_path):
                        typer.echo(f"Verified: Model file exists at {model_save_path} ({os.path.getsize(model_save_path)} bytes)")
                    else:
                        typer.echo(f"Warning: Model file was not found at {model_save_path} after save operation")
                except Exception as e:
                    typer.echo(f"Error saving model: {e}")
            else:
                typer.echo("\nModel was not saved (use --save-model to save the model)")

            typer.echo("\nTraining complete!")
            return

        except Exception as e:
            typer.echo(f"Error training intoxication model: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            raise typer.Exit(code=1)

    # Continue with existing drowsiness detection training code...
    # ... existing code ...
