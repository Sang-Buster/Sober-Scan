"""Video processing and analysis for intoxication detection."""

import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sober_scan.config import DEFAULT_MODELS, BACLevel
from sober_scan.feature_extraction import extract_features
from sober_scan.utils import create_progress_bar, logger


def extract_frames(
    video_path: Union[str, Path], max_frames: int = 0, frame_interval: int = 5
) -> Generator[np.ndarray, None, None]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (0 for all)
        frame_interval: Interval between extracted frames

    Yields:
        Extracted video frames
    """
    video_path = Path(video_path)

    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return

    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        frame_count = 0
        frames_read = 0

        while True:
            # Read frame
            ret, frame = cap.read()

            if not ret:
                break

            # Process only every Nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
                frames_read += 1

                if max_frames > 0 and frames_read >= max_frames:
                    break

            frame_count += 1

        # Release video capture
        cap.release()

    except Exception as e:
        logger.error(f"Error extracting frames from video: {e}")


def process_video(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    frame_interval: int = 5,
    max_frames: int = 0,
    batch_size: int = 16,
    verbose: bool = False,
) -> List[Dict[str, Union[np.ndarray, Dict[str, float]]]]:
    """Process a video file for intoxication detection.

    Args:
        video_path: Path to the video file
        output_path: Path to save the processed video
        frame_interval: Interval between processed frames
        max_frames: Maximum number of frames to process (0 for all)
        batch_size: Number of frames to process as a batch
        verbose: Whether to display progress

    Returns:
        List of dictionaries containing frames and extracted features
    """
    video_path = Path(video_path)

    # Open video to get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create VideoWriter if output path is provided
    output_writer = None
    if output_path:
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Extract frames
    frames_to_process = total_frames // frame_interval
    if max_frames > 0:
        frames_to_process = min(frames_to_process, max_frames)

    # Create progress bar
    if verbose:
        progress = create_progress_bar(frames_to_process, prefix=f"Processing {video_path.name}:", suffix="")

    # Process frames
    results = []
    for i, frame in enumerate(extract_frames(video_path, max_frames, frame_interval)):
        # Extract facial features
        features = extract_features(frame)

        # Save result
        results.append(
            {
                "frame": frame,
                "features": features,
                "frame_index": i * frame_interval,
            }
        )

        # Write to output video if needed
        if output_writer is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_writer.write(frame_bgr)

        # Update progress
        if verbose:
            progress(i + 1)

    # Release output writer
    if output_writer is not None:
        output_writer.release()

    logger.info(f"Processed {len(results)} frames from {video_path}")
    return results


class TemporalModel(nn.Module):
    """Temporal model for video-based intoxication detection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        bidirectional: bool = True,
        model_type: str = "lstm",
    ):
        """Initialize the temporal model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent layers
            num_classes: Number of output classes
            bidirectional: Whether to use bidirectional RNN
            model_type: Type of model ('lstm' or '3dcnn')
        """
        super(TemporalModel, self).__init__()

        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        if model_type == "lstm":
            # LSTM for sequential feature processing
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )

            # Fully connected layers for classification
            fc_input_size = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(fc_input_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
            )

        elif model_type == "3dcnn":
            # 3D CNN for spatiotemporal feature learning
            self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
            self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

            # Calculate output size after convolutions
            self.fc_input_size = self._get_conv_output_size((16, 3, 224, 224))

            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.fc_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers."""
        bs, c, t, h, w = shape
        x = torch.zeros(bs, c, t, h, w)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor
                For LSTM: (batch_size, seq_length, input_size)
                For 3DCNN: (batch_size, channels, frames, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        if self.model_type == "lstm":
            # Pass through LSTM
            rnn_out, _ = self.rnn(x)

            # Use the last time step output
            out = rnn_out[:, -1, :]

            # Pass through fully connected layers
            out = self.fc(out)

        elif self.model_type == "3dcnn":
            # Pass through 3D CNN
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)

            # Flatten and pass through fully connected layers
            x = x.view(x.size(0), -1)
            out = self.fc(x)

        return out


class VideoAnalyzer:
    """Analyzer for video-based intoxication detection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sequence_length: int = 16,
        feature_size: int = 30,
        model_type: str = "lstm",
    ):
        """Initialize the video analyzer.

        Args:
            model_path: Path to the temporal model
            sequence_length: Number of frames in a sequence
            feature_size: Size of feature vector
            model_type: Type of temporal model ('lstm' or '3dcnn')
        """
        self.model_path = model_path or DEFAULT_MODELS["video"]["path"]
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize and load model
        self._load_model()

    def _load_model(self) -> bool:
        """Load the temporal model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.model_type == "lstm":
                self.model = TemporalModel(
                    input_size=self.feature_size, hidden_size=128, num_layers=2, num_classes=4, model_type="lstm"
                )
            elif self.model_type == "3dcnn":
                self.model = TemporalModel(
                    input_size=0,  # Not used for 3DCNN
                    model_type="3dcnn",
                )
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

    def extract_feature_sequences(self, video_results: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract feature sequences from video processing results.

        Args:
            video_results: List of dictionaries with frames and features

        Returns:
            List of feature sequences
        """
        sequences = []

        # Check if there are enough frames
        if len(video_results) < self.sequence_length:
            logger.warning("Not enough frames for a complete sequence")
            return sequences

        # Create sequences with sliding window
        for i in range(len(video_results) - self.sequence_length + 1):
            sequence = []

            for j in range(i, i + self.sequence_length):
                # Check if features were extracted for this frame
                if not video_results[j]["features"]:
                    break

                # Extract feature values in a consistent order
                feature_values = []
                for key in sorted(video_results[j]["features"].keys()):
                    feature_values.append(video_results[j]["features"][key])

                sequence.append(feature_values)

            # Only add complete sequences
            if len(sequence) == self.sequence_length:
                sequences.append(np.array(sequence))

        return sequences

    def predict_sequence(self, feature_sequence: np.ndarray) -> Tuple[BACLevel, float]:
        """Predict intoxication level from a feature sequence.

        Args:
            feature_sequence: Array of shape (sequence_length, feature_size)

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot predict.")
            return BACLevel.SOBER, 0.0

        try:
            # Convert to tensor and add batch dimension
            if self.model_type == "lstm":
                # Pad or truncate feature sequence if needed
                if feature_sequence.shape[1] < self.feature_size:
                    padded = np.zeros((feature_sequence.shape[0], self.feature_size))
                    padded[:, : feature_sequence.shape[1]] = feature_sequence
                    feature_sequence = padded
                elif feature_sequence.shape[1] > self.feature_size:
                    feature_sequence = feature_sequence[:, : self.feature_size]

                sequence_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0)
                sequence_tensor = sequence_tensor.to(self.device)

            elif self.model_type == "3dcnn":
                # For 3DCNN, we need the actual video frames
                logger.error("3DCNN prediction from feature sequence not implemented")
                return BACLevel.SOBER, 0.0

            # Run inference
            with torch.no_grad():
                output = self.model(sequence_tensor)
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

    def analyze_video(
        self, video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, verbose: bool = False
    ) -> Tuple[BACLevel, float]:
        """Analyze a video for intoxication detection.

        Args:
            video_path: Path to the video file
            output_path: Path to save the processed video
            verbose: Whether to display progress

        Returns:
            Tuple of (BAC level enum, confidence score)
        """
        # Process video frames
        video_results = process_video(video_path, output_path, verbose=verbose)

        if not video_results:
            logger.error("No frames processed from video")
            return BACLevel.SOBER, 0.0

        # Extract feature sequences
        sequences = self.extract_feature_sequences(video_results)

        if not sequences:
            logger.error("No complete sequences extracted from video")
            return BACLevel.SOBER, 0.0

        # Predict on each sequence
        predictions = []
        confidences = []

        for sequence in sequences:
            bac_level, confidence = self.predict_sequence(sequence)
            predictions.append(bac_level.value)
            confidences.append(confidence)

        # Aggregate predictions
        if not predictions:
            return BACLevel.SOBER, 0.0

        # Get the most common prediction
        pred_counts = np.bincount(predictions)
        most_common_pred = np.argmax(pred_counts)
        confidence = float(pred_counts[most_common_pred] / len(predictions))

        return BACLevel(most_common_pred), confidence
