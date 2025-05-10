"""Command for detecting intoxication from videos."""

from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import typer

from sober_scan.utils import draw_intoxication_result, logger, setup_logger
from sober_scan.video_analysis import VideoAnalyzer


class VideoModelType(str, Enum):
    """Supported video model types for intoxication detection."""

    LSTM = "lstm"
    CNN_3D = "3dcnn"


def detect_video_command(
    video_path: Path = typer.Argument(..., help="Path to the input video", exists=True),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save the output video with results"
    ),
    model_type: VideoModelType = typer.Option(
        VideoModelType.LSTM, "--model", "-m", help="Model type to use for detection"
    ),
    model_path: Optional[Path] = typer.Option(None, "--model-path", help="Custom path to the model file"),
    frame_interval: int = typer.Option(5, "--frame-interval", "-f", help="Process every Nth frame"),
    max_frames: int = typer.Option(0, "--max-frames", help="Maximum number of frames to process (0 = all)"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Visualize results during processing"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
) -> None:
    """Detect intoxication from a video of a person's face.

    This command processes a video to detect signs of alcohol intoxication over time,
    analyzing facial features in a sequence of frames. It then classifies the intoxication
    level using temporal modeling (LSTM or 3D CNN).
    """
    # Setup logger
    setup_logger(verbose)

    # Create video analyzer
    logger.info(f"Initializing {model_type} video analyzer")
    analyzer = VideoAnalyzer(model_path=str(model_path) if model_path else None, model_type=model_type.value)

    # Analyze video
    logger.info(f"Analyzing video: {video_path}")
    bac_level, confidence = analyzer.analyze_video(
        video_path=video_path, output_path=output_path if output_path else None, verbose=verbose
    )

    # Print results
    result_str = f"Detected intoxication level: {bac_level.name} (confidence: {confidence:.2f})"
    typer.echo(result_str)

    # Visualize the result if requested
    if visualize and output_path and output_path.exists():
        # Open the output video for visualization
        cap = cv2.VideoCapture(str(output_path))

        if not cap.isOpened():
            typer.echo("Error: Failed to open output video for visualization")
            return

        # Display the result on each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for consistency with our utils
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw the result
            result_frame = draw_intoxication_result(frame_rgb, bac_level, confidence)

            # Convert back to BGR for display
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

            # Display the frame
            cv2.imshow("Video Analysis Result", result_frame_bgr)

            # Break on 'q' key press
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        # Release and close
        cap.release()
        cv2.destroyAllWindows()

    # Return success
    return
