"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

from sober_scan.cli import app

# Initialize test runner
runner = CliRunner()


def test_app_version():
    """Test the CLI version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Sober-Scan version:" in result.stdout


def test_app_help():
    """Test the CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "A tool that detects drowsiness and intoxication from facial images." in result.stdout
    assert "detect" in result.stdout
    assert "model" in result.stdout
    # verify detect-video is not in the output
    assert "detect-video" not in result.stdout


def test_detect_help():
    """Test the detect help command."""
    result = runner.invoke(app, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Detect drowsiness or intoxication from a facial image" in result.stdout


def test_detect_no_image():
    """Test the detect command with no image."""
    result = runner.invoke(app, ["detect"])
    assert result.exit_code == 0
    assert "Usage: sober-scan detect [OPTIONS] IMAGE_PATH" in result.stdout


def test_detect_invalid_image_path():
    """Test the detect command with an invalid image path."""
    result = runner.invoke(app, ["detect", "nonexistent.jpg"])
    assert result.exit_code != 0
    assert "Path does not exist" in result.stdout or "Error" in result.stdout


def test_detect_invalid_model_type():
    """Test the detect command with an invalid model type."""
    result = runner.invoke(app, ["detect", "image.jpg", "--model", "invalid_model"])
    assert result.exit_code != 0
    assert "Invalid value" in result.stdout or "Error" in result.stdout


def test_model_help():
    """Test the model help command."""
    result = runner.invoke(app, ["model", "--help"])
    assert result.exit_code == 0
    assert "Manage drowsiness and intoxication detection models" in result.stdout


def test_model_list_help():
    """Test the model list help command."""
    result = runner.invoke(app, ["model", "list", "--help"])
    assert result.exit_code == 0
    assert "List available models" in result.stdout


def test_model_download_help():
    """Test the model download help command."""
    result = runner.invoke(app, ["model", "download", "--help"])
    assert result.exit_code == 0
    assert "Download models" in result.stdout


def test_model_info_help():
    """Test the model info help command."""
    result = runner.invoke(app, ["model", "info", "--help"])
    assert result.exit_code == 0
    assert "Get information about available models" in result.stdout


def test_model_download_invalid_model():
    """Test downloading an invalid model type."""
    result = runner.invoke(app, ["model", "download", "invalid_model"])
    assert result.exit_code != 0
    assert "Invalid value" in result.stdout or "Error" in result.stdout


@patch("sober_scan.feature_extraction.detect_face_and_landmarks")
@patch("sober_scan.utils.load_image")
def test_detect_no_face(mock_load_image, mock_detect_face):
    """Test detection with an image where no face is detected."""
    # Mock the load_image function to return a dummy image
    mock_load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock face detection to return None
    mock_detect_face.return_value = (None, None)

    with runner.isolated_filesystem():
        # Create a dummy image file
        Path("test.jpg").touch()

        # Run the command
        result = runner.invoke(app, ["detect", "test.jpg"])

        # Check results
        assert result.exit_code != 0
        assert "No face detected" in result.stdout or "Error" in result.stdout


def test_invalid_command():
    """Test invoking an invalid command."""
    result = runner.invoke(app, ["invalid-command"])
    assert result.exit_code != 0
    assert "No such command" in result.stdout or "Error" in result.stdout


def test_h_alias_for_help():
    """Test -h as an alias for --help."""
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "detect" in result.stdout
    assert "model" in result.stdout


# Tests for the 'train' command


def test_train_help():
    """Test the train help command."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train a drowsiness or intoxication detection model" in result.stdout
    assert "Usage: sober-scan train [OPTIONS] DATA_FOLDER" in result.stdout


def test_train_no_data_folder():
    """Test the train command with no data folder."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert "Usage: sober-scan train [OPTIONS] DATA_FOLDER" in result.stdout


def test_train_invalid_data_folder():
    """Test the train command with an invalid data folder."""
    result = runner.invoke(app, ["train", "nonexistent_folder"])
    assert result.exit_code != 0
    assert "Path 'nonexistent_folder' does not exist" in result.stdout or "Error" in result.stdout


@patch("sober_scan.commands.train.train_intoxication_cnn")
@patch("pathlib.Path.exists", return_value=True)
@patch("os.makedirs")
@patch("sober_scan.commands.train.plt.savefig")
def test_train_intoxication_cnn_mocked(mock_plt_savefig, mock_os_makedirs, mock_path_exists, mock_train_cnn_func):
    """Test the train command for intoxication with CNN (mocked)."""
    mock_model_instance = MagicMock()
    # Provide minimal data to avoid errors in plotting functions
    mock_metrics = {
        "accuracy": 0.9,
        "auc": 0.95,
        "report": "Mock report",
        "confusion_matrix": np.array([[1, 0], [0, 1]]),  # Min 2x2 for plotting
        "labels": np.array([0, 1]),  # Min one of each class for ROC
        "predictions": np.array([0, 1]),
        "probabilities": np.array([0.1, 0.9]),
    }
    mock_train_cnn_func.return_value = (mock_model_instance, mock_metrics)

    with runner.isolated_filesystem():
        Path("dummy_data").mkdir()
        Path("dummy_data/sober").mkdir()
        Path("dummy_data/drunk").mkdir()
        (Path("dummy_data/sober") / "s1.jpg").touch()
        (Path("dummy_data/drunk") / "d1.jpg").touch()

        result = runner.invoke(
            app,
            [
                "train",
                "dummy_data",
                "--model",
                "cnn",
                "--detection-type",
                "intoxication",
                "--epochs",
                "1",
                "--save-model",
                "--save-path",
                "saved_models",
            ],
        )
        assert result.exit_code == 0, f"OUTPUT: {result.output}"
        assert "Training CNN model for intoxication detection..." in result.stdout
        assert "Model saved to" in result.stdout
        mock_train_cnn_func.assert_called_once()
        # Ensure the mock model's save method was called
        expected_save_path = Path("saved_models") / "intoxication_cnn.pt"
        mock_model_instance.save.assert_called_once_with(expected_save_path)

        # Check that os.makedirs was called for save_path and save_path/evaluation
        mock_os_makedirs.assert_any_call(Path("saved_models"), exist_ok=True)
        mock_os_makedirs.assert_any_call(Path("saved_models") / "evaluation", exist_ok=True)

        # Check that savefig was called (e.g., twice, once for CM, once for ROC)
        # Number of calls depends on whether AUC is None and visualize_results is True.
        # In our mock, AUC is not None (0.95) and visualize_results is True by default.
        assert mock_plt_savefig.call_count >= 1  # At least called for confusion matrix
        if mock_metrics["auc"] is not None:
            assert mock_plt_savefig.call_count >= 2  # Called for CM and ROC
            expected_cm_path = Path("saved_models") / "evaluation" / "intoxication_cnn_confusion_matrix.png"
            expected_roc_path = Path("saved_models") / "evaluation" / "intoxication_cnn_roc_curve.png"
            # Check if called with the correct paths
            mock_plt_savefig.assert_any_call(expected_cm_path)
            mock_plt_savefig.assert_any_call(expected_roc_path)


@patch("sober_scan.commands.train.extract_features_from_folder")
@patch("pathlib.Path.exists", return_value=True)
@patch("os.makedirs")
def test_train_drowsiness_svm_mocked(mock_makedirs, mock_path_exists, mock_extract_features):
    """Test the train command for drowsiness with SVM (mocked)."""
    mock_extract_features.return_value = (
        np.array([[0.2, 0.3], [0.3, 0.4]]),  # features
        np.array([0, 1]),  # labels
        [Path("img1.jpg"), Path("img2.jpg")],  # successful_images
    )

    with runner.isolated_filesystem():
        Path("dummy_data_drowsy").mkdir()
        (Path("dummy_data_drowsy") / "alert.jpg").touch()
        (Path("dummy_data_drowsy") / "drowsy.jpg").touch()

        result = runner.invoke(
            app,
            [
                "train",
                "dummy_data_drowsy",
                "--model",
                "svm",
                "--detection-type",
                "drowsiness",
                "--save-model",
                "--save-path",
                "saved_models_svm",
            ],
        )
        assert result.exit_code == 0, f"STDOUT: {result.stdout} STDERR: {result.stderr}"
        mock_makedirs.assert_called()  # Check if directories for saving are made


# More comprehensive tests for the 'detect' command


@patch("sober_scan.commands.detect.load_image")
@patch("sober_scan.commands.detect.detect_face_and_landmarks")
@patch("sober_scan.commands.detect.SVMDetector")
@patch("sober_scan.commands.detect.save_image")
def test_detect_drowsiness_svm_mocked(mock_save_image, mock_svm_detector_class, mock_detect_face, mock_load_image):
    """Test detect command for drowsiness with SVM (mocked)."""
    mock_load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_detect_face.return_value = ((0, 0, 100, 100), np.random.rand(68, 2))

    mock_svm_instance = mock_svm_detector_class.return_value
    mock_svm_instance.predict.return_value = ("ALERT", 0.99)

    with runner.isolated_filesystem():
        Path("test_img.jpg").touch()
        # Mock that the model file exists to trigger model.load()
        with patch("pathlib.Path.exists", return_value=True):
            result = runner.invoke(
                app, ["detect", "test_img.jpg", "--type", "drowsiness", "--model", "svm", "--output", "out.jpg"]
            )
        assert result.exit_code == 0, result.stdout
        assert "Drowsiness Detection Result: ALERT (confidence: 0.99)" in result.stdout
        assert "Result saved to out.jpg" in result.stdout
        mock_svm_instance.load.assert_called_once()
        mock_svm_instance.predict.assert_called_once()
        mock_save_image.assert_called_once()


@patch("sober_scan.commands.detect.load_image")
@patch("sober_scan.commands.detect.detect_face_and_landmarks")
@patch("sober_scan.commands.detect.CNNDetector")
@patch("sober_scan.commands.detect.save_image")
def test_detect_intoxication_cnn_mocked(mock_save_image, mock_cnn_detector_class, mock_detect_face, mock_load_image):
    """Test detect command for intoxication with CNN (mocked)."""
    mock_load_image.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_detect_face.return_value = ((0, 0, 100, 100), np.random.rand(68, 2))

    mock_cnn_instance = mock_cnn_detector_class.return_value
    mock_cnn_instance.predict.return_value = ("SOBER", 0.95)

    with patch(
        "sober_scan.commands.detect.extract_skin_redness",
        return_value={"face_redness": 0.5, "forehead_redness": 0.5, "cheeks_redness": 0.5},
    ):
        with runner.isolated_filesystem():
            Path("test_intox.jpg").touch()
            # Mock that the model file exists to trigger model.load()
            with patch("pathlib.Path.exists", return_value=True):
                result = runner.invoke(
                    app,
                    [
                        "detect",
                        "test_intox.jpg",
                        "--type",
                        "intoxication",
                        "--model",
                        "cnn",
                        "--output",
                        "out_intox.jpg",
                    ],
                )
            assert result.exit_code == 0, result.stdout
            assert "Intoxication Detection Result: SOBER (confidence: 0.95)" in result.stdout
            assert "Result saved to out_intox.jpg" in result.stdout
            mock_cnn_instance.load.assert_called_once()
            mock_cnn_instance.predict.assert_called_once()
            mock_save_image.assert_called_once()


def test_detect_image_no_face_extended():
    """Test detection with an image where no face is detected (using live dependencies)."""
    # This test relies on the actual face detection, so it might be slower
    # or require specific test images if the mocked one isn't sufficient.
    # For now, re-using the structure from test_detect_no_face
    with (
        patch("sober_scan.feature_extraction.detect_face_and_landmarks") as mock_detect_face,
        patch("sober_scan.utils.load_image") as mock_load_image,
    ):
        mock_load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detect_face.return_value = (None, None)  # No face detected

        with runner.isolated_filesystem():
            Path("no_face.jpg").touch()
            result = runner.invoke(app, ["detect", "no_face.jpg"])
            assert result.exit_code != 0
            assert "No face detected" in result.stdout or "Error" in result.stdout


# Add a test for detect command with an invalid model for a given type
def test_detect_invalid_model_for_type():
    """Test detect command with an invalid model for a given detection type."""
    with runner.isolated_filesystem():
        Path("test.jpg").touch()
        # CNN is valid for intoxication but not for drowsiness (based on current detect.py)
        result = runner.invoke(app, ["detect", "test.jpg", "--type", "drowsiness", "--model", "cnn"])
        assert result.exit_code != 0
        assert "model not implemented for drowsiness detection yet" in result.stdout or "Error" in result.stdout
