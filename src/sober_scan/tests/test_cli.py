"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import patch

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
    assert "A tool that detects alcohol intoxication from facial images." in result.stdout
    assert "detect" in result.stdout
    assert "model" in result.stdout
    # verify detect-video is not in the output
    assert "detect-video" not in result.stdout


def test_detect_help():
    """Test the detect help command."""
    result = runner.invoke(app, ["detect", "--help"])
    assert result.exit_code == 0
    assert "Detect intoxication from a facial image" in result.stdout
    assert "This command only works with static image files" in result.stdout


def test_detect_no_image():
    """Test the detect command with no image."""
    result = runner.invoke(app, ["detect"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stdout or "Error" in result.stdout


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
    assert "Manage intoxication detection models" in result.stdout


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
    import numpy as np

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
