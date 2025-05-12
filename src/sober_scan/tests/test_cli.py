"""Tests for the CLI module."""

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
    assert "detect-image" in result.stdout
    assert "detect-video" in result.stdout


def test_detect_image_help():
    """Test the detect-image help command."""
    result = runner.invoke(app, ["detect-image", "--help"])
    assert result.exit_code == 0
    assert "Detect intoxication from a facial image" in result.stdout


def test_detect_video_help():
    """Test the detect-video help command."""
    result = runner.invoke(app, ["detect-video", "--help"])
    assert result.exit_code == 0
    assert "Detect intoxication from a video" in result.stdout


def test_detect_image_no_image():
    """Test the detect-image command with no image."""
    result = runner.invoke(app, ["detect-image"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stdout


def test_detect_video_no_video():
    """Test the detect-video command with no video."""
    result = runner.invoke(app, ["detect-video"])
    assert result.exit_code != 0
    assert "Missing argument" in result.stdout
