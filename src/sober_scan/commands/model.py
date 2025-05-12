"""Command for managing model files for Sober-Scan."""

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from sober_scan.config import MODEL_DESCRIPTIONS, MODEL_DIR, MODEL_FILENAMES, MODEL_URLS, ModelType
from sober_scan.utils import create_progress_bar, logger, setup_logger

# Create Typer app for model subcommands
model_app = typer.Typer(
    name="model",
    help="Manage drowsiness and intoxication detection models",
    no_args_is_help=True,
)


def download_file(url: str, destination: Path, chunk_size: int = 8192, verbose: bool = False) -> bool:
    """Download a file from a URL to a destination path.

    Args:
        url: URL to download from
        destination: Path to save the file to
        chunk_size: Size of chunks to download at a time
        verbose: Whether to show download progress

    Returns:
        True if download was successful, False otherwise
    """
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.info().get("Content-Length", 0))
            downloaded = 0

            # Create progress bar if verbose
            if verbose and file_size > 0:
                progress = create_progress_bar(
                    file_size,
                    prefix=f"Downloading {destination.name}:",
                    suffix=f"Complete ({file_size / 1024 / 1024:.1f} MB)",
                )

            with open(destination, "wb") as f:
                while True:
                    buffer = response.read(chunk_size)
                    if not buffer:
                        break

                    f.write(buffer)
                    downloaded += len(buffer)

                    # Update progress bar
                    if verbose and file_size > 0:
                        progress(downloaded)

            return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if verbose:
            typer.echo(f"Error downloading {url}: {e}")
        return False


def extract_bz2(source: Path, destination: Path) -> bool:
    """Extract a bz2 file.

    Args:
        source: Path to the bz2 file
        destination: Path to extract to

    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        import bz2

        with bz2.open(source, "rb") as f_in:
            with open(destination, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Error extracting {source}: {e}")
        return False


@model_app.command("list")
def list_models(
    available: bool = typer.Option(False, "--available", "-a", help="List all available models for download"),
    installed: bool = typer.Option(False, "--installed", "-i", help="List all installed models"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom directory to check for installed models"),
) -> None:
    """List available models for download or installed models."""
    if not available and not installed:
        # Default: show both
        available = True
        installed = True

    # Determine the target directory for installed models
    target_dir = output_dir if output_dir else MODEL_DIR

    if available:
        typer.echo("\nAvailable models for download:")
        typer.echo("------------------------------")
        for model_type in ModelType:
            if model_type == ModelType.ALL:
                continue
            url_status = "" if MODEL_URLS[model_type] != "N/A" else " (needs to be trained locally)"
            typer.echo(f"{model_type.value}: {MODEL_DESCRIPTIONS[model_type]}{url_status}")

    if installed:
        typer.echo(f"\nInstalled models (in {target_dir}):")
        typer.echo("----------------")
        for model_type in ModelType:
            if model_type == ModelType.ALL:
                continue

            model_path = target_dir / MODEL_FILENAMES[model_type]
            status = "✅ Installed" if model_path.exists() else "❌ Not installed"
            typer.echo(f"{model_type.value}: {status}")

    typer.echo("\nTip: Run 'sober-scan model download all' to download all models")


@model_app.command("download")
def download_model(
    model_type: ModelType = typer.Argument(..., help="Type of model to download"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom directory to store downloaded models"),
    force: bool = typer.Option(False, "--force", "-f", help="Force download even if model already exists"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output"),
) -> None:
    """Download models required for intoxication detection."""
    # Setup logger
    setup_logger(verbose)

    # Determine the target directory
    target_dir = output_dir if output_dir else MODEL_DIR

    # Make sure the target directory exists
    try:
        os.makedirs(target_dir, exist_ok=True)
        if verbose:
            typer.echo(f"Using model directory: {target_dir}")
    except Exception as e:
        typer.echo(f"Error creating directory {target_dir}: {e}")
        raise typer.Exit(code=1)

    # Function to download a specific model
    def download_specific_model(mtype: ModelType) -> bool:
        if mtype not in MODEL_URLS:
            typer.echo(f"No download URL specified for {mtype}")
            return False

        url = MODEL_URLS[mtype]

        # Skip if URL is marked as N/A
        if url == "N/A":
            typer.echo(f"No download URL available for {mtype}. This model needs to be trained locally.")
            return False

        filename = MODEL_FILENAMES[mtype]
        destination = target_dir / filename

        # Check if the model already exists
        if destination.exists() and not force:
            typer.echo(f"{mtype} model already exists at {destination}")
            return True

        typer.echo(f"Downloading {mtype} model...")

        # Special handling for dlib shape predictor (bz2 archive)
        if mtype == ModelType.DLIB_SHAPE_PREDICTOR:
            # Download the bz2 file
            temp_path = destination.with_suffix(".dat.bz2")
            success = download_file(url, temp_path, verbose=verbose)

            if success:
                typer.echo(f"Extracting {temp_path.name}...")
                success = extract_bz2(temp_path, destination)
                # Remove the bz2 file
                temp_path.unlink()
        else:
            # Direct download for other models
            success = download_file(url, destination, verbose=verbose)

        if success:
            typer.echo(f"Successfully downloaded {mtype} model to {destination}")
            return True
        else:
            typer.echo(f"Failed to download {mtype} model")
            return False

    # Download requested model(s)
    if model_type == ModelType.ALL:
        # Download all models
        typer.echo("Downloading all models...")
        success = True
        for mtype in ModelType:
            if mtype != ModelType.ALL:
                if not download_specific_model(mtype):
                    # Only count as failure if it's not an N/A URL
                    if MODEL_URLS.get(mtype) != "N/A":
                        success = False

        if success:
            typer.echo("All models downloaded successfully")
        else:
            typer.echo("Some models failed to download")
    else:
        # Download specific model
        download_specific_model(model_type)


@model_app.command("info")
def model_info(
    model_type: Optional[ModelType] = typer.Argument(None, help="Type of model to get info about"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom directory to check for models"),
) -> None:
    """Get information about available models."""
    # Determine the target directory
    target_dir = output_dir if output_dir else MODEL_DIR

    if model_type is None:
        # Show info about all models
        typer.echo("Available Models Information:")
        typer.echo("----------------------------")
        for mtype in ModelType:
            if mtype == ModelType.ALL:
                continue

            model_path = target_dir / MODEL_FILENAMES[mtype]
            status = "Installed" if model_path.exists() else "Not installed"
            url_status = "" if MODEL_URLS[mtype] != "N/A" else " (needs to be trained locally)"

            typer.echo(f"\n{mtype.value}:")
            typer.echo(f"  Description: {MODEL_DESCRIPTIONS[mtype]}{url_status}")
            typer.echo(f"  Status: {status}")
            if model_path.exists():
                typer.echo(f"  Path: {model_path}")
                typer.echo(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        # Show info about specific model
        if model_type == ModelType.ALL:
            typer.echo("Please specify a specific model type for detailed info.")
            return

        model_path = target_dir / MODEL_FILENAMES[model_type]
        status = "Installed" if model_path.exists() else "Not installed"
        url_status = "" if MODEL_URLS[model_type] != "N/A" else " (needs to be trained locally)"

        typer.echo(f"{model_type.value}:")
        typer.echo(f"  Description: {MODEL_DESCRIPTIONS[model_type]}{url_status}")
        typer.echo(f"  Status: {status}")
        if model_path.exists():
            typer.echo(f"  Path: {model_path}")
            typer.echo(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

        typer.echo(f"\nDownload URL: {MODEL_URLS[model_type]}")
