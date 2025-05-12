"""Command-line interface for Sober-Scan."""

import importlib.metadata
import sys
from typing import Optional

import typer

from sober_scan.commands.detect import detect_image_command
from sober_scan.commands.model import model_app
from sober_scan.commands.train import train_model_command

# Create Typer app
app = typer.Typer(
    name="sober-scan",
    help="A tool that detects drowsiness and intoxication from facial images.",
    add_completion=False,
    no_args_is_help=True,  # Show help when no arguments are provided
    context_settings={"help_option_names": ["--help", "-h"]},  # Add -h as alias for --help
)


def version_callback(value: bool) -> None:
    """Print the version of the package and exit."""
    if value:
        try:
            version = importlib.metadata.version("Sober-Scan")
            typer.echo(f"Sober-Scan version: {version}")
        except importlib.metadata.PackageNotFoundError:
            from sober_scan import __version__

            typer.echo(f"Sober-Scan version: {__version__}")
        raise typer.Exit()


# Simple handler for -h flag
if len(sys.argv) == 2 and sys.argv[1] == "-h":
    sys.argv[1] = "--help"


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        help="Show the version and exit.",
        is_eager=True,
    ),
) -> None:
    """Sober-Scan: Detect drowsiness and intoxication from facial images.

    This CLI tool provides commands to analyze facial images for
    signs of drowsiness and intoxication, using both traditional computer vision
    techniques and machine learning methods.
    """
    return


# Add commands
app.command(name="detect", no_args_is_help=True)(detect_image_command)
app.command(name="train", no_args_is_help=True)(train_model_command)

# Add model commands as a subcommand group
app.add_typer(model_app, name="model")


if __name__ == "__main__":
    app()
