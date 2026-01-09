"""Command for managing models and datasets."""

from pathlib import Path
from typing import Optional

import typer

from sober_scan.config import DATA_DIR, MODEL_DIR
from sober_scan.dataset_management import DatasetManager
from sober_scan.model_management import ModelVersionManager

# Create Typer apps for management subcommands
manage_app = typer.Typer(
    name="manage",
    help="Manage models, datasets, and versions",
    no_args_is_help=True,
)

model_version_app = typer.Typer(
    name="version",
    help="Model version management commands",
    no_args_is_help=True,
)

dataset_app = typer.Typer(
    name="dataset",
    help="Dataset management commands",
    no_args_is_help=True,
)

# Add sub-apps to main manage app
manage_app.add_typer(model_version_app, name="version")
manage_app.add_typer(dataset_app, name="dataset")


# Model version management commands
@model_version_app.command("list")
def list_model_versions(
    model_name: str = typer.Argument(..., help="Name of the model (without extension)"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
) -> None:
    """List all available versions of a model."""
    version_manager = ModelVersionManager(models_dir)
    versions = version_manager.list_versions(model_name)

    if not versions:
        typer.echo(f"No versions found for model: {model_name}")
        return

    typer.echo(f"\nVersions for model '{model_name}':")
    typer.echo("-" * 80)

    for i, version in enumerate(versions, 1):
        typer.echo(f"\n{i}. Version: {version['timestamp']}")
        typer.echo(f"   Backup Path: {version['backup_path']}")
        typer.echo(f"   File Size: {version['file_size'] / 1024 / 1024:.2f} MB")
        if version['notes']:
            typer.echo(f"   Notes: {version['notes']}")


@model_version_app.command("backup")
def backup_model(
    model_path: Path = typer.Argument(..., help="Path to the model file to backup", exists=True),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Notes about this backup"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
) -> None:
    """Create a backup of a model file."""
    version_manager = ModelVersionManager(models_dir)
    backup_path = version_manager.backup_model(model_path, notes=notes)

    if backup_path:
        typer.echo(f"✓ Model backed up successfully to: {backup_path}")
    else:
        typer.echo("✗ Failed to backup model")
        raise typer.Exit(code=1)


@model_version_app.command("restore")
def restore_model(
    model_name: str = typer.Argument(..., help="Name of the model (without extension)"),
    version: Optional[str] = typer.Option(None, "--version", "-v", help="Version timestamp to restore (latest if not specified)"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
) -> None:
    """Restore a model from backup."""
    version_manager = ModelVersionManager(models_dir)

    if version_manager.restore_model(model_name, version):
        typer.echo(f"✓ Model '{model_name}' restored successfully")
    else:
        typer.echo(f"✗ Failed to restore model '{model_name}'")
        raise typer.Exit(code=1)


@model_version_app.command("clean")
def clean_old_backups(
    model_name: str = typer.Argument(..., help="Name of the model (without extension)"),
    keep_last: int = typer.Option(5, "--keep", "-k", help="Number of recent backups to keep"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
) -> None:
    """Remove old backups, keeping only the most recent ones."""
    version_manager = ModelVersionManager(models_dir)
    deleted = version_manager.clean_old_backups(model_name, keep_last=keep_last)

    typer.echo(f"Deleted {deleted} old backup(s) for '{model_name}'")
    typer.echo(f"Kept the {keep_last} most recent backups")


@model_version_app.command("info")
def model_version_info(
    model_name: str = typer.Argument(..., help="Name of the model (without extension)"),
    models_dir: Path = typer.Option(MODEL_DIR, "--models-dir", help="Directory containing models"),
) -> None:
    """Get information about a model and its versions."""
    version_manager = ModelVersionManager(models_dir)
    info = version_manager.get_model_info(model_name)

    if not info:
        typer.echo(f"No information found for model: {model_name}")
        return

    typer.echo(f"\nModel: {info['model_name']}")
    typer.echo(f"Current Version: {info['current_version']}")
    typer.echo(f"Total Backups: {info['total_backups']}")

    if info['backups']:
        typer.echo("\nMost Recent Backups:")
        for backup in info['backups'][-5:]:  # Show last 5
            typer.echo(f"  - {backup['timestamp']}: {backup.get('notes', 'No notes')}")


# Dataset management commands
@dataset_app.command("register")
def register_dataset(
    dataset_name: str = typer.Argument(..., help="Name identifier for the dataset"),
    dataset_path: Path = typer.Argument(..., help="Path to the dataset directory", exists=True),
    description: str = typer.Option("", "--description", "-d", help="Description of the dataset"),
    dataset_type: str = typer.Option("intoxication", "--type", "-t", help="Type of dataset (intoxication or drowsiness)"),
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
) -> None:
    """Register a dataset in the dataset registry."""
    dataset_manager = DatasetManager(data_dir)

    if dataset_manager.register_dataset(dataset_name, dataset_path, description, dataset_type):
        typer.echo(f"✓ Dataset '{dataset_name}' registered successfully")
    else:
        typer.echo(f"✗ Failed to register dataset '{dataset_name}'")
        raise typer.Exit(code=1)


@dataset_app.command("list")
def list_datasets(
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
) -> None:
    """List all registered datasets."""
    dataset_manager = DatasetManager(data_dir)
    datasets = dataset_manager.list_datasets()

    if not datasets:
        typer.echo("No datasets registered")
        return

    typer.echo("\nRegistered Datasets:")
    typer.echo("=" * 80)

    for name, info in datasets.items():
        typer.echo(f"\n{name}:")
        typer.echo(f"  Path: {info['path']}")
        typer.echo(f"  Type: {info['type']}")
        typer.echo(f"  Description: {info['description']}")
        typer.echo(f"  Image Count: {info['image_count']}")


@dataset_app.command("merge")
def merge_datasets(
    dataset_names: str = typer.Argument(..., help="Comma-separated list of dataset names to merge"),
    output_name: str = typer.Argument(..., help="Name for the merged dataset"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path for the merged dataset"),
    copy_files: bool = typer.Option(True, "--copy/--symlink", help="Copy files or create symlinks"),
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
) -> None:
    """Merge multiple datasets into a single dataset."""
    dataset_manager = DatasetManager(data_dir)

    # Parse dataset names
    names_list = [name.strip() for name in dataset_names.split(",")]

    typer.echo(f"Merging datasets: {', '.join(names_list)}")

    merged_path = dataset_manager.merge_datasets(names_list, output_name, output_path, copy_files)

    if merged_path:
        typer.echo(f"✓ Datasets merged successfully to: {merged_path}")
    else:
        typer.echo("✗ Failed to merge datasets")
        raise typer.Exit(code=1)


@dataset_app.command("split")
def split_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to split"),
    train_ratio: float = typer.Option(0.8, "--train-ratio", "-r", help="Ratio of training data (0-1)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for split datasets"),
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
) -> None:
    """Split a dataset into train and test sets."""
    dataset_manager = DatasetManager(data_dir)

    typer.echo(f"Splitting dataset '{dataset_name}' with train ratio: {train_ratio}")

    result = dataset_manager.split_dataset(dataset_name, train_ratio, output_dir)

    if result:
        train_path, test_path = result
        typer.echo("✓ Dataset split successfully:")
        typer.echo(f"  Train: {train_path}")
        typer.echo(f"  Test: {test_path}")
    else:
        typer.echo("✗ Failed to split dataset")
        raise typer.Exit(code=1)


@dataset_app.command("info")
def dataset_info(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
) -> None:
    """Get information about a specific dataset."""
    dataset_manager = DatasetManager(data_dir)
    info = dataset_manager.get_dataset_info(dataset_name)

    if not info:
        typer.echo(f"Dataset '{dataset_name}' not found")
        return

    typer.echo(f"\nDataset: {dataset_name}")
    typer.echo(f"Path: {info['path']}")
    typer.echo(f"Type: {info['type']}")
    typer.echo(f"Description: {info['description']}")
    typer.echo(f"Image Count: {info['image_count']}")
    typer.echo(f"Registered: {info['registered_date']}")


@dataset_app.command("remove")
def remove_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    delete_files: bool = typer.Option(False, "--delete-files", help="Also delete the dataset files"),
    data_dir: Path = typer.Option(DATA_DIR, "--data-dir", help="Base data directory"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a dataset from the registry."""
    if not confirm:
        if not typer.confirm(f"Are you sure you want to remove dataset '{dataset_name}'?"):
            typer.echo("Cancelled")
            return

    dataset_manager = DatasetManager(data_dir)

    if dataset_manager.remove_dataset(dataset_name, delete_files):
        typer.echo(f"✓ Dataset '{dataset_name}' removed successfully")
    else:
        typer.echo(f"✗ Failed to remove dataset '{dataset_name}'")
        raise typer.Exit(code=1)
