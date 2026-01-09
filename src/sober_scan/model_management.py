"""Model version management and backup utilities."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from sober_scan.utils import logger


class ModelVersionManager:
    """Manages model versions and backups."""

    def __init__(self, models_dir: Path):
        """Initialize the model version manager.

        Args:
            models_dir: Directory containing models
        """
        self.models_dir = Path(models_dir)
        self.backups_dir = self.models_dir / "backups"
        self.versions_file = self.models_dir / "model_versions.json"

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.backups_dir, exist_ok=True)

        # Load or initialize versions metadata
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict:
        """Load version metadata from JSON file."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading versions file: {e}")
                return {}
        return {}

    def _save_versions(self):
        """Save version metadata to JSON file."""
        try:
            with open(self.versions_file, "w") as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions file: {e}")

    def backup_model(self, model_path: Path, notes: Optional[str] = None) -> Optional[Path]:
        """Create a backup of a model file.

        Args:
            model_path: Path to the model file to backup
            notes: Optional notes about this version

        Returns:
            Path to the backup file or None if backup failed
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_path.stem
            backup_name = f"{model_name}_v{timestamp}{model_path.suffix}"
            backup_path = self.backups_dir / backup_name

            # Copy the model file
            shutil.copy2(model_path, backup_path)

            # Also backup associated config files
            config_path = model_path.with_suffix(".config.joblib")
            if config_path.exists():
                backup_config_path = backup_path.with_suffix(".config.joblib")
                shutil.copy2(config_path, backup_config_path)

            # Update version metadata
            if model_name not in self.versions:
                self.versions[model_name] = {"backups": [], "current_version": None}

            backup_info = {
                "timestamp": timestamp,
                "backup_path": str(backup_path),
                "original_path": str(model_path),
                "notes": notes or "",
                "file_size": os.path.getsize(backup_path),
            }

            self.versions[model_name]["backups"].append(backup_info)
            self.versions[model_name]["current_version"] = timestamp
            self._save_versions()

            logger.info(f"Model backed up to: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None

    def restore_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Restore a model from backup.

        Args:
            model_name: Name of the model (without extension)
            version: Specific version timestamp to restore, or None for latest

        Returns:
            True if restoration was successful
        """
        if model_name not in self.versions:
            logger.error(f"No backups found for model: {model_name}")
            return False

        backups = self.versions[model_name]["backups"]
        if not backups:
            logger.error(f"No backups available for model: {model_name}")
            return False

        # Find the backup to restore
        if version is None:
            # Get the latest backup
            backup_info = backups[-1]
        else:
            # Find specific version
            backup_info = next((b for b in backups if b["timestamp"] == version), None)
            if backup_info is None:
                logger.error(f"Version {version} not found for model: {model_name}")
                return False

        try:
            backup_path = Path(backup_info["backup_path"])
            original_path = Path(backup_info["original_path"])

            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Backup the current model before restoring
            if original_path.exists():
                self.backup_model(original_path, notes="Pre-restore backup")

            # Restore the model
            shutil.copy2(backup_path, original_path)

            # Restore config file if it exists
            backup_config = backup_path.with_suffix(".config.joblib")
            if backup_config.exists():
                original_config = original_path.with_suffix(".config.joblib")
                shutil.copy2(backup_config, original_config)

            logger.info(f"Model restored from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring model: {e}")
            return False

    def list_versions(self, model_name: str) -> list:
        """List all available versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of version information dictionaries
        """
        if model_name not in self.versions:
            return []

        return self.versions[model_name]["backups"]

    def clean_old_backups(self, model_name: str, keep_last: int = 5) -> int:
        """Remove old backups, keeping only the most recent ones.

        Args:
            model_name: Name of the model
            keep_last: Number of recent backups to keep

        Returns:
            Number of backups deleted
        """
        if model_name not in self.versions:
            return 0

        backups = self.versions[model_name]["backups"]
        if len(backups) <= keep_last:
            return 0

        # Get backups to delete (all except the last N)
        backups_to_delete = backups[:-keep_last]
        deleted_count = 0

        for backup_info in backups_to_delete:
            try:
                backup_path = Path(backup_info["backup_path"])
                if backup_path.exists():
                    backup_path.unlink()

                # Also delete config file if it exists
                config_path = backup_path.with_suffix(".config.joblib")
                if config_path.exists():
                    config_path.unlink()

                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting backup {backup_path}: {e}")

        # Update metadata
        self.versions[model_name]["backups"] = backups[-keep_last:]
        self._save_versions()

        logger.info(f"Deleted {deleted_count} old backups for {model_name}")
        return deleted_count

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model and its versions.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information or None
        """
        if model_name not in self.versions:
            return None

        return {
            "model_name": model_name,
            "current_version": self.versions[model_name]["current_version"],
            "total_backups": len(self.versions[model_name]["backups"]),
            "backups": self.versions[model_name]["backups"],
        }
