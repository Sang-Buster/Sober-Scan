"""Dataset management utilities for organizing and merging training data."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sober_scan.utils import logger


class DatasetManager:
    """Manages training datasets with support for merging and versioning."""

    def __init__(self, base_data_dir: Path):
        """Initialize the dataset manager.

        Args:
            base_data_dir: Base directory for datasets
        """
        self.base_data_dir = Path(base_data_dir)
        self.metadata_file = self.base_data_dir / "dataset_metadata.json"

        # Create directories if they don't exist
        os.makedirs(self.base_data_dir, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata file: {e}")
                return {"datasets": {}}
        return {"datasets": {}}

    def _save_metadata(self):
        """Save dataset metadata to JSON file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata file: {e}")

    def register_dataset(
        self,
        dataset_name: str,
        dataset_path: Path,
        description: str = "",
        dataset_type: str = "intoxication",
    ) -> bool:
        """Register a dataset in the metadata.

        Args:
            dataset_name: Name identifier for the dataset
            dataset_path: Path to the dataset directory
            description: Description of the dataset
            dataset_type: Type of dataset ('intoxication' or 'drowsiness')

        Returns:
            True if registration was successful
        """
        try:
            dataset_path = Path(dataset_path)

            # Count images in the dataset
            image_count = self._count_images(dataset_path, dataset_type)

            dataset_info = {
                "path": str(dataset_path),
                "description": description,
                "type": dataset_type,
                "image_count": image_count,
                "registered_date": str(Path(dataset_path).stat().st_mtime),
            }

            self.metadata["datasets"][dataset_name] = dataset_info
            self._save_metadata()

            logger.info(f"Dataset '{dataset_name}' registered with {image_count} images")
            return True

        except Exception as e:
            logger.error(f"Error registering dataset: {e}")
            return False

    def _count_images(self, dataset_path: Path, dataset_type: str) -> Dict[str, int]:
        """Count images in a dataset.

        Args:
            dataset_path: Path to the dataset
            dataset_type: Type of dataset

        Returns:
            Dictionary with image counts
        """
        counts = {}

        if dataset_type == "intoxication":
            # Count sober and drunk images
            sober_dir = dataset_path / "sober"
            drunk_dir = dataset_path / "drunk"

            counts["sober"] = len(list(sober_dir.glob("*.jpg"))) + len(list(sober_dir.glob("*.png"))) if sober_dir.exists() else 0
            counts["drunk"] = len(list(drunk_dir.glob("*.jpg"))) + len(list(drunk_dir.glob("*.png"))) if drunk_dir.exists() else 0
            counts["total"] = counts["sober"] + counts["drunk"]

        elif dataset_type == "drowsiness":
            # Count all images
            counts["total"] = len(list(dataset_path.glob("*.jpg"))) + len(list(dataset_path.glob("*.png")))

        return counts

    def merge_datasets(
        self,
        dataset_names: List[str],
        output_name: str,
        output_path: Optional[Path] = None,
        copy_files: bool = True,
    ) -> Optional[Path]:
        """Merge multiple datasets into a single dataset.

        Args:
            dataset_names: List of dataset names to merge
            output_name: Name for the merged dataset
            output_path: Path for the merged dataset (defaults to base_data_dir/merged_<output_name>)
            copy_files: If True, copy files; if False, create symlinks

        Returns:
            Path to the merged dataset or None if merge failed
        """
        # Validate that all datasets exist
        for name in dataset_names:
            if name not in self.metadata["datasets"]:
                logger.error(f"Dataset '{name}' not found in registry")
                return None

        # Get dataset type (all should be the same)
        dataset_type = self.metadata["datasets"][dataset_names[0]]["type"]
        for name in dataset_names[1:]:
            if self.metadata["datasets"][name]["type"] != dataset_type:
                logger.error("Cannot merge datasets of different types")
                return None

        # Set output path
        if output_path is None:
            output_path = self.base_data_dir / f"merged_{output_name}"

        try:
            os.makedirs(output_path, exist_ok=True)

            if dataset_type == "intoxication":
                # Create sober and drunk directories
                sober_out = output_path / "sober"
                drunk_out = output_path / "drunk"
                os.makedirs(sober_out, exist_ok=True)
                os.makedirs(drunk_out, exist_ok=True)

                # Merge datasets
                for dataset_name in dataset_names:
                    dataset_path = Path(self.metadata["datasets"][dataset_name]["path"])

                    # Copy/link sober images
                    sober_dir = dataset_path / "sober"
                    if sober_dir.exists():
                        self._copy_or_link_images(sober_dir, sober_out, f"{dataset_name}_", copy_files)

                    # Copy/link drunk images
                    drunk_dir = dataset_path / "drunk"
                    if drunk_dir.exists():
                        self._copy_or_link_images(drunk_dir, drunk_out, f"{dataset_name}_", copy_files)

            elif dataset_type == "drowsiness":
                # Copy all images to output directory
                for dataset_name in dataset_names:
                    dataset_path = Path(self.metadata["datasets"][dataset_name]["path"])
                    self._copy_or_link_images(dataset_path, output_path, f"{dataset_name}_", copy_files)

            # Register the merged dataset
            self.register_dataset(
                output_name,
                output_path,
                description=f"Merged dataset from: {', '.join(dataset_names)}",
                dataset_type=dataset_type,
            )

            logger.info(f"Successfully merged datasets into: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return None

    def _copy_or_link_images(self, source_dir: Path, dest_dir: Path, prefix: str = "", copy: bool = True):
        """Copy or symlink images from source to destination.

        Args:
            source_dir: Source directory
            dest_dir: Destination directory
            prefix: Prefix to add to filenames
            copy: If True, copy files; if False, create symlinks
        """
        for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
            for img_file in source_dir.glob(f"*.{ext}"):
                dest_file = dest_dir / f"{prefix}{img_file.name}"

                # Avoid overwriting existing files
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{prefix}{img_file.stem}_{counter}{img_file.suffix}"
                    counter += 1

                if copy:
                    shutil.copy2(img_file, dest_file)
                else:
                    os.symlink(img_file.absolute(), dest_file)

    def list_datasets(self) -> Dict[str, Dict]:
        """List all registered datasets.

        Returns:
            Dictionary of dataset information
        """
        return self.metadata["datasets"]

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information dictionary or None
        """
        return self.metadata["datasets"].get(dataset_name)

    def remove_dataset(self, dataset_name: str, delete_files: bool = False) -> bool:
        """Remove a dataset from the registry.

        Args:
            dataset_name: Name of the dataset
            delete_files: If True, also delete the dataset files

        Returns:
            True if removal was successful
        """
        if dataset_name not in self.metadata["datasets"]:
            logger.error(f"Dataset '{dataset_name}' not found")
            return False

        try:
            dataset_path = Path(self.metadata["datasets"][dataset_name]["path"])

            if delete_files and dataset_path.exists():
                shutil.rmtree(dataset_path)
                logger.info(f"Deleted dataset files at: {dataset_path}")

            del self.metadata["datasets"][dataset_name]
            self._save_metadata()

            logger.info(f"Removed dataset '{dataset_name}' from registry")
            return True

        except Exception as e:
            logger.error(f"Error removing dataset: {e}")
            return False

    def split_dataset(
        self,
        dataset_name: str,
        train_ratio: float = 0.8,
        output_dir: Optional[Path] = None,
    ) -> Optional[Tuple[Path, Path]]:
        """Split a dataset into train and test sets.

        Args:
            dataset_name: Name of the dataset to split
            train_ratio: Ratio of training data (0-1)
            output_dir: Output directory for split datasets

        Returns:
            Tuple of (train_path, test_path) or None if split failed
        """
        if dataset_name not in self.metadata["datasets"]:
            logger.error(f"Dataset '{dataset_name}' not found")
            return None

        try:
            import random

            dataset_info = self.metadata["datasets"][dataset_name]
            dataset_path = Path(dataset_info["path"])
            dataset_type = dataset_info["type"]

            if output_dir is None:
                output_dir = self.base_data_dir / f"split_{dataset_name}"

            train_dir = output_dir / "train"
            test_dir = output_dir / "test"

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            if dataset_type == "intoxication":
                # Split sober and drunk images separately
                for class_name in ["sober", "drunk"]:
                    class_dir = dataset_path / class_name
                    if not class_dir.exists():
                        continue

                    # Get all images
                    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    random.shuffle(images)

                    # Split
                    split_idx = int(len(images) * train_ratio)
                    train_images = images[:split_idx]
                    test_images = images[split_idx:]

                    # Create class directories
                    train_class_dir = train_dir / class_name
                    test_class_dir = test_dir / class_name
                    os.makedirs(train_class_dir, exist_ok=True)
                    os.makedirs(test_class_dir, exist_ok=True)

                    # Copy images
                    for img in train_images:
                        shutil.copy2(img, train_class_dir / img.name)
                    for img in test_images:
                        shutil.copy2(img, test_class_dir / img.name)

            elif dataset_type == "drowsiness":
                # Get all images
                images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
                random.shuffle(images)

                # Split
                split_idx = int(len(images) * train_ratio)
                train_images = images[:split_idx]
                test_images = images[split_idx:]

                # Copy images
                for img in train_images:
                    shutil.copy2(img, train_dir / img.name)
                for img in test_images:
                    shutil.copy2(img, test_dir / img.name)

            # Register split datasets
            self.register_dataset(f"{dataset_name}_train", train_dir, f"Training split of {dataset_name}", dataset_type)
            self.register_dataset(f"{dataset_name}_test", test_dir, f"Test split of {dataset_name}", dataset_type)

            logger.info(f"Dataset split successfully: train={train_dir}, test={test_dir}")
            return train_dir, test_dir

        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            return None
