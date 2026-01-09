# Sober-Scan Quick Reference Guide

Complete command reference for all CLI functionalities including detection, training, incremental learning, model management, and dataset operations.

---

## 📋 Main Commands Overview

```bash
sober-scan detect       # Detect drowsiness/intoxication from images
sober-scan train        # Train or update models
sober-scan manage       # Manage models and datasets
  ├── version          # Model version control
  └── dataset          # Dataset management
sober-scan model        # Download/manage model files
```

---

## 🔍 Detection Commands

### Basic Detection

```bash
# Detect drowsiness (default)
sober-scan detect image.jpg

# Detect intoxication
sober-scan detect image.jpg --type intoxication --model cnn

# With visualization
sober-scan detect image.jpg --visualize

# Save output image
sober-scan detect image.jpg -o output.jpg

# Save features as CSV
sober-scan detect image.jpg --save-features
```

### Detection Options

```bash
--type, -t              # Detection type: drowsiness or intoxication
--model, -m             # Model to use: svm, rf, knn, nb, cnn (or path to model file)
--output, -o            # Path to save output image
--color                 # Use color images for intoxication (default: infrared)
--visualize, -v         # Show visualization with landmarks
--save-features         # Save extracted features as CSV
--verbose               # Enable verbose output
```

---

## 🏋️ Training Commands

### Initial Training

```bash
# Train drowsiness detection (SVM)
sober-scan train /path/to/data \
  --model svm \
  --detection-type drowsiness \
  --save-model

# Train intoxication detection (CNN)
sober-scan train /path/to/data \
  --model cnn \
  --detection-type intoxication \
  --epochs 20 \
  --batch-size 32 \
  --augmentation \
  --save-model
```

### All Training Options

```bash
# Required
<data_folder>                      # Path to training images

# Model selection
--model, -m <type>                 # Model: svm, rf, knn, nb, cnn
--detection-type, -d <type>        # drowsiness or intoxication

# Saving
--save-model, -s                   # Save the trained model
--save-path <path>                 # Where to save (default: models/)

# Incremental learning (NEW!)
--incremental                      # Enable incremental updates
--use-incremental-svm              # Use SGDClassifier for SVM
--continue-training                # Continue CNN training from weights
--backup / --no-backup             # Automatic backup (default: on)
--model-path <path>                # Path to existing model to update

# CNN-specific
--epochs <n>                       # Training epochs (default: 10)
--batch-size <n>                   # Batch size (default: 32)
--learning-rate <lr>               # Learning rate (default: 0.001, use 0.0001 for fine-tuning)
--augmentation / --no-augmentation # Data augmentation (default: on)
--infrared / --no-infrared         # Use infrared/grayscale (default: on)
--image-size <n>                   # Input image size (default: 224)
--use-face-detection               # Detect and crop faces before training

# Traditional ML parameters
--svm-c <value>                    # SVM regularization (default: 0.1)
--svm-kernel <type>                # SVM kernel (default: rbf)
--knn-neighbors <n>                # KNN neighbors (default: 5)
--rf-estimators <n>                # Random Forest trees (default: 100)
--rf-max-depth <n>                 # RF max depth (default: 5)

# Evaluation
--test-size <float>                # Test split ratio (default: 0.2)
--visualize, -v                    # Generate ROC/confusion matrix
--use-cross-validation             # Use k-fold cross-validation
--cv-folds <n>                     # Number of CV folds (default: 5)

# Other
--random-seed <n>                  # Random seed (default: 42)
--verbose                          # Verbose output
```

---

## 🎯 Common Tasks

### Adding New Training Data

**Method 1: Fine-Tune Existing Model (Fastest)**

```bash
sober-scan train /path/to/new_data \
  --model cnn \
  --model-path models/intoxication_cnn.pt \
  --continue-training \
  --epochs 5 \
  --backup \
  --save-model
```

**Method 2: Incremental Learning (Traditional ML)**

```bash
# SVM with incremental mode
sober-scan train /path/to/new_data \
  --model svm \
  --use-incremental-svm \
  --incremental \
  --model-path models/drowsiness_svm.joblib \
  --save-model

# Naive Bayes (naturally incremental)
sober-scan train /path/to/new_data \
  --model nb \
  --incremental \
  --model-path models/drowsiness_nb.joblib \
  --save-model
```

**Method 3: Full Retrain on Merged Data**

```bash
# Merge datasets first
sober-scan manage dataset merge old_data,new_data combined

# Train on combined data
sober-scan train /data/merged_combined \
  --model cnn \
  --epochs 20 \
  --backup \
  --save-model
```

---

## 🔄 Model Version Management Commands

```bash
sober-scan manage version list <model_name>
  # List all backup versions of a model
  # Options:
  #   --models-dir <path>    # Custom models directory

sober-scan manage version backup <model_path>
  # Create a manual backup
  # Options:
  #   --notes, -n <text>     # Notes about this backup
  #   --models-dir <path>    # Custom models directory

sober-scan manage version restore <model_name>
  # Restore a model from backup
  # Options:
  #   --version, -v <timestamp>  # Specific version to restore (latest if not specified)
  #   --models-dir <path>        # Custom models directory

sober-scan manage version clean <model_name>
  # Remove old backups, keep recent ones
  # Options:
  #   --keep, -k <n>         # Number of backups to keep (default: 5)
  #   --models-dir <path>    # Custom models directory

sober-scan manage version info <model_name>
  # Get information about model and versions
  # Options:
  #   --models-dir <path>    # Custom models directory
```

---

## 📊 Dataset Management Commands

```bash
sober-scan manage dataset register <name> <path>
  # Register a dataset
  # Options:
  #   --description, -d <text>  # Description of dataset
  #   --type, -t <type>         # Dataset type: intoxication or drowsiness
  #   --data-dir <path>         # Base data directory

sober-scan manage dataset list
  # List all registered datasets
  # Options:
  #   --data-dir <path>      # Base data directory

sober-scan manage dataset merge <names> <output_name>
  # Merge multiple datasets (comma-separated names)
  # Options:
  #   --output, -o <path>    # Output path for merged dataset
  #   --copy / --symlink     # Copy files or create symlinks (default: copy)
  #   --data-dir <path>      # Base data directory

sober-scan manage dataset split <name>
  # Split dataset into train/test sets
  # Options:
  #   --train-ratio, -r <float>  # Train ratio 0-1 (default: 0.8)
  #   --output, -o <path>        # Output directory
  #   --data-dir <path>          # Base data directory

sober-scan manage dataset info <name>
  # Get information about a dataset
  # Options:
  #   --data-dir <path>      # Base data directory

sober-scan manage dataset remove <name>
  # Remove dataset from registry
  # Options:
  #   --delete-files         # Also delete the files
  #   --yes, -y              # Skip confirmation
  #   --data-dir <path>      # Base data directory
```

---

## 📦 Model Download Commands

```bash
sober-scan model list
  # List available and installed models
  # Options:
  #   --available, -a        # Show available models
  #   --installed, -i        # Show installed models
  #   --output, -o <path>    # Custom models directory

sober-scan model download <type>
  # Download pre-trained models
  # Types: dlib-shape-predictor, svm, nb, knn, rf, cnn, all
  # Options:
  #   --output, -o <path>    # Custom output directory
  #   --force, -f            # Force redownload
  #   --verbose, -v          # Show download progress

sober-scan model info [type]
  # Get information about models
  # Options:
  #   --output, -o <path>    # Custom models directory
```

---

## 🚀 Workflow Templates

### Initial Model Training

```bash
# Step 1: Register dataset
sober-scan manage dataset register initial_data /path/to/data

# Step 2: Train model
sober-scan train /path/to/data \
  --model cnn \
  --detection-type intoxication \
  --epochs 20 \
  --augmentation \
  --save-model
```

### Monthly Data Update (Recommended)

```bash
# Step 1: Register new data
sober-scan manage dataset register monthly_jan /path/to/jan_data

# Step 2: Fine-tune model (fast, preserves learning)
sober-scan train /path/to/jan_data \
  --model cnn \
  --model-path models/intoxication_cnn.pt \
  --continue-training \
  --epochs 5 \
  --learning-rate 0.0001 \
  --backup \
  --save-model

# Step 3: Test performance
sober-scan detect test_image.jpg --type intoxication --model cnn

# Step 4: If worse, restore previous version
sober-scan manage version restore intoxication_cnn
```

### Quarterly Full Retrain

```bash
# Step 1: Merge all datasets
sober-scan manage dataset merge \
  initial_data,jan_data,feb_data,mar_data \
  q1_combined

# Step 2: Split for validation
sober-scan manage dataset split q1_combined --train-ratio 0.8

# Step 3: Train on combined data
sober-scan train /data/merged_q1_combined \
  --model cnn \
  --epochs 25 \
  --backup \
  --save-model

# Step 4: Clean old backups
sober-scan manage version clean intoxication_cnn --keep 3
```

---

## 🔍 Decision Tree: Which Method to Use?

```
Do you have new training data?
│
├─ Yes → How much new data? (relative to existing)
│   │
│   ├─ Small (<20%) → Use fine-tuning
│   │   └─ sober-scan train --continue-training --epochs 5-10
│   │
│   ├─ Medium (20-50%) → Options:
│   │   ├─ Fine-tune (faster): --continue-training
│   │   └─ Full retrain (better): merge + train from scratch
│   │
│   └─ Large (>50%) → Full retrain recommended
│       └─ Merge datasets + train from scratch
│
└─ No → Maintain existing model
    └─ Periodic backups: sober-scan manage version backup
```

---

## ⚡ Performance Tips

| Task                                | Recommended Method   | Time         | Accuracy   |
| ----------------------------------- | -------------------- | ------------ | ---------- |
| Small data update (<100 images)     | Fine-tuning          | ★★★★★ Fast   | ★★★★☆ Good |
| Medium data update (100-500 images) | Fine-tuning or Full  | ★★★☆☆ Medium | ★★★★★ Best |
| Large data update (>500 images)     | Full retrain         | ★★☆☆☆ Slow   | ★★★★★ Best |
| Incremental (SVM/NB)                | partial_fit          | ★★★★★ Fast   | ★★★★☆ Good |
| Incremental (KNN/RF)                | update_with_new_data | ★★★☆☆ Medium | ★★★★☆ Good |

---

## 📊 Model Selection Guide

| Model                 | Incremental  | Speed | Accuracy | Best For         |
| --------------------- | ------------ | ----- | -------- | ---------------- |
| **SVM** (incremental) | ✅ True      | ★★★★★ | ★★★★☆    | Frequent updates |
| **SVM** (kernel)      | ❌ No        | ★★★☆☆ | ★★★★★    | Best accuracy    |
| **Naive Bayes**       | ✅ True      | ★★★★★ | ★★★☆☆    | Fast prototyping |
| **KNN**               | ⚠️ Pseudo    | ★★★★☆ | ★★★★☆    | Small datasets   |
| **Random Forest**     | ⚠️ Pseudo    | ★★★☆☆ | ★★★★☆    | Balanced         |
| **CNN**               | ✅ Fine-tune | ★★★☆☆ | ★★★★★    | Image data       |

---

## 🎓 Best Practices

1. **Always backup before retraining**: `--backup`
2. **Use lower learning rate for fine-tuning**: `--learning-rate 0.0001`
3. **Keep data augmentation enabled**: `--augmentation`
4. **Register datasets for tracking**: `sober-scan manage dataset register`
5. **Clean old backups periodically**: `--keep 5`
6. **Test after updates**: Use validation set
7. **Document changes**: Use `--notes` in backups

---

## ❓ Troubleshooting

| Problem                       | Solution                                                 |
| ----------------------------- | -------------------------------------------------------- |
| Model loading fails           | Check if using same model type (incremental vs. regular) |
| Out of memory                 | Reduce `--batch-size` or use `--use-face-detection`      |
| Poor performance after update | Lower learning rate or restore previous version          |
| Dataset merge fails           | Ensure all datasets are registered with same type        |
| Backups taking too much space | Run `clean` command more frequently                      |

---

---

## 🆕 What's New in This Version

### Incremental Learning

- **SVM**: Use `--use-incremental-svm` with `--incremental` for true online learning
- **Naive Bayes**: Use `--incremental` for native incremental updates
- **KNN/RF**: Automatically accumulates data for pseudo-incremental learning
- **CNN**: Use `--continue-training` to fine-tune existing models

### Model Version Control

- Automatic backups before retraining (use `--backup`)
- Track all model versions with timestamps
- Easy restoration to any previous version
- Clean old backups to manage disk space

### Dataset Management

- Register datasets with metadata
- Merge multiple datasets easily
- Split datasets for train/test validation
- Track image counts and descriptions

### Data Augmentation

- Automatic augmentation for CNN (rotation, flip, color jitter, affine)
- Control with `--augmentation` / `--no-augmentation`
- Improves model accuracy by 2-3%

---

---

## 📊 Model Evaluation

All trained models have evaluation plots automatically saved to `models/evaluation/`:

- Confusion matrices (showing classification performance)
- ROC curves (showing discriminative ability)

**Accessing Evaluation Results:**

- Plots are automatically generated when training with `--save-model --visualize` (default: on)
- Location: `models/evaluation/<detection_type>_<model>_<metric>.png`
- View the **MODEL_EVALUATION_REPORT.md** for detailed analysis

---

## 📚 Additional Documentation

- **[README.md](README.md)** - Main project documentation and quick start
- **[CHANGELOG.md](CHANGELOG.md)** - Complete version history and all changes
- **[MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md)** - Detailed model performance analysis
