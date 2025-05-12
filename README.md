<div align="center">
   <a href="https://github.com/Sang-Buster/Sober-Scan">
      <img src="https://raw.githubusercontent.com/Sang-Buster/Sober-Scan/refs/heads/main/README.assets/logo.png" width=40% alt="logo">
   </a>
   <h1>Sober Scan</h1>
   <a href="https://deepwiki.com/Sang-Buster/Sober-Scan"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
   <a href="https://pypi.org/project/sober-scan/"><img src="https://img.shields.io/pypi/v/sober-scan" alt="PyPI"></a>
   <a href="https://github.com/Sang-Buster/Sober-Scan/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Sang-Buster/Sober-Scan" alt="License"></a>
   <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
   <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
   <a href="https://pepy.tech/project/sober-scan"><img src="https://img.shields.io/pypi/dm/sober-scan" alt="Downloads"></a>
   <a href="https://github.com/Sang-Buster/Sober-Scan/commits/main"><img src="https://img.shields.io/github/last-commit/Sang-Buster/Sober-Scan" alt="Last Commit"></a>
   <h6><small>A CLI tool that detects alcohol intoxication from facial images.</small></h6>
   <p><b>#Alcohol Intoxication &emsp; #Facial Recognition &emsp; #CLI</b></p>
</div>

---

[▶️ Watch Demo Video]()

<div align="center">
  <h2>🚀 Getting Started</h2>
</div>

It is recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create a virtual environment and pip install the following package.

```bash
pip install sober-scan
```

To run the application, simply type:

```bash
sober-scan
# or
sober-scan --help
```

---

<div align="center">
  <h2>👨‍💻 Development Setup</h2>
</div>

1. **Clone the repository and navigate to project folder:**
   ```bash
   git clone https://github.com/Sang-Buster/Sober-Scan
   cd Sober-Scan
   ```

2. **Install uv first:**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   ```powershell
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create a virtual environment at `Sober-Scan/.venv/`:**
   ```bash
   uv venv --python 3.10
   ```

4. **Activate the virtual environment:**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   ```

   ```powershell
   # Windows
   .venv\Scripts\activate
   ```

5. **Install the required packages:**
   ```bash
   uv pip install -e .
   ```

6. **Install ruff and pre-commit:**
   ```bash
   uv pip install ruff pre-commit
   ```
   - `ruff` is a super fast Python linter and formatter.
   - `pre-commit` helps maintain code quality by running automated checks before commits are made.

7. **Install git hooks:**
   ```bash
   pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
   ```

   These hooks perform different checks at various stages:
   - `commit-msg`: Ensures commit messages follow the conventional format
   - `pre-commit`: Runs Ruff linting and formatting checks before each commit
   - `pre-push`: Performs final validation before pushing to remote
  
8. **Code Linting:**
   ```bash
   ruff check
   ruff check --fix
   ruff check --select I
   ruff check --select I --fix
   ruff format
   ```

9.  **Run the application:**
   ```bash
   uv run src/sober_scan/app.py
   ```

---

<div align="center">
  <h2>📝 File Structure</h2>
</div>

```text
📂Sober-Scan
 ┣ 📂src                         // Source Code
 ┃ ┗ 📦sober_scan                  // Python package
 ┃ ┃ ┣ 📂commands                      // Command line interface
 ┃ ┃ ┃ ┣ 📄detect.py
 ┃ ┃ ┃ ┣ 📄model.py
 ┃ ┃ ┃ ┗ 📄train.py
 ┃ ┃ ┣ 📂models                        // Model files
 ┃ ┃ ┃ ┣ 📄cnn.py
 ┃ ┃ ┃ ┣ 📄knn.py
 ┃ ┃ ┃ ┣ 📄nb.py
 ┃ ┃ ┃ ┣ 📄rf.py
 ┃ ┃ ┃ ┗ 📄svm.py
 ┃ ┃ ┣ 📂tests                        // Test files
 ┃ ┃ ┃ ┗ 📄test_cli.py
 ┃ ┃ ┣ 📄cli.py                      // CLI interface
 ┃ ┃ ┣ 📄config.py                   // Configuration
 ┃ ┃ ┣ 📄feature_extraction.py       // Feature extraction
 ┃ ┃ ┗ 📄utils.py                    // Utility functions
 ┣ 📄.gitignore                  // Git ignore patterns (env, cache, database)
 ┣ 📄.pre-commit-config.yaml     // Pre-commit hooks (ruff, commit message)
 ┣ 📄.pre-commit_msg_template.py // Commit message format validator
 ┣ 📄.python-version             // Python version
 ┣ 📄LICENSE                     // MIT License
 ┣ 📄README.md                   // Project documentation
 ┣ 📄pyproject.toml              // Project configuration
 ┗ 📄uv.lock                     // Lock file
 ```