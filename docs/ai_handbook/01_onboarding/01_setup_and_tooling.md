# **filename: docs/ai_handbook/01_onboarding/01_setup_and_tooling.md**

# **Onboarding: Environment Setup & Tooling**

This guide provides the essential steps to set up the development environment. It is the single source of truth for tooling and environment management.

## **1. Requirements**

* **Python:** Version 3.10 or higher.
* **Package Manager:** **UV**. This project explicitly uses uv for all dependency management. Do not use pip, conda, or poetry.
* **Virtual Environment:** All dependencies must be installed in a .venv/ directory at the project root.

## **2. Quick Start Commands**

These are the most common commands for development.

### **Install All Dependencies**

This command syncs your virtual environment with the dependencies specified in pyproject.toml, including development tools.

uv sync --extra dev

### **Run Tests**

Execute the test suite using pytest.

uv run pytest tests/ -v

### **Format & Lint Code**

Automatically format and lint the codebase to ensure consistency.

# Run the linter and formatter
uv run ruff check . --fix
uv run ruff format .

### **Run a Script**

Execute any Python script within the project's virtual environment.

uv run python path/to/your/script.py

## **4. Environment Variables & Secrets**

- Copy `.env.template` to `.env.local` and populate API keys (OpenAI, Upstage, Weights & Biases, GitHub tokens, etc.).
- Keep `.env` under version control; it only contains non-sensitive defaults such as `AGENT_CONTEXT_LOG_DIR` and an optional `AGENT_CONTEXT_LOG_LABEL`.
- The context logging helpers automatically read `AGENT_CONTEXT_LOG_DIR` and `AGENT_CONTEXT_LOG_LABEL`. Override these in `.env.local` or export them in your shell before starting a session.
- Never commit real API keys—`.env.local` is ignored by git.

## **3. VS Code Integration**

The repository is pre-configured for a seamless experience in Visual Studio Code.

* **Python Interpreter:** The recommended Python interpreter is automatically set to use the ./.venv/bin/python executable.
* **Terminal Activation:** Opening a new terminal within VS Code will automatically activate the .venv virtual environment.
* **Testing & Formatting:** The editor is configured to use pytest for testing and ruff for formatting on save.

**IMPORTANT:** Always use the uv run prefix when executing commands manually in the terminal to ensure you are using the tools installed in the project's virtual environment.
