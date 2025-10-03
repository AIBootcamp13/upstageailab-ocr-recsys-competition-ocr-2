# OCR Receipt Text Detection - Docker Development Setup

This guide covers everything you need to set up and run the OCR Receipt Text Detection project using Docker for development.

## 🚀 Quick Start (Recommended)

### Option 1: Docker Development (GPU)
```bash
# Start GPU development environment
docker-compose --profile dev up -d

# Or for CPU-only development
docker-compose --profile cpu up -d

# Access via SSH (recommended for VS Code remote development)
ssh vscode@localhost -p 2222

# Or access directly
docker-compose exec dev bash
```

### Option 2: Local Development
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Setup environment
./setup.sh

# Run training
python runners/train.py

# Run prediction
python runners/predict.py
```

## 📋 Prerequisites

### Required
- **Python 3.9+** (managed by uv)
- **Git**
- **curl** (for uv installation)

### Optional (but recommended)
- **Docker** (for containerized development)
- **NVIDIA GPU** (for accelerated ML training)

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential git curl

# macOS
brew install git curl

# Windows (WSL recommended)
# Use Ubuntu WSL and follow Ubuntu instructions
```

## 🛠️ Manual Setup Steps

### 1. Install uv Package Manager
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
uv --version
```

### 2. Clone and Setup Repository
```bash
# Clone repository
git clone <repository-url>
cd upstage-ocr-receipt-text-detection

# Install dependencies
./setup.sh
```

### 3. Configure VS Code (Recommended)
```json
// .vscode/settings.json (already configured)
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": false,
  "terminal.integrated.shell.linux": "/usr/bin/bash",
  "python.analysis.extraPaths": [".", "ocr", "tests"]
}
```

## 🐳 Docker Setup

### Development Environments

#### GPU Development (Recommended)
```bash
docker-compose --profile dev up -d
docker-compose exec dev bash
```

#### CPU Development
```bash
docker-compose --profile cpu up -d
docker-compose exec dev-cpu bash
```

### Docker Helper Script
```bash
# Check status of all services
./docker-helper.sh status

# Start development environment
./docker-helper.sh start

# Stop all services
./docker-helper.sh stop

# Clean up Docker resources
./docker-helper.sh clean
```

### SSH Access
The development containers expose SSH on port 2222 for VS Code remote development:

```bash
# Setup SSH keys for container access
./docker/setup-ssh-container.sh

# Or use SSH volume persistence
./docker/setup-ssh-volume.sh
```

### Enhanced Shell Environment
The containers include a comprehensive `.bashrc` with:
- **Color prompt** with git branch display
- **Korean locale support** (ko_KR.UTF-8)
- **Bash completions** for better tab completion
- **Development aliases** (train, predict, test, etc.)
- **Utility functions** (extract, mkcd, sysinfo, etc.)
- **Disabled history expansion** (! commands)
- **PyEnv integration** (if available)

## 🔧 Development Workflow

### Daily Development
```bash
# Activate environment (automatic with uv)
cd /path/to/project

# Install new dependencies
uv add package-name

# Run training
python runners/train.py

# Run prediction
python runners/predict.py

# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix
```

### Code Quality Tools
```bash
# Install development dependencies
uv sync --group dev

# Run all quality checks
uv run pre-commit run --all-files

# Format and lint
uv run ruff format .
uv run ruff check . --fix
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "uv command not found"
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### 2. GPU not available in Docker
```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# For Docker Compose
docker-compose --profile dev up
```

#### 3. Permission issues with Docker
```bash
# Fix file permissions
./docker-helper.sh fix-permissions
```

#### 4. Virtual environment conflicts
```bash
# Deactivate any manual activation
deactivate

# Use uv exclusively
uv run python script.py
```

### Environment Variables
```bash
# Disable dynamic Python path (if causing issues)
export SKIP_DYNAMIC_PYTHONPATH=1

# Disable venv prompt
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Set log level
export LOG_LEVEL=DEBUG
```

## 📊 Assessment: Development Options

### ✅ Highly Recommended
- **Docker GPU development** (if you have NVIDIA GPU)
- **uv package management** (fast, reliable dependency management)
- **Docker helper scripts** (simplify container management)

### ⚠️ Alternative Options
- **Docker CPU development** (for systems without GPU)
- **Local development** (if Docker is not preferred)

### 🎯 Recommended Improvements

#### High Priority
1. **Automated testing** in Docker environment
2. **Pre-commit hooks** for code quality
3. **CI/CD pipeline** for automated testing

#### Medium Priority
1. **VS Code dev container** configuration
2. **Environment-specific configs** (dev vs prod settings)

---

**Need help?** Check the troubleshooting section or open an issue with your setup details.</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/SETUP.md
