# Debugging Artifacts Organization Framework

## Overview
This framework provides standardized procedures for creating, storing, and organizing debugging artifacts, logs, and session documentation. The goal is to maintain consistency, traceability, and reusability across debugging sessions.

## Folder Structure Convention

### Base Organization
```
docs/ai_handbook/
├── 01_project_setup/           # Project setup and configuration docs
├── 02_requirements/            # Requirements analysis
├── 03_architecture/            # System architecture docs
├── 04_experiments/             # Experimental work and debugging
│   ├── YYYY-MM-DD_experiment_name/
│   │   ├── 01_initial_analysis.md
│   │   ├── 02_investigation.md
│   │   ├── 03_findings.md
│   │   ├── artifacts/           # Debugging artifacts
│   │   ├── logs/               # Rolling logs
│   │   └── scripts/            # Investigation scripts
│   └── YYYY-MM-DD_experiment_name/
└── 05_maintenance/             # Maintenance and operations
```

### Experiment-Specific Structure
```
YYYY-MM-DD_experiment_name/
├── 01_initial_analysis.md      # Problem statement and initial investigation
├── 02_investigation.md         # Detailed investigation steps
├── 03_findings.md              # Results and insights
├── 04_recommendations.md       # Actionable recommendations
├── artifacts/                  # Debugging artifacts (organized)
│   ├── cache_dumps/           # Cache state snapshots
│   ├── performance_profiles/  # Profiling data
│   ├── test_outputs/          # Test results and logs
│   └── config_snapshots/      # Configuration files used
├── logs/                      # Rolling logs (time-ordered)
│   ├── YYYY-MM-DD_HH-MM-SS_session.log
│   ├── YYYY-MM-DD_HH-MM-SS_performance.log
│   └── YYYY-MM-DD_HH-MM-SS_debug.log
├── scripts/                   # Reusable investigation scripts
│   ├── profile_cache_performance.py
│   ├── analyze_dataset_patterns.py
│   └── validate_cache_correctness.py
└── README.md                  # Session overview and navigation
```

## Naming Conventions

### Files
- **Date Prefix:** `YYYY-MM-DD_` for all session files
- **Sequential Numbering:** `01_`, `02_`, `03_` for ordered documents
- **Descriptive Names:** Clear, specific names without abbreviations
- **Extensions:** `.md` for docs, `.py` for scripts, `.log` for logs

**Examples:**
- `2025-10-08_initial_analysis.md`
- `2025-10-08_cache_performance_profile.py`
- `2025-10-08_14-30-00_debug.log`

### Folders
- **Date-Based:** `YYYY-MM-DD_experiment_name`
- **Descriptive:** Clear purpose indication
- **Hierarchical:** Logical grouping (artifacts/, logs/, scripts/)

## Artifact Creation Guidelines

### Cache Dumps
```bash
# Create timestamped cache dump
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
cp .cache/polygon_cache/polygon_cache.pkl \
   docs/ai_handbook/04_experiments/2025-10-08_performance_assessment/artifacts/cache_dumps/cache_dump_${TIMESTAMP}.pkl
```

### Performance Profiles
```python
# In profiling scripts
import cProfile
import pstats
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
profile_file = f"docs/ai_handbook/04_experiments/current_experiment/artifacts/performance_profiles/profile_{timestamp}.prof"

cProfile.run('main_function()', profile_file)

# Generate report
with open(f"artifacts/performance_profiles/profile_{timestamp}.txt", 'w') as f:
    stats = pstats.Stats(profile_file, stream=f)
    stats.sort_stats('cumulative').print_stats(20)
```

### Log Rolling
```bash
# Create rolling log function
create_rolling_log() {
    local log_type=$1
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    local log_dir="docs/ai_handbook/04_experiments/current_experiment/logs"
    local log_file="${log_dir}/${timestamp}_${log_type}.log"

    # Create log directory if needed
    mkdir -p "$log_dir"

    # Redirect output to rolling log
    exec > >(tee "$log_file") 2>&1
    echo "Rolling log started: $log_file"
}
```

### Configuration Snapshots
```bash
# Snapshot current configuration
TIMESTAMP=$(date +"%Y-%m-%D_%H-%M-%S")
CONFIG_DIR="docs/ai_handbook/04_experiments/current_experiment/artifacts/config_snapshots/${TIMESTAMP}"
mkdir -p "$CONFIG_DIR"

# Copy all active configs
cp configs/*.yaml "$CONFIG_DIR/"
cp configs/**/*.yaml "$CONFIG_DIR/" 2>/dev/null || true

# Create config manifest
ls -la "$CONFIG_DIR" > "$CONFIG_DIR/manifest.txt"
```

## Session Documentation Workflow

### Pre-Session Setup
1. Create dated experiment folder
2. Initialize basic structure (README.md, 01_initial_analysis.md)
3. Set up artifact directories
4. Create initial log file

### During Session
1. Log all commands and outputs to rolling logs
2. Save important artifacts immediately when created
3. Document findings incrementally
4. Update progress in README.md

### Post-Session
1. Run post-debugging session prompt (see below)
2. Organize loose files following naming conventions
3. Update project documentation if needed
4. Create session handover for continuation

## Quality Standards

### Completeness
- All debugging outputs saved as artifacts
- Commands logged with timestamps
- Configuration changes documented
- Test results preserved

### Traceability
- Clear links between artifacts and findings
- Session logs reference relevant artifacts
- Configuration changes tracked
- Performance metrics timestamped

### Reusability
- Scripts parameterized for different scenarios
- Documentation includes reproduction steps
- Artifacts include metadata (creation context)
- Naming conventions enable easy searching

## Automation Scripts

### Session Setup Script
```bash
#!/bin/bash
# setup_debugging_session.sh

EXPERIMENT_NAME=$1
DATE=$(date +"%Y-%m-%d")
SESSION_DIR="docs/ai_handbook/04_experiments/${DATE}_${EXPERIMENT_NAME}"

# Create directory structure
mkdir -p "$SESSION_DIR"/{artifacts/{cache_dumps,performance_profiles,test_outputs,config_snapshots},logs,scripts}

# Create initial files
cat > "$SESSION_DIR/README.md" << EOF
# $EXPERIMENT_NAME - Debugging Session

**Date:** $DATE
**Status:** IN PROGRESS

## Overview
[Brief description of the debugging session]

## Key Documents
- [01_initial_analysis.md](01_initial_analysis.md) - Problem statement
- [02_investigation.md](02_investigation.md) - Investigation steps
- [03_findings.md](03_findings.md) - Results and insights

## Artifacts
- [artifacts/](artifacts/) - Debugging artifacts
- [logs/](logs/) - Session logs
- [scripts/](scripts/) - Investigation scripts
EOF

echo "Debugging session setup complete: $SESSION_DIR"
```

### Artifact Organization Script
```bash
#!/bin/bash
# organize_debugging_artifacts.sh

SESSION_DIR=$1

# Find and organize loose files
find . -name "*debug*" -type f | while read file; do
    timestamp=$(date -r "$file" +"%Y-%m-%d_%H-%M-%S")
    extension="${file##*.}"
    base_name=$(basename "$file" ."$extension")

    # Create organized name
    organized_name="${timestamp}_${base_name}.${extension}"

    # Move to appropriate artifact directory
    if [[ $file == *"cache"* ]]; then
        mv "$file" "$SESSION_DIR/artifacts/cache_dumps/$organized_name"
    elif [[ $file == *"profile"* ]]; then
        mv "$file" "$SESSION_DIR/artifacts/performance_profiles/$organized_name"
    else
        mv "$file" "$SESSION_DIR/artifacts/test_outputs/$organized_name"
    fi
done

echo "Artifacts organized in $SESSION_DIR"
```

---

**Usage:** Run setup script at session start, organization script at session end. Follow naming conventions for all new files.
