# Process Management Guide

## Overview

This guide covers the process management system implemented to prevent orphaned training processes and ensure clean resource management in the OCR training pipeline.

## Problem Statement

Training processes can leave behind orphaned worker processes when:
- Training is interrupted (Ctrl+C, UI crashes, system restarts)
- PyTorch DataLoader workers survive when the main process is killed
- Multiple training instances are started without proper cleanup

## Solution Architecture

### 1. Signal Handling in Training Script

The `runners/train.py` script now includes comprehensive signal handling:

```python
# Process group management
os.setpgrp()  # Create new process group

# Signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")

    # Cleanup logic for trainer and data modules
    # Terminate entire process group
    os.killpg(os.getpgrp(), signal.SIGTERM)
```

### 2. Safe DataLoader Configuration

Updated `configs/train.yaml` with safer DataLoader settings:

```yaml
dataloaders:
  train_dataloader:
    persistent_workers: false  # Prevents orphaned workers
    num_workers: 2             # Reduced for better control
  val_dataloader:
    persistent_workers: false
    num_workers: 1
  # ... similar for test and predict
```

### 3. UI Process Management

Enhanced `ui/utils/command_builder.py` with process group control:

```python
# Create process with new session/group
process = subprocess.Popen(
    command.split(),
    preexec_fn=os.setsid,  # New process group
    # ... other options
)

# Termination method for process groups
def terminate_process_group(self, process: subprocess.Popen):
    """Terminate entire process group safely."""
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
```

### 4. Process Monitor Utility

Comprehensive monitoring and cleanup tool at `scripts/process_monitor.py`.

## Usage Guide

### Before Starting Training

Always check for orphaned processes:

```bash
# List any existing training processes
python scripts/process_monitor.py --list

# Clean up if any exist
python scripts/process_monitor.py --cleanup
```

### During Training

Monitor processes if needed:

```bash
# Watch for orphaned processes
watch -n 30 "python scripts/process_monitor.py --list"
```

### After Training Issues

Clean up forcefully if needed:

```bash
# Graceful cleanup (recommended)
python scripts/process_monitor.py --cleanup

# Force cleanup if processes are stuck
python scripts/process_monitor.py --cleanup --force

# Preview what would be cleaned
python scripts/process_monitor.py --cleanup --dry-run
```

## Process Types Monitored

### Training Processes
- Main training script (`runners/train.py`)
- Identified by command line containing `runners/train.py`

### Worker Processes
- PyTorch DataLoader workers spawned by training
- Child processes of training processes
- Identified by parent process ID (PPID)

## Safety Features

### Graceful Shutdown
- SIGTERM sent first for clean shutdown
- 5-second timeout before force kill
- Process group termination ensures all children are killed

### Error Handling
- Continues operation even if individual process termination fails
- Logs failed terminations for manual intervention
- Safe fallback methods using different system tools

### Dry-run Mode
- Preview actions without executing them
- Verify correct process identification
- Safe testing of cleanup logic

## Troubleshooting

### Processes Still Running After Cleanup

If processes persist after cleanup:

1. Check process ownership:
   ```bash
   ps aux | grep train.py
   ```

2. Manual termination with specific PIDs:
   ```bash
   kill -TERM <PID>
   # Wait 5 seconds
   kill -KILL <PID>
   ```

3. Check for permission issues (processes owned by different users)

### False Positives in Detection

The monitor might detect unrelated processes. Check:
- Command line contains exact `runners/train.py` path
- Process is actually related to OCR training
- Use `--dry-run` to verify before cleanup

### Performance Impact

- Process monitoring has minimal performance impact
- Only runs when explicitly called
- Uses efficient system calls (`pgrep`, `ps`)

## Best Practices

### Development Workflow
1. Always run process monitor before starting training
2. Use UI for training (better process management)
3. Monitor long-running training sessions
4. Clean up after interrupted sessions

### Production Deployment
1. Integrate process monitoring into CI/CD pipelines
2. Set up automated cleanup scripts
3. Monitor system resources for orphaned processes
4. Implement health checks for training processes

### Resource Management
1. Limit concurrent training processes
2. Monitor GPU memory usage
3. Set appropriate DataLoader worker counts
4. Use process groups for better isolation

## Configuration Reference

### Signal Handling Configuration
- SIGINT: Keyboard interrupt (Ctrl+C)
- SIGTERM: Termination signal
- Process groups: Automatic creation and management

### DataLoader Configuration
- `persistent_workers: false` - Prevents orphaned workers
- `num_workers: 1-4` - Balance performance vs. control
- `pin_memory: true` - GPU transfer optimization

### Process Monitor Options
- `--list`: Show current processes
- `--cleanup`: Terminate processes
- `--force`: Use SIGKILL instead of SIGTERM
- `--dry-run`: Preview actions

## Files Modified

### Core Implementation
- `runners/train.py` - Signal handling and process groups
- `configs/train.yaml` - Safe DataLoader configuration
- `ui/utils/command_builder.py` - Process group management

### Utilities
- `scripts/process_monitor.py` - Monitoring and cleanup tool

### Documentation
- `docs/maintenance/project-state.md` - Process management status
- `README.md` - Usage instructions and directory structure

## Future Improvements

### Planned Enhancements
- Automatic process monitoring during training
- Integration with training UI for real-time monitoring
- Resource usage tracking and limits
- Automated cleanup on system startup

### Monitoring Extensions
- GPU memory monitoring per process
- CPU usage tracking
- Network I/O monitoring for distributed training
- Log aggregation for process lifecycle events
