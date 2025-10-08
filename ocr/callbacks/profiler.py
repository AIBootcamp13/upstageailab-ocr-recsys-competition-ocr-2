import json
import logging
import os
from typing import Any

import torch
from lightning.pytorch import Callback
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

logger = logging.getLogger(__name__)


class ProfilerCallback(Callback):
    """
    PyTorch Profiler integration for training pipeline analysis.

    Features:
    - CPU/GPU/Memory profiling
    - Chrome trace export
    - Automated bottleneck detection
    - Configurable profiling windows
    """

    def __init__(
        self,
        enabled: bool = True,
        profile_epochs: list[int] | None = None,  # Which epochs to profile
        profile_steps: int = 100,  # Steps to profile per epoch
        warmup_steps: int = 5,
        activities: list[str] | None = None,  # ["cpu", "cuda"]
        record_shapes: bool = True,
        with_stack: bool = False,  # Stack traces (slow but detailed)
        output_dir: str = "profiler_traces",
        export_chrome_trace: bool = True,
        log_top_k_ops: int = 10,  # Top-k slowest operations
    ):
        super().__init__()
        self.enabled = enabled
        self.profile_epochs = profile_epochs or [1]  # Profile first epoch by default
        self.profile_steps = profile_steps
        self.warmup_steps = warmup_steps
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.output_dir = output_dir
        self.export_chrome_trace = export_chrome_trace
        self.log_top_k_ops = log_top_k_ops

        # Convert activity strings to ProfilerActivity enums
        self.activities = []
        if activities is None:
            # Default to both CPU and CUDA if available
            self.activities.append(ProfilerActivity.CPU)
            if torch.cuda.is_available():
                self.activities.append(ProfilerActivity.CUDA)
        else:
            for activity in activities:
                if activity.lower() == "cpu":
                    self.activities.append(ProfilerActivity.CPU)
                elif activity.lower() == "cuda" and torch.cuda.is_available():
                    self.activities.append(ProfilerActivity.CUDA)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize profiler as None initially
        self.profiler = None
        self.step_counter = 0

    def _should_profile_epoch(self, current_epoch: int) -> bool:
        """Check if the current epoch should be profiled."""
        return self.enabled and current_epoch in self.profile_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        """Initialize profiler at the start of the epoch if needed."""
        current_epoch = trainer.current_epoch
        if self._should_profile_epoch(current_epoch):
            logger.info(f"[Profiler] Epoch {current_epoch}: Profiling steps 0-{min(self.profile_steps, trainer.num_training_batches)}...")

            # Create a schedule for the profiler
            # Ensure we have at least 1 warmup step to avoid profiler warnings
            actual_warmup = max(1, self.warmup_steps)
            profiler_schedule = schedule(
                skip_first=0,  # Don't skip first steps
                wait=0,  # No waiting between profiling
                warmup=actual_warmup,
                active=self.profile_steps,
                repeat=1,
            )

            # Initialize the profiler
            self.profiler = profile(
                activities=self.activities,
                schedule=profiler_schedule,
                on_trace_ready=tensorboard_trace_handler(self.output_dir, worker_name=f"epoch_{current_epoch}")
                if self.export_chrome_trace
                else None,
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                profile_memory=True,  # Enable memory profiling
            )
            self.profiler.__enter__()
        else:
            self.profiler = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Step the profiler after each training batch."""
        if self.profiler is not None:
            # Check if we're still within the profiling window
            if self.step_counter < self.profile_steps:
                self.profiler.step()
                self.step_counter += 1
            else:
                # If we've exceeded the profile steps, disable profiler
                if not self.profiler.profiler_ready():
                    self.profiler.__exit__(None, None, None)
                    self.profiler = None

    def on_train_epoch_end(self, trainer, pl_module):
        """Finalize profiling at the end of the epoch."""
        if self.profiler is not None:
            # Make sure all steps are processed
            while self.step_counter < self.profile_steps:
                self.profiler.step()
                self.step_counter += 1

            # Exit and export
            self.profiler.__exit__(None, None, None)

            # Reset step counter for next epoch
            self.step_counter = 0

            # Get current epoch for naming
            current_epoch = trainer.current_epoch

            # Export Chrome trace if required
            if self.export_chrome_trace:
                # The tensorboard_trace_handler creates files with a timestamp
                # Format: worker_name.trace.json (e.g., epoch_1.trace.json)
                trace_file = os.path.join(self.output_dir, f"epoch_{current_epoch}.trace.json")

                # If the expected file doesn't exist, try the alternative format that PyTorch profiler uses
                if not os.path.exists(trace_file):
                    import glob

                    pattern = os.path.join(self.output_dir, f"epoch_{current_epoch}*.json")
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        trace_file = matching_files[0]  # Use the first matching file
                    else:
                        # If no files found with the epoch pattern, look for any JSON file in output_dir
                        all_json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
                        if all_json_files:
                            # Get the most recently modified file
                            trace_file = max(all_json_files, key=os.path.getmtime)

                if os.path.exists(trace_file):
                    logger.info(f"[Profiler] Trace exported: {trace_file}")

                    # Analyze and log bottlenecks
                    self._analyze_and_log_bottlenecks(trace_file)
                else:
                    logger.warning(f"[Profiler] Expected trace file not found: {trace_file}")
                    # Try to list all JSON files in the output directory for debugging
                    import glob

                    all_json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
                    if all_json_files:
                        logger.warning(f"[Profiler] Available JSON files: {all_json_files}")

            # Reset profiler
            self.profiler = None

    def _analyze_and_log_bottlenecks(self, trace_path: str):
        """Analyze Chrome trace file to identify bottlenecks."""
        try:
            trace_data = self._load_trace(trace_path)
            bottlenecks = self.analyze_trace(trace_data)

            # Log top operations
            self._log_top_operations(bottlenecks)

        except Exception as e:
            logger.error(f"[Profiler] Error analyzing trace {trace_path}: {e}")

    def _load_trace(self, trace_path: str) -> dict[str, Any]:
        """Load trace data from JSON file."""
        with open(trace_path) as f:
            return json.load(f)

    def analyze_trace(self, trace_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze Chrome trace data to identify bottlenecks.

        Returns:
            {
                "top_cpu_ops": [(op_name, time_us, percent), ...],
                "top_cuda_ops": [(op_name, time_us, percent), ...],
                "memory_peaks": [(time, bytes), ...],
                "dataloader_time_pct": float,
            }
        """
        events = trace_data.get("traceEvents", [])

        # Separate events by process and category
        cpu_ops: dict[str, int] = {}  # {op_name: total_duration}
        cuda_ops: dict[str, int] = {}  # {op_name: total_duration}
        memory_events = []
        dataloader_events = []

        # Process events to categorize and aggregate times
        for event in events:
            if "dur" in event and "name" in event:  # Only timed events
                duration_us = event["dur"]  # Duration in microseconds
                name = event.get("name", "unknown")

                # Categorize operations based on their names and categories
                if "cat" in event:
                    if any(cuda_indicator in event["cat"].lower() for cuda_indicator in ["cuda", "gpu", "kernel"]):
                        if name in cuda_ops:
                            cuda_ops[name] += duration_us
                        else:
                            cuda_ops[name] = duration_us
                    elif any(cpu_indicator in event["cat"].lower() for cpu_indicator in ["cpu", "op", "compute"]):
                        if name in cpu_ops:
                            cpu_ops[name] += duration_us
                        else:
                            cpu_ops[name] = duration_us
                    elif "memory" in event["cat"].lower() or "mem" in event["cat"].lower():
                        memory_events.append({"ts": event.get("ts", 0), "dur": duration_us})
                else:
                    # Heuristic-based categorization if no category is provided
                    if any(cuda_indicator in name.lower() for cuda_indicator in ["cuda", "kernel", "volta", "gemm", "cudnn"]):
                        if name in cuda_ops:
                            cuda_ops[name] += duration_us
                        else:
                            cuda_ops[name] = duration_us
                    elif any(loader_indicator in name.lower() for loader_indicator in ["loader", "data", "dataloader"]):
                        dataloader_events.append({"name": name, "dur": duration_us})
                    else:
                        if name in cpu_ops:
                            cpu_ops[name] += duration_us
                        else:
                            cpu_ops[name] = duration_us

        # Convert to sorted lists
        cpu_ops_list = [(name, duration, 0.0) for name, duration in cpu_ops.items()]
        cuda_ops_list = [(name, duration, 0.0) for name, duration in cuda_ops.items()]

        # Calculate percentages and sort by duration (descending)
        total_cpu_time = sum(duration for _, duration, _ in cpu_ops_list) if cpu_ops_list else 1
        total_cuda_time = sum(duration for _, duration, _ in cuda_ops_list) if cuda_ops_list else 1

        # Add percentages and sort
        cpu_ops_with_pct = [(name, duration, (duration / total_cpu_time) * 100) for name, duration, _ in cpu_ops_list]
        cuda_ops_with_pct = [(name, duration, (duration / total_cuda_time) * 100) for name, duration, _ in cuda_ops_list]

        cpu_ops_with_pct.sort(key=lambda x: x[1], reverse=True)
        cuda_ops_with_pct.sort(key=lambda x: x[1], reverse=True)

        # Get top-k operations
        top_cpu_ops = cpu_ops_with_pct[: self.log_top_k_ops]
        top_cuda_ops = cuda_ops_with_pct[: self.log_top_k_ops]

        # Find memory peaks (simplified - look for memory allocation events)
        memory_peaks = []
        if memory_events:
            # For simplicity, we'll just return the total number of memory events
            # In a full implementation, we'd parse memory snapshots to find actual peaks
            memory_peaks = [(event["ts"], event["dur"]) for event in memory_events[: self.log_top_k_ops]]

        # Calculate dataloader time percentage
        total_dataloader_time = sum(e["dur"] for e in dataloader_events)
        dataloader_time_pct = (total_dataloader_time / total_cpu_time) * 100 if total_cpu_time > 0 else 0

        return {
            "top_cpu_ops": top_cpu_ops,
            "top_cuda_ops": top_cuda_ops,
            "memory_peaks": memory_peaks,
            "dataloader_time_pct": dataloader_time_pct,
        }

    def _log_top_operations(self, bottlenecks: dict[str, Any]):
        """Log the top operations to console."""
        # Log top CPU operations
        if bottlenecks["top_cpu_ops"]:
            logger.info(f"[Profiler] Top {self.log_top_k_ops} CPU operations:")
            for i, (op_name, time_us, pct) in enumerate(bottlenecks["top_cpu_ops"], 1):
                time_ms = time_us / 1000  # Convert to milliseconds
                logger.info(f"  {i}. {op_name} - {time_ms:.1f}ms ({pct:.1f}%)")

        # Log top CUDA operations
        if bottlenecks["top_cuda_ops"]:
            logger.info(f"[Profiler] Top {self.log_top_k_ops} CUDA operations:")
            for i, (op_name, time_us, pct) in enumerate(bottlenecks["top_cuda_ops"], 1):
                time_ms = time_us / 1000  # Convert to milliseconds
                logger.info(f"  {i}. {op_name} - {time_ms:.1f}ms ({pct:.1f}%)")

        # Log summary
        logger.info("[Profiler] Summary:")
        logger.info(f"  Dataloader time: {bottlenecks['dataloader_time_pct']:.1f}%")

        # Log recommendations if applicable
        cpu_ops = bottlenecks["top_cpu_ops"]
        if cpu_ops and "conv2d" in cpu_ops[0][0].lower():
            logger.info("[Profiler] ⚠️ Bottleneck detected:")
            logger.info("  - High conv2d CPU time (consider torch.compile)")

        if bottlenecks["dataloader_time_pct"] < 10:
            logger.info("  - Dataloader time <10% (good efficiency)")
        else:
            logger.info(f"  - Dataloader time {bottlenecks['dataloader_time_pct']:.1f}% (could be optimized)")
