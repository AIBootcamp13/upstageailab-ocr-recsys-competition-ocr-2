# Enhanced logging utilities with Rich and IceCream integration
"""
Logging utilities for OCR framework with rich console output and debugging support.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    RichHandler = None  # type: ignore

try:
    from icecream import ic, install  # type: ignore

    ICECREAM_AVAILABLE = True
except ImportError:
    ICECREAM_AVAILABLE = False

    def ic(*args):  # type: ignore
        print(*args)

    def install():
        pass


class OCRLogger:
    """Enhanced logger with Rich console output and structured logging."""

    def __init__(
        self,
        name: str = "ocr",
        level: str = "INFO",
        log_file: str | Path | None = None,
        rich_console: bool = True,
    ):
        """Initialize OCR logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            rich_console: Whether to use Rich console output
        """
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_file = Path(log_file) if log_file else None
        self.rich_console = rich_console and RICH_AVAILABLE

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add Rich handler for console output
        if self.rich_console:
            console = Console(color_system="256", force_terminal=True)
            rich_handler = RichHandler(
                console=console,
                show_time=False,  # We'll use custom format instead
                show_level=True,  # Enable level display for coloring
                show_path=False,
                enable_link_path=False,
            )
            rich_format = "%(name)s - %(message)s"
            rich_handler.setFormatter(logging.Formatter(rich_format))
            rich_handler.setLevel(self.level)
            self.logger.addHandler(rich_handler)
        else:
            # Fallback to basic console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Configure IceCream if available
        if ICECREAM_AVAILABLE:
            install()
            ic.configureOutput(includeContext=True)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log training/validation metrics with rich formatting."""
        if self.rich_console:
            table = Table(title=f"Metrics {'(Step ' + str(step) + ')' if step else ''}")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")

            for key, value in metrics.items():
                if isinstance(value, float):
                    table.add_row(key, ".4f")
                else:
                    table.add_row(key, str(value))

            console = Console()
            console.print(table)
        else:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            self.logger.info(f"Metrics: {metrics_str}")

    def log_config(self, config: dict[str, Any]):
        """Log configuration with rich formatting."""
        if self.rich_console:
            config_text = self._format_config(config)
            panel = Panel(config_text, title="Configuration", expand=False)
            console = Console()
            console.print(panel)
        else:
            self.logger.info(f"Configuration: {config}")

    def _format_config(self, config: dict[str, Any], indent: int = 0) -> str:
        """Format configuration dictionary for display."""
        lines = []
        for key, value in config.items():
            if isinstance(value, dict):
                lines.append("  " * indent + f"{key}:")
                lines.append(self._format_config(value, indent + 1))
            else:
                lines.append("  " * indent + f"{key}: {value}")
        return "\n".join(lines)

    def progress_bar(self, iterable=None, description="Processing", total=None):
        """Create a progress bar for iterations."""
        if self.rich_console and iterable is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=Console(),
            ) as progress:
                task = progress.add_task(description, total=total or len(iterable))
                for item in iterable:
                    yield item
                    progress.update(task, advance=1)
        else:
            # Fallback to tqdm if available
            try:
                from tqdm import tqdm

                yield from tqdm(iterable, desc=description, total=total)
            except ImportError:
                yield from iterable


class DebugTools:
    """Debugging utilities with IceCream integration."""

    def __init__(self):
        """Initialize debug tools."""
        if ICECREAM_AVAILABLE:
            # Configure IceCream for better debugging
            ic.configureOutput(includeContext=True, contextAbsPath=True, prefix="🐛 ")

    @staticmethod
    def debug_tensor(tensor, name: str = "tensor"):
        """Debug tensor with shape and statistics."""
        if not TORCH_AVAILABLE or not torch.is_tensor(tensor):
            ic(f"{name}: {tensor}")
            return

        info = {
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "requires_grad": tensor.requires_grad,
        }

        if ICECREAM_AVAILABLE:
            ic(f"{name}_info", info)
            ic(
                f"{name}_sample",
                tensor.flatten()[:10] if tensor.numel() > 10 else tensor,
            )
        else:
            print(f"{name}_info: {info}")
            print(f"{name}_sample: {tensor.flatten()[:10] if tensor.numel() > 10 else tensor}")

    @staticmethod
    def debug_model(model, input_shape: tuple | None = None):
        """Debug model architecture and parameters."""
        if not TORCH_AVAILABLE:
            ic(f"Model debug not available - torch not installed: {model}")
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_class": model.__class__.__name__,
            "modules": len(list(model.modules())),
        }

        if input_shape and ICECREAM_AVAILABLE:
            try:
                dummy_input = torch.randn(1, *input_shape)
                with torch.no_grad():
                    output = model(dummy_input)
                info["input_shape"] = input_shape
                info["output_shape"] = output.shape
            except Exception as e:
                info["forward_error"] = str(e)

        if ICECREAM_AVAILABLE:
            ic(f"model_info: {info}")
        else:
            print(f"Model info: {info}")

    @staticmethod
    def time_function(func):
        """Decorator to time function execution."""

        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time
            if ICECREAM_AVAILABLE:
                ic(f"{func.__name__} execution time: {duration:.4f}")
            else:
                print(f"{func.__name__} execution time: {duration:.4f}")

            return result

        return wrapper


# Global logger instance
logger = OCRLogger()

# Global debug tools instance
debug = DebugTools()


# Convenience functions
def log_experiment_start(experiment_name: str, config: dict[str, Any]):
    """Log experiment start with configuration."""
    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.log_config(config)


def log_experiment_end(experiment_name: str, metrics: dict[str, Any]):
    """Log experiment completion with final metrics."""
    logger.info(f"✅ Completed experiment: {experiment_name}")
    logger.log_metrics(metrics)


def create_experiment_logger(experiment_name: str, output_dir: str | Path) -> OCRLogger:
    """Create a logger for a specific experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / "logs" / f"{experiment_name}_{timestamp}.log"

    return OCRLogger(
        name=f"ocr.{experiment_name}",
        level="DEBUG",
        log_file=log_file,
        rich_console=True,
    )


# Export convenience functions
__all__ = [
    "OCRLogger",
    "DebugTools",
    "logger",
    "debug",
    "log_experiment_start",
    "log_experiment_end",
    "create_experiment_logger",
    "get_rich_console",
]


def get_rich_console(**console_kwargs: Any):
    """Return a Rich Console instance when the dependency is available.

    Falls back to ``None`` so callers can gracefully degrade to tqdm.
    """
    if not RICH_AVAILABLE or Console is None:
        return None
    try:
        return Console(**console_kwargs)
    except Exception:  # pragma: no cover - console construction errors should not crash pipeline
        return None
