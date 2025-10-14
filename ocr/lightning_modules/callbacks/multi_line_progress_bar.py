# Custom progress bar with line breaks for better readability
from lightning.pytorch.callbacks import RichProgressBar


class MultiLineRichProgressBar(RichProgressBar):
    """Custom RichProgressBar that displays progress information with better formatting."""

    def refresh(self):
        """Override refresh to ensure proper formatting."""
        # Call parent refresh without adding extra blank lines
        super().refresh()
