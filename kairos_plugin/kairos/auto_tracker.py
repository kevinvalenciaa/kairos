"""
Automatic cell tracking for Jupyter notebooks.

This module provides IPython hooks that automatically track every cell execution
without requiring users to wrap their code in context managers.

Usage:
    # In a Jupyter notebook - just one line to start:
    import kairos
    kairos.auto_track()

    # Or with configuration:
    kairos.auto_track(hourly_rate=32.77, provider="aws", instance="p4d.24xlarge")

    # All subsequent cells are automatically tracked!
    # No code changes needed.

    # View status anytime:
    kairos.status()

    # View summary:
    kairos.summary()
"""

from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
import threading
import logging
import weakref
import atexit

from kairos.tracker import KairosTracker
from kairos.config import KairosConfig, CloudProvider
from kairos.cost_calculator import CellExecution

logger = logging.getLogger(__name__)


class AutoTracker:
    """
    Automatic cell tracking using IPython hooks.

    Registers callbacks with IPython to automatically track every cell
    execution without user intervention.
    """

    _instance: Optional["AutoTracker"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one auto tracker per session."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        tracker: Optional[KairosTracker] = None,
        config: Optional[KairosConfig] = None,
        show_cell_output: bool = True,
        show_status_on_start: bool = True,
        auto_save: bool = True,
        save_path: Optional[str] = None,
        quiet: bool = False,
    ):
        """
        Initialize auto tracker.

        Args:
            tracker: Optional existing tracker to use
            config: Optional configuration (ignored if tracker provided)
            show_cell_output: Show cost after each cell
            show_status_on_start: Show status when tracking starts
            auto_save: Automatically save session data
            save_path: Path for session data (default: .kairos/)
            quiet: Suppress all output except errors
        """
        # Avoid re-initialization if already set up
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._tracker = tracker or KairosTracker(config=config, auto_sample=True)
        self._show_cell_output = show_cell_output
        self._show_status_on_start = show_status_on_start
        self._auto_save = auto_save
        self._save_path = save_path
        self._quiet = quiet

        # IPython state
        self._ipython = None
        self._registered = False
        self._current_cell_info: Optional[Dict[str, Any]] = None
        self._execution_count = 0

        # Cell tracking
        self._cell_start_time: Optional[datetime] = None
        self._cell_code: Optional[str] = None

        # Callbacks
        self._on_cell_complete: List[Callable[[CellExecution], None]] = []

        self._initialized = True

        # Register cleanup
        atexit.register(self._cleanup)

    @property
    def tracker(self) -> KairosTracker:
        """Get the underlying tracker."""
        return self._tracker

    @property
    def is_registered(self) -> bool:
        """Check if hooks are registered with IPython."""
        return self._registered

    def start(self) -> "AutoTracker":
        """
        Start automatic tracking.

        Registers hooks with IPython to track all cell executions.

        Returns:
            Self for chaining
        """
        if self._registered:
            if not self._quiet:
                logger.info("Auto tracking already active")
            return self

        # Get IPython instance
        try:
            from IPython import get_ipython
            self._ipython = get_ipython()

            if self._ipython is None:
                raise RuntimeError(
                    "No IPython instance found. "
                    "Auto tracking only works in Jupyter/IPython environments."
                )
        except ImportError:
            raise RuntimeError(
                "IPython not installed. "
                "Install with: pip install ipython"
            )

        # Register event callbacks
        self._ipython.events.register("pre_run_cell", self._on_pre_run_cell)
        self._ipython.events.register("post_run_cell", self._on_post_run_cell)

        self._registered = True

        if not self._quiet:
            logger.info(f"Kairos auto tracking started (session: {self._tracker.session_id})")

            if self._show_status_on_start:
                self._tracker.status()

        return self

    def stop(self) -> "AutoTracker":
        """
        Stop automatic tracking.

        Unregisters hooks from IPython.

        Returns:
            Self for chaining
        """
        if not self._registered:
            return self

        if self._ipython:
            try:
                self._ipython.events.unregister("pre_run_cell", self._on_pre_run_cell)
                self._ipython.events.unregister("post_run_cell", self._on_post_run_cell)
            except ValueError:
                # Already unregistered
                pass

        self._registered = False

        if not self._quiet:
            logger.info("Kairos auto tracking stopped")

        return self

    def _on_pre_run_cell(self, info) -> None:
        """
        Callback before cell execution.

        Args:
            info: IPython cell info object
        """
        self._cell_start_time = datetime.now()
        self._execution_count += 1

        # Store cell info
        self._current_cell_info = {
            "execution_count": self._execution_count,
            "raw_cell": getattr(info, "raw_cell", ""),
            "store_history": getattr(info, "store_history", True),
        }

        # Extract cell code for tagging
        raw_cell = self._current_cell_info.get("raw_cell", "")
        self._cell_code = raw_cell[:200] if raw_cell else None

        # Start tracking in calculator
        tags = self._extract_tags(raw_cell)
        self._tracker._calculator.start_cell(
            cell_number=self._execution_count,
            tags=tags,
        )

    def _on_post_run_cell(self, result) -> None:
        """
        Callback after cell execution.

        Args:
            result: IPython execution result
        """
        if self._cell_start_time is None:
            return

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - self._cell_start_time).total_seconds()

        # Check for errors
        error = None
        if result.error_in_exec is not None:
            error = str(result.error_in_exec)
        elif result.error_before_exec is not None:
            error = str(result.error_before_exec)

        # End cell tracking
        cell = self._tracker._calculator.end_cell(error=error)

        # Show output
        if cell and self._show_cell_output and not self._quiet:
            self._tracker._display.render_cell_log(cell)

        # Auto-save
        if self._auto_save and cell:
            self._save_cell(cell)

        # Fire callbacks
        if cell:
            for callback in self._on_cell_complete:
                try:
                    callback(cell)
                except Exception as e:
                    logger.warning(f"Cell callback error: {e}")

        # Reset state
        self._cell_start_time = None
        self._current_cell_info = None
        self._cell_code = None

    def _extract_tags(self, code: str) -> List[str]:
        """
        Extract tags from cell code.

        Looks for:
        - # @kairos: tag1, tag2
        - # kairos-tags: tag1, tag2
        - Common ML keywords in the code

        Args:
            code: Cell source code

        Returns:
            List of tags
        """
        tags = []

        if not code:
            return tags

        # Look for explicit kairos tags
        import re

        # Pattern: # @kairos: tag1, tag2 or # kairos-tags: tag1, tag2
        tag_patterns = [
            r"#\s*@kairos:\s*(.+?)(?:\n|$)",
            r"#\s*kairos-tags:\s*(.+?)(?:\n|$)",
            r"#\s*tags:\s*(.+?)(?:\n|$)",
        ]

        for pattern in tag_patterns:
            match = re.search(pattern, code, re.IGNORECASE)
            if match:
                tag_str = match.group(1)
                tags.extend([t.strip() for t in tag_str.split(",") if t.strip()])

        # Auto-detect common ML operations
        code_lower = code.lower()

        ml_keywords = {
            "training": [".fit(", ".train(", "trainer.train", "model.train()"],
            "inference": [".predict(", ".inference(", "model.eval()"],
            "data-loading": ["dataloader", "dataset", "load_data", "read_csv", "read_parquet"],
            "preprocessing": ["preprocess", "transform", "tokenize", "normalize"],
            "evaluation": [".eval(", "evaluate", "accuracy", "f1_score", "precision"],
            "visualization": ["plt.", "plot(", "figure(", "seaborn", "sns."],
            "model-init": ["nn.module", "nn.linear", "model = ", "from transformers"],
        }

        for tag, keywords in ml_keywords.items():
            if any(kw in code_lower for kw in keywords):
                if tag not in tags:
                    tags.append(tag)

        return tags[:5]  # Limit to 5 tags

    def _save_cell(self, cell: CellExecution) -> None:
        """Save cell data to persistent storage."""
        # Import here to avoid circular dependency
        try:
            from kairos.storage import SessionStorage
            storage = SessionStorage(base_path=self._save_path)
            storage.save_cell(self._tracker.session_id, cell)
        except Exception as e:
            logger.debug(f"Failed to save cell: {e}")

    def on_cell_complete(self, callback: Callable[[CellExecution], None]) -> None:
        """
        Register a callback for cell completion.

        Args:
            callback: Function to call with CellExecution after each cell
        """
        self._on_cell_complete.append(callback)

    def status(self) -> None:
        """Display current tracking status."""
        self._tracker.status()

    def summary(self) -> None:
        """Display session summary."""
        self._tracker.summary()

    def pause(self) -> None:
        """Temporarily pause tracking (cells won't be tracked)."""
        self.stop()
        if not self._quiet:
            self._tracker.alert("Tracking paused. Use kairos.resume() to continue.", "info")

    def resume(self) -> None:
        """Resume tracking after pause."""
        self.start()
        if not self._quiet:
            self._tracker.alert("Tracking resumed.", "info")

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        self.stop()
        if self._auto_save:
            try:
                from kairos.storage import SessionStorage
                storage = SessionStorage(base_path=self._save_path)
                storage.save_session(self._tracker)
            except Exception as e:
                logger.debug(f"Failed to save session on exit: {e}")


# Global auto tracker instance
_auto_tracker: Optional[AutoTracker] = None


def auto_track(
    hourly_rate: Optional[float] = None,
    provider: Optional[str] = None,
    instance: Optional[str] = None,
    show_cell_output: bool = True,
    show_status_on_start: bool = True,
    auto_save: bool = True,
    quiet: bool = False,
) -> AutoTracker:
    """
    Start automatic cell tracking.

    Call this once at the start of your notebook to automatically
    track all subsequent cell executions.

    Args:
        hourly_rate: Custom hourly rate in USD (overrides provider/instance)
        provider: Cloud provider ("aws", "gcp", "azure", "local")
        instance: Instance type (e.g., "p4d.24xlarge")
        show_cell_output: Show cost after each cell execution
        show_status_on_start: Display status when tracking starts
        auto_save: Automatically save session data
        quiet: Suppress all output except errors

    Returns:
        AutoTracker instance

    Example:
        >>> import kairos
        >>> kairos.auto_track()  # Uses auto-detection

        >>> # Or with configuration:
        >>> kairos.auto_track(provider="aws", instance="p4d.24xlarge")

        >>> # All cells are now automatically tracked!
    """
    global _auto_tracker

    # Build configuration
    config = KairosConfig()

    if hourly_rate is not None:
        config.custom_hourly_rate = hourly_rate

    if provider:
        provider_map = {
            "aws": CloudProvider.AWS,
            "gcp": CloudProvider.GCP,
            "azure": CloudProvider.AZURE,
            "local": CloudProvider.LOCAL,
        }
        config.cloud_provider = provider_map.get(provider.lower(), CloudProvider.LOCAL)

    if instance:
        config.instance_type = instance

    # Create tracker
    tracker = KairosTracker(config=config, auto_sample=True)

    # Create and start auto tracker
    _auto_tracker = AutoTracker(
        tracker=tracker,
        show_cell_output=show_cell_output,
        show_status_on_start=show_status_on_start,
        auto_save=auto_save,
        quiet=quiet,
    )

    return _auto_tracker.start()


def stop_tracking() -> None:
    """Stop automatic cell tracking."""
    global _auto_tracker
    if _auto_tracker:
        _auto_tracker.stop()


def get_tracker() -> Optional[KairosTracker]:
    """Get the current tracker instance."""
    global _auto_tracker
    if _auto_tracker:
        return _auto_tracker.tracker
    return None


def status() -> None:
    """Display current tracking status."""
    global _auto_tracker
    if _auto_tracker:
        _auto_tracker.status()
    else:
        print("No active tracker. Start with: kairos.auto_track()")


def summary() -> None:
    """Display session summary."""
    global _auto_tracker
    if _auto_tracker:
        _auto_tracker.summary()
    else:
        print("No active tracker. Start with: kairos.auto_track()")
