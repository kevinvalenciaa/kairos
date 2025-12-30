"""
KairosTracker - Main interface for ML cost tracking in Jupyter notebooks.

This is the primary class that users interact with. It provides a simple API
for tracking GPU costs, logging cell executions, and viewing summaries.

Example:
    >>> from kairos import KairosTracker
    >>> tracker = KairosTracker()
    >>> tracker.status()  # View current costs

    >>> # Option 1: Manual cell logging
    >>> tracker.log_cell(duration=120, tags=["training"])

    >>> # Option 2: Context manager for automatic timing
    >>> with tracker.track_cell(tags=["inference"]):
    ...     model.predict(data)

    >>> tracker.summary()  # View session summary
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime
import time
import threading
import logging
import atexit

from kairos.config import KairosConfig, CloudProvider
from kairos.gpu_monitor import GPUMonitor, GPUSnapshot
from kairos.cost_calculator import CostCalculator, CellExecution, SessionSummary
from kairos.display import KairosDisplay

logger = logging.getLogger(__name__)


@dataclass
class TrackerStatus:
    """Current tracker status snapshot."""
    session_id: str
    elapsed_seconds: float
    total_cost_usd: float
    hourly_rate: float
    cell_count: int
    cloud_provider: str
    instance_type: Optional[str]
    gpu_available: bool
    gpu_utilization: Optional[float]
    memory_used_mb: Optional[int]
    memory_total_mb: Optional[int]
    is_tracking: bool


class KairosTracker:
    """
    Main interface for ML cost tracking.

    The KairosTracker provides a simple, intuitive API for tracking GPU costs
    in Jupyter notebooks. It automatically detects GPU hardware, calculates
    costs based on cloud pricing, and provides rich visual summaries.

    Attributes:
        config: Configuration object with pricing and settings
        session_id: Unique identifier for this tracking session

    Example:
        >>> # Basic usage
        >>> tracker = KairosTracker()
        >>> tracker.status()

        >>> # With custom configuration
        >>> tracker = KairosTracker.for_aws("p4d.24xlarge")
        >>> tracker.status()

        >>> # Track a cell execution
        >>> with tracker.track_cell(tags=["training"]):
        ...     train_model()

        >>> # View summary
        >>> tracker.summary()
    """

    def __init__(
        self,
        config: Optional[KairosConfig] = None,
        auto_sample: bool = True,
        sample_interval_seconds: float = 5.0,
    ):
        """
        Initialize the Kairos tracker.

        Args:
            config: Optional configuration. If None, auto-detects from environment.
            auto_sample: Whether to automatically sample GPU utilization in background
            sample_interval_seconds: Interval between automatic GPU samples
        """
        self.config = config or KairosConfig.from_env()
        self._gpu_monitor = GPUMonitor()
        self._calculator = CostCalculator(self.config, self._gpu_monitor)
        self._display = KairosDisplay(enable_html=self.config.enable_html_output)

        # Auto-detection of GPU type
        if self.config.auto_detect_gpu and self._gpu_monitor.is_available:
            primary_gpu = self._gpu_monitor.get_primary_gpu()
            if primary_gpu:
                self.config._detected_gpu_type = primary_gpu.gpu_type
                logger.info(f"Detected GPU: {primary_gpu.name} ({primary_gpu.gpu_type.value})")

        # Background sampling
        self._auto_sample = auto_sample
        self._sample_interval = sample_interval_seconds
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_sampling = threading.Event()

        if auto_sample:
            self._start_sampling()

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info(
            f"Kairos tracker initialized. Session: {self.session_id}, "
            f"Rate: ${self.hourly_rate:.2f}/hr, GPU available: {self._gpu_monitor.is_available}"
        )

    # =========================================================================
    # Class Methods for Common Configurations
    # =========================================================================

    @classmethod
    def for_aws(
        cls,
        instance_type: str,
        **kwargs,
    ) -> "KairosTracker":
        """
        Create tracker for AWS instance.

        Args:
            instance_type: AWS instance type (e.g., 'p4d.24xlarge')
            **kwargs: Additional arguments passed to __init__

        Example:
            >>> tracker = KairosTracker.for_aws("p4d.24xlarge")
        """
        config = KairosConfig.for_aws(instance_type)
        return cls(config=config, **kwargs)

    @classmethod
    def for_gcp(
        cls,
        instance_type: str,
        **kwargs,
    ) -> "KairosTracker":
        """
        Create tracker for GCP instance.

        Args:
            instance_type: GCP instance type (e.g., 'a2-highgpu-8g')
            **kwargs: Additional arguments passed to __init__

        Example:
            >>> tracker = KairosTracker.for_gcp("a2-highgpu-8g")
        """
        config = KairosConfig.for_gcp(instance_type)
        return cls(config=config, **kwargs)

    @classmethod
    def for_azure(
        cls,
        instance_type: str,
        **kwargs,
    ) -> "KairosTracker":
        """
        Create tracker for Azure instance.

        Args:
            instance_type: Azure instance type (e.g., 'NC24ads_A100_v4')
            **kwargs: Additional arguments passed to __init__

        Example:
            >>> tracker = KairosTracker.for_azure("NC24ads_A100_v4")
        """
        config = KairosConfig.for_azure(instance_type)
        return cls(config=config, **kwargs)

    @classmethod
    def with_rate(
        cls,
        hourly_rate: float,
        **kwargs,
    ) -> "KairosTracker":
        """
        Create tracker with custom hourly rate.

        Args:
            hourly_rate: Custom hourly rate in USD
            **kwargs: Additional arguments passed to __init__

        Example:
            >>> tracker = KairosTracker.with_rate(5.00)  # $5/hour
        """
        config = KairosConfig(custom_hourly_rate=hourly_rate)
        return cls(config=config, **kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._calculator.session_id

    @property
    def hourly_rate(self) -> float:
        """Get current hourly rate in USD."""
        return self.config.get_hourly_rate()

    @property
    def total_cost(self) -> float:
        """Get total cost so far in USD."""
        return self._calculator.total_cost

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self._calculator.elapsed_seconds

    @property
    def cell_count(self) -> int:
        """Get number of cells executed."""
        return len(self._calculator.cells)

    @property
    def gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._gpu_monitor.is_available

    # =========================================================================
    # Core Methods
    # =========================================================================

    def status(self) -> TrackerStatus:
        """
        Display and return current tracking status.

        Shows a rich visual display in Jupyter notebooks with current cost,
        runtime, GPU utilization, and projections.

        Returns:
            TrackerStatus object with current state

        Example:
            >>> tracker.status()
            # Displays visual status panel in Jupyter
        """
        snapshot = self._gpu_monitor.get_snapshot() if self._gpu_monitor.is_available else None

        status = TrackerStatus(
            session_id=self.session_id,
            elapsed_seconds=self.elapsed_seconds,
            total_cost_usd=self.total_cost,
            hourly_rate=self.hourly_rate,
            cell_count=self.cell_count,
            cloud_provider=self.config.cloud_provider.value,
            instance_type=self.config.instance_type,
            gpu_available=self._gpu_monitor.is_available,
            gpu_utilization=snapshot.total_utilization if snapshot and snapshot.gpus else None,
            memory_used_mb=snapshot.total_memory_used_mb if snapshot else None,
            memory_total_mb=snapshot.total_memory_total_mb if snapshot else None,
            is_tracking=self._calculator.current_cell is not None,
        )

        # Render display
        self._display.render_status(
            session_id=status.session_id,
            elapsed_seconds=status.elapsed_seconds,
            total_cost=status.total_cost_usd,
            hourly_rate=status.hourly_rate,
            gpu_available=status.gpu_available,
            gpu_utilization=status.gpu_utilization,
            memory_used_mb=status.memory_used_mb,
            memory_total_mb=status.memory_total_mb,
            cell_count=status.cell_count,
            cloud_provider=status.cloud_provider,
            instance_type=status.instance_type,
        )

        return status

    def log_cell(
        self,
        duration: Optional[float] = None,
        cell_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
        gpu_utilization: Optional[float] = None,
        memory_used_mb: Optional[int] = None,
        show: bool = True,
    ) -> CellExecution:
        """
        Log a cell execution manually.

        Use this to record a cell execution after it completes. For automatic
        timing, use the track_cell() context manager instead.

        Args:
            duration: Execution duration in seconds. If None, uses 0.
            cell_number: Optional Jupyter cell number
            tags: Optional list of tags for categorization
            gpu_utilization: Optional GPU utilization percentage (0-100)
            memory_used_mb: Optional memory used in MB
            show: Whether to display the cell log entry

        Returns:
            CellExecution record

        Example:
            >>> import time
            >>> start = time.time()
            >>> train_model()
            >>> tracker.log_cell(
            ...     duration=time.time() - start,
            ...     tags=["training", "bert"]
            ... )
        """
        # Get current GPU stats if not provided
        if gpu_utilization is None or memory_used_mb is None:
            snapshot = self._gpu_monitor.get_snapshot() if self._gpu_monitor.is_available else None
            if snapshot and snapshot.gpus:
                gpu_utilization = gpu_utilization or snapshot.total_utilization
                memory_used_mb = memory_used_mb or snapshot.total_memory_used_mb

        cell = self._calculator.log_cell(
            duration_seconds=duration or 0.0,
            cell_number=cell_number,
            tags=tags,
            gpu_utilization=gpu_utilization or 0.0,
            memory_used_mb=memory_used_mb or 0,
        )

        if show:
            self._display.render_cell_log(cell)

        return cell

    @contextmanager
    def track_cell(
        self,
        cell_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
        show: bool = True,
    ) -> Generator[CellExecution, None, None]:
        """
        Context manager for tracking cell execution.

        Automatically tracks timing and GPU utilization for code within the
        context block.

        Args:
            cell_number: Optional Jupyter cell number
            tags: Optional list of tags for categorization
            show: Whether to display the cell log entry when complete

        Yields:
            CellExecution object (updates as execution progresses)

        Example:
            >>> with tracker.track_cell(tags=["training"]) as cell:
            ...     train_model()
            >>> print(f"Cost: ${cell.cost_usd:.4f}")
        """
        cell = self._calculator.start_cell(cell_number=cell_number, tags=tags)
        error = None

        try:
            yield cell
        except Exception as e:
            error = str(e)
            raise
        finally:
            completed_cell = self._calculator.end_cell(error=error)
            if show and completed_cell:
                self._display.render_cell_log(completed_cell)

    def summary(
        self,
        show_top_cells: int = 5,
        show_recommendations: bool = True,
    ) -> SessionSummary:
        """
        Display and return session summary.

        Shows a comprehensive summary of the tracking session including total
        cost, runtime, GPU utilization, top expensive cells, and optimization
        recommendations.

        Args:
            show_top_cells: Number of most expensive cells to show (0 to hide)
            show_recommendations: Whether to show optimization recommendations

        Returns:
            SessionSummary object with aggregated statistics

        Example:
            >>> tracker.summary()
            # Displays visual summary panel in Jupyter
        """
        summary = self._calculator.get_summary()
        top_cells = self._calculator.get_top_cells(show_top_cells) if show_top_cells > 0 else None

        self._display.render_summary(
            summary=summary,
            top_cells=top_cells,
            show_recommendations=show_recommendations,
        )

        return summary

    # =========================================================================
    # Additional Utility Methods
    # =========================================================================

    def get_projection(self, hours: float = 1.0) -> Dict[str, float]:
        """
        Get cost projections.

        Args:
            hours: Number of hours to project forward

        Returns:
            Dictionary with cost projections

        Example:
            >>> proj = tracker.get_projection(hours=8)
            >>> print(f"8-hour projection: ${proj['projected_8h_usd']:.2f}")
        """
        return self._calculator.get_cost_projection(hours)

    def get_cell_costs(self) -> List[Dict[str, Any]]:
        """
        Get detailed costs for all logged cells.

        Returns:
            List of cell execution dictionaries

        Example:
            >>> cells = tracker.get_cell_costs()
            >>> total = sum(c['cost_usd'] for c in cells)
        """
        return self._calculator.get_cell_costs()

    def get_gpu_snapshot(self) -> Optional[GPUSnapshot]:
        """
        Get current GPU state snapshot.

        Returns:
            GPUSnapshot with current GPU stats, or None if unavailable

        Example:
            >>> snap = tracker.get_gpu_snapshot()
            >>> if snap:
            ...     print(f"GPU Utilization: {snap.total_utilization}%")
        """
        if not self._gpu_monitor.is_available:
            return None
        return self._gpu_monitor.get_snapshot()

    def alert(self, message: str, alert_type: str = "warning") -> None:
        """
        Display an alert message.

        Args:
            message: Alert message to display
            alert_type: Type of alert (warning, error, info)

        Example:
            >>> tracker.alert("Cost exceeds budget!", alert_type="error")
        """
        self._display.render_alert(message, alert_type)

    def check_budget(self, budget_usd: float) -> bool:
        """
        Check if current cost exceeds budget.

        Args:
            budget_usd: Budget threshold in USD

        Returns:
            True if cost exceeds budget

        Example:
            >>> if tracker.check_budget(10.00):
            ...     print("Budget exceeded!")
        """
        if self.total_cost >= budget_usd:
            self.alert(
                f"Budget exceeded! Current: ${self.total_cost:.2f}, Budget: ${budget_usd:.2f}",
                alert_type="error"
            )
            return True
        return False

    def set_alert_threshold(self, threshold_usd: float) -> None:
        """
        Set an automatic cost alert threshold.

        Args:
            threshold_usd: Alert when cost exceeds this value

        Example:
            >>> tracker.set_alert_threshold(50.00)
        """
        self.config.alert_threshold_usd = threshold_usd
        logger.info(f"Alert threshold set to ${threshold_usd:.2f}")

    def reset(self) -> None:
        """
        Reset the tracker for a new session.

        Clears all logged cells and resets the timer. Useful for starting
        fresh without creating a new tracker instance.

        Example:
            >>> tracker.reset()
            >>> tracker.status()  # Fresh session
        """
        self._calculator.reset()
        logger.info(f"Tracker reset. New session: {self.session_id}")

    def end_session(self) -> SessionSummary:
        """
        End the tracking session and get final summary.

        Returns:
            Final session summary

        Example:
            >>> summary = tracker.end_session()
            >>> print(f"Total cost: ${summary.total_cost_usd:.2f}")
        """
        self._stop_sampling.set()
        return self._calculator.end_session()

    def to_dict(self) -> Dict[str, Any]:
        """
        Export tracker state as dictionary.

        Useful for logging, serialization, or sending to external systems.

        Returns:
            Dictionary with tracker state

        Example:
            >>> data = tracker.to_dict()
            >>> import json
            >>> json.dumps(data)
        """
        summary = self._calculator.get_summary()
        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "summary": summary.to_dict(),
            "cells": self.get_cell_costs(),
            "projections": self.get_projection(),
        }

    # =========================================================================
    # Background Sampling
    # =========================================================================

    def _start_sampling(self) -> None:
        """Start background GPU sampling thread."""
        if self._sampling_thread is not None and self._sampling_thread.is_alive():
            return

        self._stop_sampling.clear()
        self._sampling_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="kairos-sampler",
        )
        self._sampling_thread.start()
        logger.debug("Background sampling started")

    def _sampling_loop(self) -> None:
        """Background loop for GPU sampling."""
        while not self._stop_sampling.is_set():
            try:
                self._calculator.sample_utilization()

                # Check alerts
                if self.config.alert_threshold_usd and self._calculator.check_alert():
                    self.alert(
                        f"Cost alert! Current: ${self.total_cost:.2f} "
                        f"exceeds threshold: ${self.config.alert_threshold_usd:.2f}",
                        alert_type="warning"
                    )

            except Exception as e:
                logger.debug(f"Sampling error: {e}")

            self._stop_sampling.wait(self._sample_interval)

    def _cleanup(self) -> None:
        """Cleanup resources on exit."""
        self._stop_sampling.set()
        if self._sampling_thread:
            self._sampling_thread.join(timeout=1.0)
        self._gpu_monitor.shutdown()

    # =========================================================================
    # Magic Methods
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"KairosTracker(session={self.session_id}, "
            f"cost=${self.total_cost:.4f}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"cells={self.cell_count})"
        )

    def __enter__(self) -> "KairosTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_session()
