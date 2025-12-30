"""
Kairos - AI/ML Cost Intelligence Layer

A Jupyter plugin for real-time GPU cost tracking and optimization.
Helps ML teams eliminate infrastructure waste from training to production.

Quick Start (Automatic Tracking):
    >>> import kairos
    >>> kairos.auto_track()  # All cells automatically tracked!
    >>>
    >>> # ... run your ML code normally ...
    >>>
    >>> kairos.summary()  # View session summary

Manual Tracking:
    >>> from kairos import KairosTracker
    >>> tracker = KairosTracker()
    >>> tracker.status()
    >>>
    >>> with tracker.track_cell(tags=["training"]):
    ...     model.fit(X, y)
    >>>
    >>> tracker.summary()

Cloud Configuration:
    >>> import kairos
    >>> kairos.auto_track(provider="aws", instance="p4d.24xlarge")

    >>> # Or manually:
    >>> tracker = KairosTracker.for_aws("p4d.24xlarge")
    >>> tracker = KairosTracker.for_gcp("a2-highgpu-8g")
    >>> tracker = KairosTracker.with_rate(5.00)  # Custom $5/hour
"""

__version__ = "0.1.0"
__author__ = "Kairos Team"

# Core classes
from kairos.tracker import KairosTracker
from kairos.config import KairosConfig, CloudProvider, GPUType
from kairos.cost_calculator import CostCalculator, CellExecution, SessionSummary
from kairos.gpu_monitor import GPUMonitor, GPUInfo, GPUSnapshot

# Auto-tracking functions
from kairos.auto_tracker import (
    auto_track,
    stop_tracking,
    get_tracker,
    status,
    summary,
    AutoTracker,
)

# Storage
from kairos.storage import SessionStorage

# Display
from kairos.display import KairosDisplay

__all__ = [
    # Version
    "__version__",
    # Core classes
    "KairosTracker",
    "KairosConfig",
    "CloudProvider",
    "GPUType",
    "CostCalculator",
    "CellExecution",
    "SessionSummary",
    "GPUMonitor",
    "GPUInfo",
    "GPUSnapshot",
    # Auto-tracking
    "auto_track",
    "stop_tracking",
    "get_tracker",
    "status",
    "summary",
    "AutoTracker",
    # Storage
    "SessionStorage",
    # Display
    "KairosDisplay",
]


# Convenience functions at module level
def track(
    hourly_rate: float = None,
    provider: str = None,
    instance: str = None,
    **kwargs,
) -> AutoTracker:
    """
    Alias for auto_track(). Start automatic cell tracking.

    Args:
        hourly_rate: Custom hourly rate in USD
        provider: Cloud provider ("aws", "gcp", "azure", "local")
        instance: Instance type (e.g., "p4d.24xlarge")
        **kwargs: Additional arguments passed to auto_track

    Returns:
        AutoTracker instance

    Example:
        >>> import kairos
        >>> kairos.track()  # Start tracking
    """
    return auto_track(
        hourly_rate=hourly_rate,
        provider=provider,
        instance=instance,
        **kwargs,
    )


def stop() -> None:
    """
    Stop automatic cell tracking.

    Example:
        >>> kairos.stop()
    """
    stop_tracking()


def pause() -> None:
    """
    Pause tracking temporarily.

    Example:
        >>> kairos.pause()
        >>> # ... cells not tracked ...
        >>> kairos.resume()
    """
    from kairos.auto_tracker import _auto_tracker
    if _auto_tracker:
        _auto_tracker.pause()


def resume() -> None:
    """
    Resume tracking after pause.

    Example:
        >>> kairos.resume()
    """
    from kairos.auto_tracker import _auto_tracker
    if _auto_tracker:
        _auto_tracker.resume()


def history(limit: int = 10) -> list:
    """
    Get session history.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of session summaries

    Example:
        >>> sessions = kairos.history()
        >>> for s in sessions:
        ...     print(f"{s['session_id']}: ${s['total_cost_usd']:.2f}")
    """
    storage = SessionStorage()
    return storage.list_sessions(limit=limit)


def export(format: str = "csv", output_path: str = None) -> str:
    """
    Export all session data.

    Args:
        format: Export format ("csv" or "json")
        output_path: Optional output file path

    Returns:
        Path to exported file

    Example:
        >>> path = kairos.export(format="csv")
        >>> print(f"Exported to: {path}")
    """
    storage = SessionStorage()
    if format == "json":
        return storage.export_json(output_path=output_path)
    return storage.export_csv(output_path=output_path)


def total_cost(days: int = None) -> dict:
    """
    Get total costs across all sessions.

    Args:
        days: Only include sessions from last N days

    Returns:
        Dictionary with cost summary

    Example:
        >>> totals = kairos.total_cost(days=30)
        >>> print(f"Total: ${totals['total_cost_usd']:.2f}")
    """
    from datetime import datetime, timedelta

    storage = SessionStorage()
    start_date = None
    if days:
        start_date = datetime.now() - timedelta(days=days)
    return storage.get_total_cost(start_date=start_date)
