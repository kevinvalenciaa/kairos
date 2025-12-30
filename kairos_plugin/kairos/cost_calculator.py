"""
Cost calculation engine for Kairos.

Provides real-time cost calculation, experiment tracking, and cost projections.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging

from kairos.config import KairosConfig, CloudProvider
from kairos.gpu_monitor import GPUSnapshot, GPUMonitor

logger = logging.getLogger(__name__)


class CostEventType(Enum):
    """Types of cost-related events."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CELL_START = "cell_start"
    CELL_END = "cell_end"
    CHECKPOINT = "checkpoint"
    IDLE_DETECTED = "idle_detected"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class CostEvent:
    """A single cost tracking event."""
    event_type: CostEventType
    timestamp: datetime
    elapsed_seconds: float
    cost_usd: float
    gpu_utilization: Optional[float] = None
    memory_used_mb: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "cost_usd": round(self.cost_usd, 4),
            "gpu_utilization": self.gpu_utilization,
            "memory_used_mb": self.memory_used_mb,
            "metadata": self.metadata,
        }


@dataclass
class CellExecution:
    """Tracks a single cell execution."""
    cell_id: str
    cell_number: Optional[int]
    start_time: datetime
    end_time: Optional[datetime] = None
    cost_usd: float = 0.0
    gpu_utilization_avg: float = 0.0
    gpu_utilization_max: float = 0.0
    memory_used_avg_mb: int = 0
    memory_used_max_mb: int = 0
    tags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    _utilization_samples: List[float] = field(default_factory=list, repr=False)
    _memory_samples: List[int] = field(default_factory=list, repr=False)

    @property
    def duration_seconds(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def is_complete(self) -> bool:
        """Check if cell execution is complete."""
        return self.end_time is not None

    def add_sample(self, utilization: float, memory_mb: int) -> None:
        """Add a GPU utilization sample."""
        self._utilization_samples.append(utilization)
        self._memory_samples.append(memory_mb)

    def finalize(self) -> None:
        """Calculate final statistics from samples."""
        if self._utilization_samples:
            self.gpu_utilization_avg = sum(self._utilization_samples) / len(self._utilization_samples)
            self.gpu_utilization_max = max(self._utilization_samples)

        if self._memory_samples:
            self.memory_used_avg_mb = int(sum(self._memory_samples) / len(self._memory_samples))
            self.memory_used_max_mb = max(self._memory_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cell_id": self.cell_id,
            "cell_number": self.cell_number,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "cost_usd": round(self.cost_usd, 4),
            "gpu_utilization_avg": round(self.gpu_utilization_avg, 1),
            "gpu_utilization_max": round(self.gpu_utilization_max, 1),
            "memory_used_avg_mb": self.memory_used_avg_mb,
            "memory_used_max_mb": self.memory_used_max_mb,
            "tags": self.tags,
            "error": self.error,
        }


@dataclass
class SessionSummary:
    """Summary of a tracking session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_cost_usd: float
    total_duration_seconds: float
    active_duration_seconds: float
    idle_duration_seconds: float
    cell_count: int
    avg_gpu_utilization: float
    peak_gpu_utilization: float
    avg_memory_used_mb: int
    peak_memory_used_mb: int
    hourly_rate: float
    cloud_provider: str
    instance_type: Optional[str]
    cost_per_cell_avg: float
    efficiency_score: float  # 0-100, based on utilization vs cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "active_duration_seconds": round(self.active_duration_seconds, 2),
            "idle_duration_seconds": round(self.idle_duration_seconds, 2),
            "cell_count": self.cell_count,
            "avg_gpu_utilization": round(self.avg_gpu_utilization, 1),
            "peak_gpu_utilization": round(self.peak_gpu_utilization, 1),
            "avg_memory_used_mb": self.avg_memory_used_mb,
            "peak_memory_used_mb": self.peak_memory_used_mb,
            "hourly_rate": self.hourly_rate,
            "cloud_provider": self.cloud_provider,
            "instance_type": self.instance_type,
            "cost_per_cell_avg": round(self.cost_per_cell_avg, 4),
            "efficiency_score": round(self.efficiency_score, 1),
        }


class CostCalculator:
    """
    Calculate and track costs for ML workloads.

    Provides real-time cost calculation, cell-level tracking, and session summaries.
    """

    def __init__(self, config: KairosConfig, gpu_monitor: Optional[GPUMonitor] = None):
        """
        Initialize cost calculator.

        Args:
            config: Kairos configuration with pricing information
            gpu_monitor: Optional GPU monitor instance
        """
        self.config = config
        self.gpu_monitor = gpu_monitor or GPUMonitor()

        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()
        self.session_end: Optional[datetime] = None

        # Cost tracking
        self.events: List[CostEvent] = []
        self.cells: List[CellExecution] = []
        self._current_cell: Optional[CellExecution] = None

        # Running totals
        self._total_cost = 0.0
        self._idle_seconds = 0.0
        self._last_activity = datetime.now()

        # Utilization tracking
        self._utilization_samples: List[float] = []
        self._memory_samples: List[int] = []

        # Record session start
        self._record_event(CostEventType.SESSION_START)

    @property
    def hourly_rate(self) -> float:
        """Get current hourly rate."""
        return self.config.get_hourly_rate()

    @property
    def cost_per_second(self) -> float:
        """Get cost per second."""
        return self.hourly_rate / 3600

    @property
    def elapsed_seconds(self) -> float:
        """Get total elapsed seconds since session start."""
        end = self.session_end or datetime.now()
        return (end - self.session_start).total_seconds()

    @property
    def total_cost(self) -> float:
        """Get total cost for this session."""
        return self.elapsed_seconds * self.cost_per_second

    @property
    def current_cell(self) -> Optional[CellExecution]:
        """Get currently executing cell if any."""
        return self._current_cell

    def _record_event(
        self,
        event_type: CostEventType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostEvent:
        """Record a cost event."""
        snapshot = self.gpu_monitor.get_snapshot() if self.gpu_monitor.is_available else None

        event = CostEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            elapsed_seconds=self.elapsed_seconds,
            cost_usd=self.total_cost,
            gpu_utilization=snapshot.total_utilization if snapshot else None,
            memory_used_mb=snapshot.total_memory_used_mb if snapshot else None,
            metadata=metadata or {},
        )

        self.events.append(event)

        # Track utilization samples
        if snapshot:
            self._utilization_samples.append(snapshot.total_utilization)
            self._memory_samples.append(snapshot.total_memory_used_mb)

        return event

    def start_cell(
        self,
        cell_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> CellExecution:
        """
        Start tracking a cell execution.

        Args:
            cell_number: Optional cell number (from Jupyter)
            tags: Optional tags for categorization

        Returns:
            CellExecution instance
        """
        # Finalize any previous cell
        if self._current_cell and not self._current_cell.is_complete:
            self.end_cell(error="Interrupted by new cell")

        cell = CellExecution(
            cell_id=str(uuid.uuid4())[:8],
            cell_number=cell_number,
            start_time=datetime.now(),
            tags=tags or [],
        )

        self._current_cell = cell
        self._last_activity = datetime.now()
        self._record_event(CostEventType.CELL_START, {"cell_id": cell.cell_id})

        return cell

    def end_cell(self, error: Optional[str] = None) -> Optional[CellExecution]:
        """
        End tracking the current cell execution.

        Args:
            error: Optional error message if cell failed

        Returns:
            Completed CellExecution or None
        """
        if not self._current_cell:
            return None

        cell = self._current_cell
        cell.end_time = datetime.now()
        cell.error = error
        cell.cost_usd = cell.duration_seconds * self.cost_per_second
        cell.finalize()

        self.cells.append(cell)
        self._current_cell = None
        self._last_activity = datetime.now()

        self._record_event(CostEventType.CELL_END, {
            "cell_id": cell.cell_id,
            "duration_seconds": cell.duration_seconds,
            "cost_usd": cell.cost_usd,
        })

        return cell

    def log_cell(
        self,
        duration_seconds: float,
        cell_number: Optional[int] = None,
        tags: Optional[List[str]] = None,
        gpu_utilization: float = 0.0,
        memory_used_mb: int = 0,
    ) -> CellExecution:
        """
        Log a completed cell execution (for manual tracking).

        Args:
            duration_seconds: How long the cell took to execute
            cell_number: Optional cell number
            tags: Optional tags for categorization
            gpu_utilization: Average GPU utilization during execution
            memory_used_mb: Peak memory used

        Returns:
            CellExecution record
        """
        now = datetime.now()
        start = now - timedelta(seconds=duration_seconds)

        cell = CellExecution(
            cell_id=str(uuid.uuid4())[:8],
            cell_number=cell_number,
            start_time=start,
            end_time=now,
            cost_usd=duration_seconds * self.cost_per_second,
            gpu_utilization_avg=gpu_utilization,
            gpu_utilization_max=gpu_utilization,
            memory_used_avg_mb=memory_used_mb,
            memory_used_max_mb=memory_used_mb,
            tags=tags or [],
        )

        self.cells.append(cell)
        self._last_activity = now

        self._record_event(CostEventType.CELL_END, {
            "cell_id": cell.cell_id,
            "duration_seconds": duration_seconds,
            "cost_usd": cell.cost_usd,
            "manual": True,
        })

        return cell

    def sample_utilization(self) -> Optional[GPUSnapshot]:
        """
        Take a GPU utilization sample for the current cell.

        Returns:
            Current GPU snapshot or None
        """
        if not self.gpu_monitor.is_available:
            return None

        snapshot = self.gpu_monitor.get_snapshot()

        if self._current_cell:
            self._current_cell.add_sample(
                snapshot.total_utilization,
                snapshot.total_memory_used_mb,
            )

        self._utilization_samples.append(snapshot.total_utilization)
        self._memory_samples.append(snapshot.total_memory_used_mb)

        return snapshot

    def check_idle(self, threshold_minutes: int = 5) -> bool:
        """
        Check if the session has been idle.

        Args:
            threshold_minutes: Minutes of inactivity to consider idle

        Returns:
            True if idle beyond threshold
        """
        idle_seconds = (datetime.now() - self._last_activity).total_seconds()

        if idle_seconds > threshold_minutes * 60:
            self._record_event(CostEventType.IDLE_DETECTED, {
                "idle_seconds": idle_seconds,
                "threshold_minutes": threshold_minutes,
            })
            return True

        return False

    def check_alert(self) -> bool:
        """
        Check if cost alert threshold has been exceeded.

        Returns:
            True if alert should be triggered
        """
        if self.config.alert_threshold_usd is None:
            return False

        if self.total_cost >= self.config.alert_threshold_usd:
            self._record_event(CostEventType.ALERT_TRIGGERED, {
                "threshold_usd": self.config.alert_threshold_usd,
                "current_cost_usd": self.total_cost,
            })
            return True

        return False

    def get_summary(self) -> SessionSummary:
        """
        Get summary statistics for the current session.

        Returns:
            SessionSummary with aggregated statistics
        """
        # Calculate active vs idle time
        active_seconds = sum(c.duration_seconds for c in self.cells)
        idle_seconds = self.elapsed_seconds - active_seconds

        # Calculate averages
        avg_utilization = (
            sum(self._utilization_samples) / len(self._utilization_samples)
            if self._utilization_samples else 0.0
        )
        peak_utilization = max(self._utilization_samples) if self._utilization_samples else 0.0

        avg_memory = (
            int(sum(self._memory_samples) / len(self._memory_samples))
            if self._memory_samples else 0
        )
        peak_memory = max(self._memory_samples) if self._memory_samples else 0

        # Calculate cost per cell
        cost_per_cell = (
            self.total_cost / len(self.cells) if self.cells else 0.0
        )

        # Calculate efficiency score (0-100)
        # Based on: utilization rate, active time ratio
        utilization_score = min(avg_utilization, 100)
        active_ratio = (active_seconds / self.elapsed_seconds * 100) if self.elapsed_seconds > 0 else 0
        efficiency_score = (utilization_score * 0.6 + active_ratio * 0.4)

        return SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start,
            end_time=self.session_end,
            total_cost_usd=self.total_cost,
            total_duration_seconds=self.elapsed_seconds,
            active_duration_seconds=active_seconds,
            idle_duration_seconds=idle_seconds,
            cell_count=len(self.cells),
            avg_gpu_utilization=avg_utilization,
            peak_gpu_utilization=peak_utilization,
            avg_memory_used_mb=avg_memory,
            peak_memory_used_mb=peak_memory,
            hourly_rate=self.hourly_rate,
            cloud_provider=self.config.cloud_provider.value,
            instance_type=self.config.instance_type,
            cost_per_cell_avg=cost_per_cell,
            efficiency_score=efficiency_score,
        )

    def get_cost_projection(self, hours: float = 1.0) -> Dict[str, float]:
        """
        Project costs forward based on current rate.

        Args:
            hours: Number of hours to project

        Returns:
            Dictionary with projections
        """
        current = self.total_cost
        hourly = self.hourly_rate

        return {
            "current_cost_usd": round(current, 4),
            "projected_1h_usd": round(current + hourly * hours, 2),
            "projected_8h_usd": round(current + hourly * 8, 2),
            "projected_24h_usd": round(current + hourly * 24, 2),
            "hourly_rate_usd": hourly,
            "daily_rate_usd": hourly * 24,
            "monthly_rate_usd": hourly * 24 * 30,
        }

    def get_cell_costs(self) -> List[Dict[str, Any]]:
        """Get cost breakdown by cell."""
        return [cell.to_dict() for cell in self.cells]

    def get_top_cells(self, n: int = 5) -> List[CellExecution]:
        """Get top N most expensive cells."""
        return sorted(self.cells, key=lambda c: c.cost_usd, reverse=True)[:n]

    def end_session(self) -> SessionSummary:
        """
        End the tracking session.

        Returns:
            Final session summary
        """
        # End any active cell
        if self._current_cell:
            self.end_cell(error="Session ended")

        self.session_end = datetime.now()
        self._record_event(CostEventType.SESSION_END)

        return self.get_summary()

    def reset(self) -> None:
        """Reset the calculator for a new session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()
        self.session_end = None
        self.events = []
        self.cells = []
        self._current_cell = None
        self._total_cost = 0.0
        self._idle_seconds = 0.0
        self._last_activity = datetime.now()
        self._utilization_samples = []
        self._memory_samples = []

        self._record_event(CostEventType.SESSION_START)
