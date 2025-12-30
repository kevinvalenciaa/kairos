"""
GPU monitoring utilities for Kairos.

Provides real-time GPU detection, utilization monitoring, and memory tracking.
Supports NVIDIA GPUs via nvidia-smi and pynvml.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import subprocess
import re
import logging
from enum import Enum

from kairos.config import GPUType

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    gpu_type: GPUType
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: float
    temperature_celsius: Optional[int] = None
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None

    @property
    def memory_utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        if self.memory_total_mb > 0:
            return (self.memory_used_mb / self.memory_total_mb) * 100
        return 0.0

    @property
    def is_idle(self) -> bool:
        """Check if GPU is considered idle (< 5% utilization)."""
        return self.utilization_percent < 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "index": self.index,
            "name": self.name,
            "gpu_type": self.gpu_type.value,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_utilization_percent": round(self.memory_utilization_percent, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "temperature_celsius": self.temperature_celsius,
            "power_draw_watts": self.power_draw_watts,
            "is_idle": self.is_idle,
        }


@dataclass
class GPUSnapshot:
    """A point-in-time snapshot of all GPU states."""
    timestamp: datetime
    gpus: List[GPUInfo]

    @property
    def total_utilization(self) -> float:
        """Average utilization across all GPUs."""
        if not self.gpus:
            return 0.0
        return sum(g.utilization_percent for g in self.gpus) / len(self.gpus)

    @property
    def total_memory_used_mb(self) -> int:
        """Total memory used across all GPUs."""
        return sum(g.memory_used_mb for g in self.gpus)

    @property
    def total_memory_total_mb(self) -> int:
        """Total memory available across all GPUs."""
        return sum(g.memory_total_mb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        """Number of GPUs detected."""
        return len(self.gpus)

    @property
    def all_idle(self) -> bool:
        """Check if all GPUs are idle."""
        return all(g.is_idle for g in self.gpus) if self.gpus else True


class GPUMonitor:
    """
    Monitor GPU status and utilization.

    Supports NVIDIA GPUs through nvidia-smi command or pynvml library.
    Falls back gracefully when no GPU is available.
    """

    def __init__(self, use_pynvml: bool = True):
        """
        Initialize GPU monitor.

        Args:
            use_pynvml: Whether to try using pynvml library (faster than nvidia-smi)
        """
        self._use_pynvml = use_pynvml
        self._pynvml_available = False
        self._nvidia_smi_available = False
        self._pynvml = None
        self._initialized = False
        self._gpu_count = 0

        self._detect_available_backends()

    def _detect_available_backends(self) -> None:
        """Detect which GPU monitoring backends are available."""
        # Try pynvml first (faster)
        if self._use_pynvml:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._gpu_count = pynvml.nvmlDeviceGetCount()
                self._pynvml = pynvml
                self._pynvml_available = True
                self._initialized = True
                logger.debug(f"pynvml initialized with {self._gpu_count} GPU(s)")
            except Exception as e:
                logger.debug(f"pynvml not available: {e}")

        # Fall back to nvidia-smi
        if not self._pynvml_available:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self._nvidia_smi_available = True
                    self._initialized = True
                    # Count GPUs from output
                    self._gpu_count = len(result.stdout.strip().split("\n"))
                    logger.debug(f"nvidia-smi available with {self._gpu_count} GPU(s)")
            except Exception as e:
                logger.debug(f"nvidia-smi not available: {e}")

        if not self._initialized:
            logger.info("No GPU monitoring backend available - running in CPU-only mode")

    @property
    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._initialized

    @property
    def gpu_count(self) -> int:
        """Get number of GPUs detected."""
        return self._gpu_count

    def _detect_gpu_type(self, name: str) -> GPUType:
        """Detect GPU type from device name string."""
        name_lower = name.lower()

        # NVIDIA H100
        if "h100" in name_lower:
            return GPUType.H100

        # NVIDIA A100 variants
        if "a100" in name_lower:
            if "80g" in name_lower or "80 g" in name_lower:
                return GPUType.A100_80GB
            return GPUType.A100_40GB

        # Other data center GPUs
        if "v100" in name_lower:
            return GPUType.V100
        if "t4" in name_lower:
            return GPUType.T4
        if "l4" in name_lower:
            return GPUType.L4
        if "a10g" in name_lower or "a10 g" in name_lower:
            return GPUType.A10G

        # Consumer GPUs
        if "4090" in name_lower:
            return GPUType.RTX_4090
        if "3090" in name_lower:
            return GPUType.RTX_3090
        if "3080" in name_lower:
            return GPUType.RTX_3080

        return GPUType.UNKNOWN

    def _get_gpu_info_pynvml(self, index: int) -> Optional[GPUInfo]:
        """Get GPU info using pynvml."""
        if not self._pynvml_available or not self._pynvml:
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(index)

            # Get basic info
            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Memory info
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Utilization
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Temperature (optional)
            try:
                temp = self._pynvml.nvmlDeviceGetTemperature(
                    handle, self._pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None

            # Power draw (optional)
            try:
                power = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = self._pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                power = None
                power_limit = None

            return GPUInfo(
                index=index,
                name=name,
                gpu_type=self._detect_gpu_type(name),
                memory_total_mb=mem_info.total // (1024 * 1024),
                memory_used_mb=mem_info.used // (1024 * 1024),
                memory_free_mb=mem_info.free // (1024 * 1024),
                utilization_percent=float(util.gpu),
                temperature_celsius=temp,
                power_draw_watts=power,
                power_limit_watts=power_limit,
            )
        except Exception as e:
            logger.warning(f"Failed to get GPU {index} info via pynvml: {e}")
            return None

    def _get_gpu_info_nvidia_smi(self) -> List[GPUInfo]:
        """Get GPU info using nvidia-smi command."""
        if not self._nvidia_smi_available:
            return []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                    "utilization.gpu,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue

                try:
                    index = int(parts[0])
                    name = parts[1]
                    mem_total = int(parts[2])
                    mem_used = int(parts[3])
                    mem_free = int(parts[4])
                    util = float(parts[5]) if parts[5] != "[N/A]" else 0.0
                    temp = int(parts[6]) if len(parts) > 6 and parts[6] != "[N/A]" else None
                    power = float(parts[7]) if len(parts) > 7 and parts[7] != "[N/A]" else None
                    power_limit = float(parts[8]) if len(parts) > 8 and parts[8] != "[N/A]" else None

                    gpus.append(GPUInfo(
                        index=index,
                        name=name,
                        gpu_type=self._detect_gpu_type(name),
                        memory_total_mb=mem_total,
                        memory_used_mb=mem_used,
                        memory_free_mb=mem_free,
                        utilization_percent=util,
                        temperature_celsius=temp,
                        power_draw_watts=power,
                        power_limit_watts=power_limit,
                    ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi output: {e}")
                    continue

            return gpus

        except Exception as e:
            logger.warning(f"nvidia-smi command failed: {e}")
            return []

    def get_snapshot(self) -> GPUSnapshot:
        """
        Get current GPU state snapshot.

        Returns:
            GPUSnapshot with current state of all GPUs
        """
        timestamp = datetime.now()
        gpus: List[GPUInfo] = []

        if self._pynvml_available:
            for i in range(self._gpu_count):
                info = self._get_gpu_info_pynvml(i)
                if info:
                    gpus.append(info)
        elif self._nvidia_smi_available:
            gpus = self._get_gpu_info_nvidia_smi()

        return GPUSnapshot(timestamp=timestamp, gpus=gpus)

    def get_primary_gpu(self) -> Optional[GPUInfo]:
        """Get info for the primary (index 0) GPU."""
        snapshot = self.get_snapshot()
        if snapshot.gpus:
            return snapshot.gpus[0]
        return None

    def get_utilization_history(
        self, duration_seconds: int = 60, interval_seconds: int = 1
    ) -> List[GPUSnapshot]:
        """
        Collect utilization history over a duration.

        Args:
            duration_seconds: Total duration to collect data
            interval_seconds: Interval between snapshots

        Returns:
            List of GPUSnapshots
        """
        import time

        history = []
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            history.append(self.get_snapshot())
            time.sleep(interval_seconds)

        return history

    def shutdown(self) -> None:
        """Clean up resources."""
        if self._pynvml_available and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._pynvml_available = False

    def __enter__(self) -> "GPUMonitor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def get_gpu_summary() -> Dict[str, Any]:
    """
    Quick utility to get a summary of available GPUs.

    Returns:
        Dictionary with GPU summary information
    """
    monitor = GPUMonitor()

    if not monitor.is_available:
        return {
            "available": False,
            "gpu_count": 0,
            "message": "No NVIDIA GPU detected",
        }

    snapshot = monitor.get_snapshot()

    return {
        "available": True,
        "gpu_count": snapshot.gpu_count,
        "total_memory_mb": snapshot.total_memory_total_mb,
        "used_memory_mb": snapshot.total_memory_used_mb,
        "average_utilization": round(snapshot.total_utilization, 1),
        "gpus": [g.to_dict() for g in snapshot.gpus],
    }
