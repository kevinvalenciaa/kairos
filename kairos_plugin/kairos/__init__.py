"""
Kairos - AI/ML Cost Intelligence Layer

A Jupyter plugin for real-time GPU cost tracking and optimization.
Helps ML teams eliminate infrastructure waste from training to production.

Example:
    >>> from kairos import KairosTracker
    >>> tracker = KairosTracker()
    >>> tracker.status()
    >>> # ... run your ML code ...
    >>> tracker.summary()
"""

from kairos.tracker import KairosTracker
from kairos.config import KairosConfig
from kairos.cost_calculator import CostCalculator
from kairos.gpu_monitor import GPUMonitor

__version__ = "0.1.0"
__author__ = "Kairos Team"
__all__ = ["KairosTracker", "KairosConfig", "CostCalculator", "GPUMonitor"]
