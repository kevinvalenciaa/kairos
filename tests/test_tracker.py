"""
Tests for KairosTracker.

Run with: pytest tests/test_tracker.py -v
"""

import time
import pytest
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "kairos_plugin"))

from kairos import KairosTracker, KairosConfig
from kairos.config import CloudProvider, GPUType
from kairos.cost_calculator import CostCalculator, CellExecution
from kairos.gpu_monitor import GPUMonitor, GPUInfo, GPUSnapshot


class TestKairosConfig:
    """Tests for KairosConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KairosConfig()
        assert config.cloud_provider == CloudProvider.LOCAL
        assert config.instance_type is None
        assert config.custom_hourly_rate is None
        assert config.auto_detect_gpu is True
        assert config.currency == "USD"

    def test_for_aws(self):
        """Test AWS configuration factory."""
        config = KairosConfig.for_aws("p4d.24xlarge")
        assert config.cloud_provider == CloudProvider.AWS
        assert config.instance_type == "p4d.24xlarge"

    def test_for_gcp(self):
        """Test GCP configuration factory."""
        config = KairosConfig.for_gcp("a2-highgpu-8g")
        assert config.cloud_provider == CloudProvider.GCP
        assert config.instance_type == "a2-highgpu-8g"

    def test_for_azure(self):
        """Test Azure configuration factory."""
        config = KairosConfig.for_azure("NC24ads_A100_v4")
        assert config.cloud_provider == CloudProvider.AZURE
        assert config.instance_type == "NC24ads_A100_v4"

    def test_get_hourly_rate_custom(self):
        """Test custom hourly rate takes precedence."""
        config = KairosConfig(custom_hourly_rate=10.0)
        assert config.get_hourly_rate() == 10.0

    def test_get_hourly_rate_aws(self):
        """Test AWS pricing lookup."""
        config = KairosConfig.for_aws("p4d.24xlarge")
        rate = config.get_hourly_rate()
        assert rate == 32.77  # Known price for p4d.24xlarge

    def test_get_hourly_rate_fallback(self):
        """Test fallback to unknown rate."""
        config = KairosConfig()
        rate = config.get_hourly_rate()
        assert rate == 0.35  # Default unknown rate

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        config = KairosConfig(
            cloud_provider=CloudProvider.AWS,
            instance_type="p4d.24xlarge",
            custom_hourly_rate=30.0,
            alert_threshold_usd=100.0,
        )
        data = config.to_dict()
        restored = KairosConfig.from_dict(data)

        assert restored.cloud_provider == config.cloud_provider
        assert restored.instance_type == config.instance_type
        assert restored.custom_hourly_rate == config.custom_hourly_rate
        assert restored.alert_threshold_usd == config.alert_threshold_usd

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict("os.environ", {
            "KAIROS_CLOUD_PROVIDER": "aws",
            "KAIROS_INSTANCE_TYPE": "p4d.24xlarge",
            "KAIROS_HOURLY_RATE": "25.0",
            "KAIROS_ALERT_THRESHOLD": "50.0",
        }):
            config = KairosConfig.from_env()
            assert config.cloud_provider == CloudProvider.AWS
            assert config.instance_type == "p4d.24xlarge"
            assert config.custom_hourly_rate == 25.0
            assert config.alert_threshold_usd == 50.0


class TestCostCalculator:
    """Tests for CostCalculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        config = KairosConfig(custom_hourly_rate=10.0)
        calc = CostCalculator(config)

        assert calc.session_id is not None
        assert len(calc.session_id) == 8
        assert calc.hourly_rate == 10.0
        assert calc.cost_per_second == 10.0 / 3600

    def test_elapsed_seconds(self):
        """Test elapsed time tracking."""
        config = KairosConfig()
        calc = CostCalculator(config)

        time.sleep(0.1)
        elapsed = calc.elapsed_seconds
        assert elapsed >= 0.1

    def test_total_cost(self):
        """Test cost calculation."""
        config = KairosConfig(custom_hourly_rate=36.0)  # $0.01/second
        calc = CostCalculator(config)

        time.sleep(0.1)
        cost = calc.total_cost
        assert cost >= 0.001  # At least 0.1 seconds * $0.01/second

    def test_log_cell(self):
        """Test manual cell logging."""
        config = KairosConfig(custom_hourly_rate=36.0)
        calc = CostCalculator(config)

        cell = calc.log_cell(
            duration_seconds=10.0,
            cell_number=1,
            tags=["test"],
            gpu_utilization=50.0,
            memory_used_mb=1000,
        )

        assert cell.cell_id is not None
        assert cell.cell_number == 1
        assert cell.duration_seconds == 10.0
        assert cell.cost_usd == pytest.approx(0.10, rel=0.01)  # 10s * $0.01/s
        assert cell.tags == ["test"]
        assert len(calc.cells) == 1

    def test_start_end_cell(self):
        """Test cell start/end tracking."""
        config = KairosConfig(custom_hourly_rate=36.0)
        calc = CostCalculator(config)

        cell = calc.start_cell(cell_number=1, tags=["training"])
        assert calc.current_cell == cell
        assert not cell.is_complete

        time.sleep(0.1)
        completed = calc.end_cell()

        assert completed is not None
        assert completed.is_complete
        assert completed.duration_seconds >= 0.1
        assert calc.current_cell is None
        assert len(calc.cells) == 1

    def test_get_summary(self):
        """Test session summary generation."""
        config = KairosConfig(custom_hourly_rate=10.0)
        calc = CostCalculator(config)

        calc.log_cell(duration_seconds=10.0)
        calc.log_cell(duration_seconds=20.0)

        summary = calc.get_summary()

        assert summary.session_id == calc.session_id
        assert summary.cell_count == 2
        assert summary.hourly_rate == 10.0
        assert summary.total_cost_usd > 0

    def test_get_cost_projection(self):
        """Test cost projections."""
        config = KairosConfig(custom_hourly_rate=10.0)
        calc = CostCalculator(config)

        proj = calc.get_cost_projection(hours=1.0)

        assert "current_cost_usd" in proj
        assert "projected_1h_usd" in proj
        assert "projected_8h_usd" in proj
        assert "projected_24h_usd" in proj
        assert proj["hourly_rate_usd"] == 10.0

    def test_reset(self):
        """Test calculator reset."""
        config = KairosConfig()
        calc = CostCalculator(config)
        original_session = calc.session_id

        calc.log_cell(duration_seconds=10.0)
        assert len(calc.cells) == 1

        calc.reset()

        assert calc.session_id != original_session
        assert len(calc.cells) == 0


class TestKairosTracker:
    """Tests for KairosTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = KairosTracker(auto_sample=False)

        assert tracker.session_id is not None
        assert tracker.hourly_rate > 0
        assert tracker.total_cost >= 0
        assert tracker.cell_count == 0

    def test_for_aws(self):
        """Test AWS factory method."""
        tracker = KairosTracker.for_aws("p4d.24xlarge", auto_sample=False)

        assert tracker.config.cloud_provider == CloudProvider.AWS
        assert tracker.config.instance_type == "p4d.24xlarge"
        assert tracker.hourly_rate == 32.77

    def test_for_gcp(self):
        """Test GCP factory method."""
        tracker = KairosTracker.for_gcp("a2-highgpu-8g", auto_sample=False)

        assert tracker.config.cloud_provider == CloudProvider.GCP
        assert tracker.config.instance_type == "a2-highgpu-8g"

    def test_with_rate(self):
        """Test custom rate factory method."""
        tracker = KairosTracker.with_rate(5.0, auto_sample=False)

        assert tracker.hourly_rate == 5.0

    def test_log_cell(self):
        """Test manual cell logging."""
        tracker = KairosTracker.with_rate(36.0, auto_sample=False)

        cell = tracker.log_cell(
            duration=10.0,
            cell_number=1,
            tags=["test"],
            show=False,
        )

        assert cell.cell_number == 1
        assert cell.duration_seconds == 10.0
        assert cell.cost_usd == pytest.approx(0.10, rel=0.01)
        assert tracker.cell_count == 1

    def test_track_cell_context_manager(self):
        """Test track_cell context manager."""
        tracker = KairosTracker.with_rate(36.0, auto_sample=False)

        with tracker.track_cell(tags=["test"], show=False) as cell:
            time.sleep(0.1)

        assert cell.is_complete
        assert cell.duration_seconds >= 0.1
        assert tracker.cell_count == 1

    def test_track_cell_with_exception(self):
        """Test track_cell handles exceptions."""
        tracker = KairosTracker.with_rate(36.0, auto_sample=False)

        with pytest.raises(ValueError):
            with tracker.track_cell(show=False) as cell:
                raise ValueError("Test error")

        assert cell.is_complete
        assert cell.error == "Test error"
        assert tracker.cell_count == 1

    def test_get_projection(self):
        """Test cost projections."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)

        proj = tracker.get_projection(hours=1.0)

        assert "current_cost_usd" in proj
        assert "projected_1h_usd" in proj
        assert proj["hourly_rate_usd"] == 10.0

    def test_check_budget(self):
        """Test budget checking."""
        tracker = KairosTracker.with_rate(3600.0, auto_sample=False)  # $1/second

        time.sleep(0.1)

        # Should not exceed very high budget
        assert not tracker.check_budget(1000.0)

    def test_set_alert_threshold(self):
        """Test setting alert threshold."""
        tracker = KairosTracker(auto_sample=False)

        tracker.set_alert_threshold(50.0)
        assert tracker.config.alert_threshold_usd == 50.0

    def test_reset(self):
        """Test tracker reset."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        original_session = tracker.session_id

        tracker.log_cell(duration=10.0, show=False)
        assert tracker.cell_count == 1

        tracker.reset()

        assert tracker.session_id != original_session
        assert tracker.cell_count == 0

    def test_to_dict(self):
        """Test export to dictionary."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, show=False)

        data = tracker.to_dict()

        assert "session_id" in data
        assert "config" in data
        assert "summary" in data
        assert "cells" in data
        assert "projections" in data
        assert len(data["cells"]) == 1

    def test_repr(self):
        """Test string representation."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        rep = repr(tracker)

        assert "KairosTracker" in rep
        assert "session=" in rep
        assert "cost=" in rep

    def test_context_manager(self):
        """Test tracker as context manager."""
        with KairosTracker.with_rate(10.0, auto_sample=False) as tracker:
            tracker.log_cell(duration=10.0, show=False)
            assert tracker.cell_count == 1

        # Session should be ended after context


class TestGPUMonitor:
    """Tests for GPUMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = GPUMonitor()
        # Should not raise, even without GPU
        assert isinstance(monitor.is_available, bool)

    def test_get_snapshot_no_gpu(self):
        """Test snapshot when no GPU available."""
        monitor = GPUMonitor()
        if not monitor.is_available:
            snapshot = monitor.get_snapshot()
            assert snapshot.gpu_count == 0
            assert snapshot.total_utilization == 0.0


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_memory_utilization(self):
        """Test memory utilization calculation."""
        info = GPUInfo(
            index=0,
            name="Test GPU",
            gpu_type=GPUType.A100_40GB,
            memory_total_mb=40960,
            memory_used_mb=20480,
            memory_free_mb=20480,
            utilization_percent=50.0,
        )

        assert info.memory_utilization_percent == 50.0

    def test_is_idle(self):
        """Test idle detection."""
        idle_gpu = GPUInfo(
            index=0,
            name="Test GPU",
            gpu_type=GPUType.A100_40GB,
            memory_total_mb=40960,
            memory_used_mb=1000,
            memory_free_mb=39960,
            utilization_percent=2.0,
        )

        active_gpu = GPUInfo(
            index=0,
            name="Test GPU",
            gpu_type=GPUType.A100_40GB,
            memory_total_mb=40960,
            memory_used_mb=20000,
            memory_free_mb=20960,
            utilization_percent=80.0,
        )

        assert idle_gpu.is_idle
        assert not active_gpu.is_idle

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = GPUInfo(
            index=0,
            name="Test GPU",
            gpu_type=GPUType.A100_40GB,
            memory_total_mb=40960,
            memory_used_mb=20480,
            memory_free_mb=20480,
            utilization_percent=50.0,
        )

        data = info.to_dict()
        assert data["index"] == 0
        assert data["name"] == "Test GPU"
        assert data["gpu_type"] == "a100_40gb"
        assert data["utilization_percent"] == 50.0


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete tracking workflow."""
        tracker = KairosTracker.with_rate(36.0, auto_sample=False)

        # Log some cells
        tracker.log_cell(duration=10.0, cell_number=1, tags=["prep"], show=False)

        with tracker.track_cell(cell_number=2, tags=["training"], show=False):
            time.sleep(0.1)

        tracker.log_cell(duration=5.0, cell_number=3, tags=["eval"], show=False)

        # Get summary
        summary = tracker._calculator.get_summary()

        assert summary.cell_count == 3
        assert summary.total_cost_usd > 0

        # Get cell costs
        cells = tracker.get_cell_costs()
        assert len(cells) == 3

        # Export
        data = tracker.to_dict()
        assert len(data["cells"]) == 3

        # End session
        final_summary = tracker.end_session()
        assert final_summary.end_time is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
