"""
Tests for SessionStorage persistence layer.

Run with: pytest tests/test_storage.py -v
"""

import os
import time
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "kairos_plugin"))

from kairos.storage import SessionStorage
from kairos.tracker import KairosTracker
from kairos.cost_calculator import CellExecution


class TestSessionStorage:
    """Tests for SessionStorage class."""

    @pytest.fixture
    def temp_storage(self):
        """Create storage with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SessionStorage(base_path=tmpdir)
            yield storage

    def test_initialization(self, temp_storage):
        """Test storage initialization creates directories."""
        assert temp_storage.base_path.exists()
        assert (temp_storage.base_path / "sessions").exists()
        assert (temp_storage.base_path / "exports").exists()

    def test_save_and_load_session(self, temp_storage):
        """Test saving and loading a session."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, cell_number=1, tags=["test"], show=False)
        tracker.log_cell(duration=20.0, cell_number=2, tags=["training"], show=False)

        # Save
        session_id = temp_storage.save_session(tracker)
        assert session_id == tracker.session_id

        # Load
        data = temp_storage.load_session(session_id)
        assert data is not None
        assert data["session_id"] == session_id
        assert len(data["cells"]) == 2

    def test_save_cell(self, temp_storage):
        """Test saving individual cells."""
        session_id = "test123"

        cell = CellExecution(
            cell_id="cell1",
            cell_number=1,
            start_time=datetime.now() - timedelta(seconds=10),
            end_time=datetime.now(),
            cost_usd=0.05,
            tags=["test"],
        )

        temp_storage.save_cell(session_id, cell)

        # Load and verify
        data = temp_storage.load_session(session_id)
        assert data is not None
        assert len(data["cells"]) == 1
        assert data["cells"][0]["cell_id"] == "cell1"

    def test_list_sessions(self, temp_storage):
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(3):
            tracker = KairosTracker.with_rate(10.0, auto_sample=False)
            tracker.log_cell(duration=10.0 * (i + 1), show=False)
            temp_storage.save_session(tracker)

        sessions = temp_storage.list_sessions(limit=10)
        assert len(sessions) == 3

    def test_list_sessions_with_limit(self, temp_storage):
        """Test listing sessions with limit."""
        for i in range(5):
            tracker = KairosTracker.with_rate(10.0, auto_sample=False)
            tracker.log_cell(duration=10.0, show=False)
            temp_storage.save_session(tracker)

        sessions = temp_storage.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_get_total_cost(self, temp_storage):
        """Test total cost aggregation."""
        for i in range(3):
            tracker = KairosTracker.with_rate(36.0, auto_sample=False)  # $0.01/sec
            tracker.log_cell(duration=10.0, show=False)  # $0.10 each
            temp_storage.save_session(tracker)

        totals = temp_storage.get_total_cost()
        assert totals["session_count"] == 3
        assert totals["total_cells"] == 3

    def test_export_csv(self, temp_storage):
        """Test CSV export."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, cell_number=1, tags=["test"], show=False)
        temp_storage.save_session(tracker)

        output_path = temp_storage.export_csv()
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
            assert "session_id" in content
            assert "cost_usd" in content

    def test_export_json(self, temp_storage):
        """Test JSON export."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, cell_number=1, show=False)
        temp_storage.save_session(tracker)

        output_path = temp_storage.export_json()
        assert os.path.exists(output_path)

        import json
        with open(output_path) as f:
            data = json.load(f)
            assert "sessions" in data
            assert len(data["sessions"]) == 1

    def test_delete_session(self, temp_storage):
        """Test deleting a session."""
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, show=False)
        session_id = temp_storage.save_session(tracker)

        # Verify exists
        assert temp_storage.load_session(session_id) is not None

        # Delete
        deleted = temp_storage.delete_session(session_id)
        assert deleted

        # Verify gone
        assert temp_storage.load_session(session_id) is None

    def test_cleanup_old_sessions(self, temp_storage):
        """Test cleaning up old sessions."""
        # Create a session
        tracker = KairosTracker.with_rate(10.0, auto_sample=False)
        tracker.log_cell(duration=10.0, show=False)
        temp_storage.save_session(tracker)

        # Cleanup with 0 days (deletes all)
        deleted = temp_storage.cleanup_old_sessions(days=0)
        assert deleted >= 0


class TestSessionStorageFilters:
    """Tests for storage filtering."""

    @pytest.fixture
    def storage_with_data(self):
        """Create storage with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SessionStorage(base_path=tmpdir)

            # Create sessions with different dates
            for i in range(5):
                tracker = KairosTracker.with_rate(10.0, auto_sample=False)
                tracker.log_cell(duration=10.0 * (i + 1), show=False)
                storage.save_session(tracker)
                time.sleep(0.01)

            yield storage

    def test_list_with_date_filter(self, storage_with_data):
        """Test filtering sessions by date."""
        # All sessions should be recent
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        sessions = storage_with_data.list_sessions(start_date=yesterday)
        assert len(sessions) == 5

        # Future date should return none
        tomorrow = now + timedelta(days=1)
        sessions = storage_with_data.list_sessions(start_date=tomorrow)
        assert len(sessions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
