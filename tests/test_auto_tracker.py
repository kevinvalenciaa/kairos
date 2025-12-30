"""
Tests for AutoTracker and automatic cell tracking.

Run with: pytest tests/test_auto_tracker.py -v
"""

import time
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kairos_plugin"))

from kairos.auto_tracker import AutoTracker, auto_track, stop_tracking, get_tracker
from kairos.tracker import KairosTracker
from kairos.config import KairosConfig


class MockIPython:
    """Mock IPython instance for testing."""

    def __init__(self):
        self.events = MockEvents()


class MockEvents:
    """Mock IPython events manager."""

    def __init__(self):
        self._callbacks = {}

    def register(self, event, callback):
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def unregister(self, event, callback):
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def trigger(self, event, *args):
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(*args)


class MockCellInfo:
    """Mock cell info object."""

    def __init__(self, raw_cell="print('test')", store_history=True):
        self.raw_cell = raw_cell
        self.store_history = store_history


class MockResult:
    """Mock execution result."""

    def __init__(self, error=None):
        self.error_in_exec = error
        self.error_before_exec = None


class TestAutoTracker:
    """Tests for AutoTracker class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        AutoTracker._instance = None
        yield
        AutoTracker._instance = None

    def test_singleton_pattern(self):
        """Test that AutoTracker is a singleton."""
        tracker1 = AutoTracker(quiet=True)
        tracker2 = AutoTracker(quiet=True)
        assert tracker1 is tracker2

    def test_initialization(self):
        """Test auto tracker initialization."""
        auto = AutoTracker(quiet=True)
        assert auto.tracker is not None
        assert not auto.is_registered

    def test_tag_extraction_explicit(self):
        """Test tag extraction from explicit comments."""
        auto = AutoTracker(quiet=True)

        code = """
        # @kairos: training, bert
        model.fit(X, y)
        """
        tags = auto._extract_tags(code)
        assert "training" in tags
        assert "bert" in tags

    def test_tag_extraction_auto_detect(self):
        """Test automatic tag detection from code."""
        auto = AutoTracker(quiet=True)

        code = "model.fit(X, y)"
        tags = auto._extract_tags(code)
        assert "training" in tags

        code = "model.predict(X)"
        tags = auto._extract_tags(code)
        assert "inference" in tags

        code = "plt.figure()"
        tags = auto._extract_tags(code)
        assert "visualization" in tags

    def test_start_with_mock_ipython(self):
        """Test starting auto tracker with mocked IPython."""
        mock_ip = MockIPython()

        auto = AutoTracker(quiet=True)
        auto._ipython = mock_ip

        # Register directly to simulate start()
        mock_ip.events.register("pre_run_cell", auto._on_pre_run_cell)
        mock_ip.events.register("post_run_cell", auto._on_post_run_cell)
        auto._registered = True

        assert auto.is_registered
        assert "pre_run_cell" in mock_ip.events._callbacks
        assert "post_run_cell" in mock_ip.events._callbacks

    def test_stop_with_mock_ipython(self):
        """Test stopping auto tracker."""
        mock_ip = MockIPython()

        auto = AutoTracker(quiet=True)
        auto._ipython = mock_ip
        mock_ip.events.register("pre_run_cell", auto._on_pre_run_cell)
        mock_ip.events.register("post_run_cell", auto._on_post_run_cell)
        auto._registered = True

        assert auto.is_registered

        auto.stop()
        assert not auto.is_registered

    def test_cell_tracking_flow_manual(self):
        """Test cell tracking through manual event simulation."""
        mock_ip = MockIPython()

        auto = AutoTracker(show_cell_output=False, quiet=True)
        auto._ipython = mock_ip
        auto._registered = True

        # Simulate cell execution
        cell_info = MockCellInfo(raw_cell="time.sleep(0.1)")
        auto._on_pre_run_cell(cell_info)

        time.sleep(0.1)

        result = MockResult()
        auto._on_post_run_cell(result)

        # Check cell was tracked
        assert len(auto.tracker._calculator.cells) == 1
        cell = auto.tracker._calculator.cells[0]
        assert cell.duration_seconds >= 0.1

    def test_cell_with_error_manual(self):
        """Test tracking cell that raises an error."""
        mock_ip = MockIPython()

        auto = AutoTracker(show_cell_output=False, quiet=True)
        auto._ipython = mock_ip
        auto._registered = True

        # Simulate cell execution with error
        cell_info = MockCellInfo(raw_cell="raise ValueError('test')")
        auto._on_pre_run_cell(cell_info)

        result = MockResult(error=ValueError("test"))
        auto._on_post_run_cell(result)

        # Check error was captured
        assert len(auto.tracker._calculator.cells) == 1
        cell = auto.tracker._calculator.cells[0]
        assert cell.error is not None

    def test_multiple_cells(self):
        """Test tracking multiple cells in sequence."""
        auto = AutoTracker(show_cell_output=False, quiet=True)
        auto._registered = True

        # Simulate 3 cells
        for i in range(3):
            cell_info = MockCellInfo(raw_cell=f"cell_{i}")
            auto._on_pre_run_cell(cell_info)
            time.sleep(0.05)
            result = MockResult()
            auto._on_post_run_cell(result)

        assert len(auto.tracker._calculator.cells) == 3

    def test_callback_registration(self):
        """Test registering callbacks for cell completion."""
        auto = AutoTracker(show_cell_output=False, quiet=True)
        auto._registered = True

        callback_called = []

        def on_complete(cell):
            callback_called.append(cell)

        auto.on_cell_complete(on_complete)

        # Simulate cell
        auto._on_pre_run_cell(MockCellInfo())
        auto._on_post_run_cell(MockResult())

        assert len(callback_called) == 1


class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset module state before and after each test."""
        import kairos.auto_tracker as at
        at._auto_tracker = None
        AutoTracker._instance = None
        yield
        at._auto_tracker = None
        AutoTracker._instance = None

    def test_get_tracker_none(self):
        """Test get_tracker when no tracker exists."""
        assert get_tracker() is None

    def test_tracker_properties(self):
        """Test that tracker has expected properties."""
        auto = AutoTracker(quiet=True)

        assert hasattr(auto, 'tracker')
        assert hasattr(auto, 'is_registered')
        assert hasattr(auto, 'start')
        assert hasattr(auto, 'stop')
        assert hasattr(auto, 'status')
        assert hasattr(auto, 'summary')


class TestTagExtraction:
    """Tests for automatic tag extraction."""

    @pytest.fixture
    def auto_tracker(self):
        """Create auto tracker for testing."""
        AutoTracker._instance = None
        auto = AutoTracker(quiet=True)
        yield auto
        AutoTracker._instance = None

    def test_kairos_comment_tags(self, auto_tracker):
        """Test extraction from # @kairos: comments."""
        code = "# @kairos: training, bert, v1\nmodel.fit()"
        tags = auto_tracker._extract_tags(code)
        assert "training" in tags
        assert "bert" in tags
        assert "v1" in tags

    def test_tags_comment(self, auto_tracker):
        """Test extraction from # tags: comments."""
        code = "# tags: inference, production\nmodel.predict()"
        tags = auto_tracker._extract_tags(code)
        assert "inference" in tags
        assert "production" in tags

    def test_ml_keyword_detection(self, auto_tracker):
        """Test detection of ML-related keywords."""
        test_cases = [
            ("trainer.train()", "training"),
            ("model.eval()", "evaluation"),
            ("DataLoader(dataset)", "data-loading"),
            ("tokenizer(text)", "preprocessing"),
            ("sns.heatmap(data)", "visualization"),
        ]

        for code, expected_tag in test_cases:
            tags = auto_tracker._extract_tags(code)
            assert expected_tag in tags, f"Expected '{expected_tag}' in tags for code: {code}"

    def test_tag_limit(self, auto_tracker):
        """Test that tags are limited to 5."""
        code = """
        # @kairos: tag1, tag2, tag3, tag4, tag5, tag6, tag7
        model.fit()
        """
        tags = auto_tracker._extract_tags(code)
        assert len(tags) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
