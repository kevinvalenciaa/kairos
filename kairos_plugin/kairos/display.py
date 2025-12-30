"""
Display utilities for Kairos.

Provides rich HTML output for Jupyter notebooks and terminal formatting.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import html

from kairos.cost_calculator import SessionSummary, CellExecution


def _is_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        return False
    except (NameError, AttributeError):
        return False


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    elif cost < 100:
        return f"${cost:.2f}"
    else:
        return f"${cost:,.2f}"


def _format_memory(mb: int) -> str:
    """Format memory in human-readable format."""
    if mb < 1024:
        return f"{mb} MB"
    else:
        return f"{mb / 1024:.1f} GB"


def _get_utilization_color(percent: float) -> str:
    """Get color based on utilization percentage."""
    if percent >= 80:
        return "#22c55e"  # Green
    elif percent >= 50:
        return "#eab308"  # Yellow
    elif percent >= 20:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red


def _get_cost_color(cost: float, hourly_rate: float) -> str:
    """Get color based on cost relative to hourly rate."""
    hourly_threshold = hourly_rate
    if cost < hourly_threshold * 0.1:
        return "#22c55e"  # Green
    elif cost < hourly_threshold * 0.5:
        return "#eab308"  # Yellow
    elif cost < hourly_threshold:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red


# CSS styles for Kairos displays
KAIROS_CSS = """
<style>
.kairos-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    color: #e0e0e0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.kairos-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.kairos-logo {
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kairos-session-id {
    font-size: 12px;
    color: #888;
    font-family: monospace;
}
.kairos-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
}
.kairos-metric {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 16px;
}
.kairos-metric-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.kairos-metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #fff;
}
.kairos-metric-sub {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}
.kairos-progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 8px;
}
.kairos-progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.kairos-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
}
.kairos-table th {
    text-align: left;
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.kairos-table td {
    padding: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 14px;
}
.kairos-table tr:hover {
    background: rgba(255,255,255,0.02);
}
.kairos-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
    margin-right: 4px;
}
.kairos-alert {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.kairos-alert-icon {
    font-size: 20px;
}
.kairos-footer {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.1);
    font-size: 11px;
    color: #666;
    display: flex;
    justify-content: space-between;
}
.kairos-recommendations {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 8px;
    padding: 16px;
    margin-top: 16px;
}
.kairos-recommendations h4 {
    color: #22c55e;
    margin: 0 0 12px 0;
    font-size: 14px;
}
.kairos-recommendations ul {
    margin: 0;
    padding-left: 20px;
}
.kairos-recommendations li {
    margin-bottom: 8px;
    color: #a0a0a0;
}
</style>
"""


class KairosDisplay:
    """
    Rich display output for Kairos in Jupyter notebooks.

    Automatically detects environment and renders appropriate output.
    """

    def __init__(self, enable_html: bool = True):
        """
        Initialize display handler.

        Args:
            enable_html: Whether to use HTML output in notebooks
        """
        self.enable_html = enable_html and _is_notebook()

    def _display_html(self, html_content: str) -> None:
        """Display HTML content in Jupyter."""
        from IPython.display import display, HTML
        display(HTML(KAIROS_CSS + html_content))

    def render_status(
        self,
        session_id: str,
        elapsed_seconds: float,
        total_cost: float,
        hourly_rate: float,
        gpu_available: bool,
        gpu_utilization: Optional[float] = None,
        memory_used_mb: Optional[int] = None,
        memory_total_mb: Optional[int] = None,
        cell_count: int = 0,
        cloud_provider: str = "local",
        instance_type: Optional[str] = None,
    ) -> Optional[str]:
        """
        Render current status display.

        Returns:
            HTML string if in notebook, None otherwise (prints to terminal)
        """
        if not self.enable_html:
            return self._render_status_terminal(
                session_id, elapsed_seconds, total_cost, hourly_rate,
                gpu_available, gpu_utilization, memory_used_mb, memory_total_mb,
                cell_count, cloud_provider, instance_type
            )

        # Calculate projections
        projected_1h = total_cost + hourly_rate
        projected_24h = total_cost + hourly_rate * 24

        # GPU info section
        if gpu_available and gpu_utilization is not None:
            util_color = _get_utilization_color(gpu_utilization)
            memory_pct = (memory_used_mb / memory_total_mb * 100) if memory_total_mb else 0
            gpu_section = f"""
            <div class="kairos-metric">
                <div class="kairos-metric-label">GPU Utilization</div>
                <div class="kairos-metric-value" style="color: {util_color}">{gpu_utilization:.1f}%</div>
                <div class="kairos-progress-bar">
                    <div class="kairos-progress-fill" style="width: {gpu_utilization}%; background: {util_color}"></div>
                </div>
            </div>
            <div class="kairos-metric">
                <div class="kairos-metric-label">GPU Memory</div>
                <div class="kairos-metric-value">{_format_memory(memory_used_mb or 0)}</div>
                <div class="kairos-metric-sub">of {_format_memory(memory_total_mb or 0)} ({memory_pct:.1f}%)</div>
            </div>
            """
        else:
            gpu_section = f"""
            <div class="kairos-metric">
                <div class="kairos-metric-label">GPU Status</div>
                <div class="kairos-metric-value" style="color: #888">{'N/A' if not gpu_available else 'Idle'}</div>
                <div class="kairos-metric-sub">{'No GPU detected' if not gpu_available else 'No active workload'}</div>
            </div>
            """

        instance_info = f" ({instance_type})" if instance_type else ""

        html_content = f"""
        <div class="kairos-container">
            <div class="kairos-header">
                <span class="kairos-logo">Kairos</span>
                <span class="kairos-session-id">Session: {html.escape(session_id)} | {cloud_provider.upper()}{html.escape(instance_info)}</span>
            </div>

            <div class="kairos-grid">
                <div class="kairos-metric">
                    <div class="kairos-metric-label">Current Cost</div>
                    <div class="kairos-metric-value" style="color: {_get_cost_color(total_cost, hourly_rate)}">{_format_cost(total_cost)}</div>
                    <div class="kairos-metric-sub">@ {_format_cost(hourly_rate)}/hour</div>
                </div>

                <div class="kairos-metric">
                    <div class="kairos-metric-label">Runtime</div>
                    <div class="kairos-metric-value">{_format_duration(elapsed_seconds)}</div>
                    <div class="kairos-metric-sub">{cell_count} cells executed</div>
                </div>

                {gpu_section}

                <div class="kairos-metric">
                    <div class="kairos-metric-label">Projections</div>
                    <div class="kairos-metric-value">{_format_cost(projected_1h)}</div>
                    <div class="kairos-metric-sub">+1h | +24h: {_format_cost(projected_24h)}</div>
                </div>
            </div>

            <div class="kairos-footer">
                <span>Updated: {datetime.now().strftime('%H:%M:%S')}</span>
                <span>usekairos.ai</span>
            </div>
        </div>
        """

        self._display_html(html_content)
        return html_content

    def _render_status_terminal(
        self,
        session_id: str,
        elapsed_seconds: float,
        total_cost: float,
        hourly_rate: float,
        gpu_available: bool,
        gpu_utilization: Optional[float],
        memory_used_mb: Optional[int],
        memory_total_mb: Optional[int],
        cell_count: int,
        cloud_provider: str,
        instance_type: Optional[str],
    ) -> None:
        """Render status for terminal output."""
        print("\n" + "=" * 60)
        print(f"  KAIROS - Cost Tracking Status")
        print("=" * 60)
        print(f"  Session:    {session_id}")
        print(f"  Provider:   {cloud_provider.upper()}" + (f" ({instance_type})" if instance_type else ""))
        print("-" * 60)
        print(f"  Cost:       {_format_cost(total_cost)} @ {_format_cost(hourly_rate)}/hr")
        print(f"  Runtime:    {_format_duration(elapsed_seconds)}")
        print(f"  Cells:      {cell_count} executed")

        if gpu_available and gpu_utilization is not None:
            print("-" * 60)
            print(f"  GPU Util:   {gpu_utilization:.1f}%")
            if memory_used_mb and memory_total_mb:
                print(f"  GPU Memory: {_format_memory(memory_used_mb)} / {_format_memory(memory_total_mb)}")
        elif not gpu_available:
            print("-" * 60)
            print("  GPU:        Not detected")

        print("-" * 60)
        print(f"  +1h:        {_format_cost(total_cost + hourly_rate)}")
        print(f"  +24h:       {_format_cost(total_cost + hourly_rate * 24)}")
        print("=" * 60 + "\n")

    def render_summary(
        self,
        summary: SessionSummary,
        top_cells: Optional[List[CellExecution]] = None,
        show_recommendations: bool = True,
    ) -> Optional[str]:
        """
        Render session summary display.

        Args:
            summary: Session summary data
            top_cells: Optional list of most expensive cells
            show_recommendations: Whether to show optimization recommendations

        Returns:
            HTML string if in notebook, None otherwise
        """
        if not self.enable_html:
            return self._render_summary_terminal(summary, top_cells, show_recommendations)

        # Calculate efficiency bar
        efficiency_color = _get_utilization_color(summary.efficiency_score)

        # Build top cells table
        cells_html = ""
        if top_cells:
            rows = ""
            for i, cell in enumerate(top_cells[:5], 1):
                tags_html = "".join(f'<span class="kairos-tag">{html.escape(t)}</span>' for t in cell.tags)
                rows += f"""
                <tr>
                    <td>#{cell.cell_number or '?'}</td>
                    <td>{_format_cost(cell.cost_usd)}</td>
                    <td>{_format_duration(cell.duration_seconds)}</td>
                    <td>{cell.gpu_utilization_avg:.1f}%</td>
                    <td>{tags_html or '-'}</td>
                </tr>
                """

            cells_html = f"""
            <table class="kairos-table">
                <thead>
                    <tr>
                        <th>Cell</th>
                        <th>Cost</th>
                        <th>Duration</th>
                        <th>GPU Util</th>
                        <th>Tags</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            """

        # Build recommendations
        recommendations_html = ""
        if show_recommendations:
            recs = self._generate_recommendations(summary)
            if recs:
                rec_items = "".join(f"<li>{html.escape(r)}</li>" for r in recs)
                recommendations_html = f"""
                <div class="kairos-recommendations">
                    <h4>Optimization Recommendations</h4>
                    <ul>{rec_items}</ul>
                </div>
                """

        # Calculate waste metrics
        idle_pct = (summary.idle_duration_seconds / summary.total_duration_seconds * 100) if summary.total_duration_seconds > 0 else 0
        idle_cost = summary.idle_duration_seconds * (summary.hourly_rate / 3600)

        html_content = f"""
        <div class="kairos-container">
            <div class="kairos-header">
                <span class="kairos-logo">Kairos</span>
                <span class="kairos-session-id">Session Summary: {html.escape(summary.session_id)}</span>
            </div>

            <div class="kairos-grid">
                <div class="kairos-metric">
                    <div class="kairos-metric-label">Total Cost</div>
                    <div class="kairos-metric-value">{_format_cost(summary.total_cost_usd)}</div>
                    <div class="kairos-metric-sub">{_format_cost(summary.cost_per_cell_avg)}/cell avg</div>
                </div>

                <div class="kairos-metric">
                    <div class="kairos-metric-label">Total Runtime</div>
                    <div class="kairos-metric-value">{_format_duration(summary.total_duration_seconds)}</div>
                    <div class="kairos-metric-sub">{summary.cell_count} cells executed</div>
                </div>

                <div class="kairos-metric">
                    <div class="kairos-metric-label">Avg GPU Utilization</div>
                    <div class="kairos-metric-value" style="color: {_get_utilization_color(summary.avg_gpu_utilization)}">{summary.avg_gpu_utilization:.1f}%</div>
                    <div class="kairos-metric-sub">Peak: {summary.peak_gpu_utilization:.1f}%</div>
                    <div class="kairos-progress-bar">
                        <div class="kairos-progress-fill" style="width: {summary.avg_gpu_utilization}%; background: {_get_utilization_color(summary.avg_gpu_utilization)}"></div>
                    </div>
                </div>

                <div class="kairos-metric">
                    <div class="kairos-metric-label">Efficiency Score</div>
                    <div class="kairos-metric-value" style="color: {efficiency_color}">{summary.efficiency_score:.0f}</div>
                    <div class="kairos-metric-sub">out of 100</div>
                    <div class="kairos-progress-bar">
                        <div class="kairos-progress-fill" style="width: {summary.efficiency_score}%; background: {efficiency_color}"></div>
                    </div>
                </div>
            </div>

            <div class="kairos-grid" style="grid-template-columns: repeat(3, 1fr);">
                <div class="kairos-metric">
                    <div class="kairos-metric-label">Active Time</div>
                    <div class="kairos-metric-value">{_format_duration(summary.active_duration_seconds)}</div>
                </div>
                <div class="kairos-metric">
                    <div class="kairos-metric-label">Idle Time</div>
                    <div class="kairos-metric-value" style="color: #ef4444">{_format_duration(summary.idle_duration_seconds)}</div>
                    <div class="kairos-metric-sub">{idle_pct:.1f}% of session</div>
                </div>
                <div class="kairos-metric">
                    <div class="kairos-metric-label">Idle Cost</div>
                    <div class="kairos-metric-value" style="color: #ef4444">{_format_cost(idle_cost)}</div>
                    <div class="kairos-metric-sub">potential waste</div>
                </div>
            </div>

            {cells_html}
            {recommendations_html}

            <div class="kairos-footer">
                <span>Session: {summary.start_time.strftime('%Y-%m-%d %H:%M')} - {summary.end_time.strftime('%H:%M') if summary.end_time else 'ongoing'}</span>
                <span>{summary.cloud_provider.upper()}{(' | ' + summary.instance_type) if summary.instance_type else ''} | usekairos.ai</span>
            </div>
        </div>
        """

        self._display_html(html_content)
        return html_content

    def _render_summary_terminal(
        self,
        summary: SessionSummary,
        top_cells: Optional[List[CellExecution]],
        show_recommendations: bool,
    ) -> None:
        """Render summary for terminal output."""
        print("\n" + "=" * 70)
        print("  KAIROS - Session Summary")
        print("=" * 70)
        print(f"  Session ID:      {summary.session_id}")
        print(f"  Provider:        {summary.cloud_provider.upper()}" +
              (f" ({summary.instance_type})" if summary.instance_type else ""))
        print("-" * 70)
        print(f"  Total Cost:      {_format_cost(summary.total_cost_usd)}")
        print(f"  Total Runtime:   {_format_duration(summary.total_duration_seconds)}")
        print(f"  Cells Executed:  {summary.cell_count}")
        print(f"  Cost/Cell Avg:   {_format_cost(summary.cost_per_cell_avg)}")
        print("-" * 70)
        print(f"  Active Time:     {_format_duration(summary.active_duration_seconds)}")
        print(f"  Idle Time:       {_format_duration(summary.idle_duration_seconds)}")
        print(f"  Efficiency:      {summary.efficiency_score:.0f}/100")
        print("-" * 70)
        print(f"  Avg GPU Util:    {summary.avg_gpu_utilization:.1f}%")
        print(f"  Peak GPU Util:   {summary.peak_gpu_utilization:.1f}%")
        print(f"  Avg Memory:      {_format_memory(summary.avg_memory_used_mb)}")
        print(f"  Peak Memory:     {_format_memory(summary.peak_memory_used_mb)}")

        if top_cells:
            print("-" * 70)
            print("  Top Expensive Cells:")
            for i, cell in enumerate(top_cells[:5], 1):
                tags = ", ".join(cell.tags) if cell.tags else "-"
                print(f"    {i}. Cell #{cell.cell_number or '?'}: {_format_cost(cell.cost_usd)} "
                      f"({_format_duration(cell.duration_seconds)}, {cell.gpu_utilization_avg:.1f}% GPU)")

        if show_recommendations:
            recs = self._generate_recommendations(summary)
            if recs:
                print("-" * 70)
                print("  Recommendations:")
                for rec in recs:
                    print(f"    * {rec}")

        print("=" * 70 + "\n")

    def _generate_recommendations(self, summary: SessionSummary) -> List[str]:
        """Generate optimization recommendations based on session data."""
        recs = []

        # Low utilization warning
        if summary.avg_gpu_utilization < 30:
            recs.append(
                f"GPU utilization is low ({summary.avg_gpu_utilization:.0f}%). "
                "Consider using a smaller instance or batching workloads."
            )

        # High idle time warning
        idle_pct = (summary.idle_duration_seconds / summary.total_duration_seconds * 100) if summary.total_duration_seconds > 0 else 0
        if idle_pct > 30:
            idle_cost = summary.idle_duration_seconds * (summary.hourly_rate / 3600)
            recs.append(
                f"High idle time ({idle_pct:.0f}%). "
                f"Auto-pause could save {_format_cost(idle_cost)}/session."
            )

        # Suggest spot instances for long sessions
        if summary.total_duration_seconds > 3600 and summary.cloud_provider != "local":
            spot_savings = summary.hourly_rate * 0.7 * (summary.total_duration_seconds / 3600)
            recs.append(
                f"Long session detected. Spot instances could save ~{_format_cost(spot_savings)}."
            )

        # Suggest right-sizing
        if summary.peak_gpu_utilization < 50 and summary.avg_gpu_utilization < 30:
            recs.append(
                "Peak utilization never exceeded 50%. Consider a smaller GPU instance."
            )

        # Suggest scheduling for off-peak
        hour = datetime.now().hour
        if 9 <= hour <= 17 and summary.total_cost_usd > 10:
            recs.append(
                "Running during peak hours. Scheduling for off-peak (nights/weekends) "
                "could reduce spot instance costs by 20-40%."
            )

        return recs

    def render_cell_log(
        self,
        cell: CellExecution,
    ) -> Optional[str]:
        """
        Render a cell execution log entry.

        Args:
            cell: Cell execution data

        Returns:
            HTML string if in notebook
        """
        if not self.enable_html:
            print(f"[Kairos] Cell #{cell.cell_number or '?'}: "
                  f"{_format_cost(cell.cost_usd)} | "
                  f"{_format_duration(cell.duration_seconds)} | "
                  f"GPU: {cell.gpu_utilization_avg:.1f}%")
            return None

        tags_html = "".join(f'<span class="kairos-tag">{html.escape(t)}</span>' for t in cell.tags)
        error_html = f'<span style="color: #ef4444"> (Error: {html.escape(cell.error)})</span>' if cell.error else ""

        html_content = f"""
        <div style="
            background: rgba(30, 30, 46, 0.9);
            border-left: 3px solid {'#ef4444' if cell.error else '#60a5fa'};
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 0 4px 4px 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 13px;
            color: #e0e0e0;
        ">
            <strong style="color: #60a5fa">Kairos</strong> |
            Cell #{cell.cell_number or '?'}:
            <span style="color: #22c55e">{_format_cost(cell.cost_usd)}</span> |
            {_format_duration(cell.duration_seconds)} |
            GPU: {cell.gpu_utilization_avg:.1f}%
            {tags_html}
            {error_html}
        </div>
        """

        self._display_html(html_content)
        return html_content

    def render_alert(
        self,
        message: str,
        alert_type: str = "warning",
    ) -> Optional[str]:
        """
        Render an alert message.

        Args:
            message: Alert message
            alert_type: Type of alert (warning, error, info)

        Returns:
            HTML string if in notebook
        """
        if not self.enable_html:
            print(f"[Kairos {alert_type.upper()}] {message}")
            return None

        colors = {
            "warning": ("#eab308", "rgba(234, 179, 8, 0.1)"),
            "error": ("#ef4444", "rgba(239, 68, 68, 0.1)"),
            "info": ("#60a5fa", "rgba(96, 165, 250, 0.1)"),
        }
        color, bg = colors.get(alert_type, colors["info"])

        html_content = f"""
        <div style="
            background: {bg};
            border: 1px solid {color};
            border-radius: 8px;
            padding: 12px 16px;
            margin: 8px 0;
            color: {color};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <strong>Kairos Alert:</strong> {html.escape(message)}
        </div>
        """

        self._display_html(html_content)
        return html_content
