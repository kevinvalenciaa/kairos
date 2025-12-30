"""
Kairos Command Line Interface.

Provides command-line access to Kairos functionality:
- View session history
- Export data
- Configure settings
- Monitor costs

Usage:
    kairos status          View current/recent session status
    kairos history         List all sessions
    kairos export          Export data to CSV/JSON
    kairos config          View/set configuration
    kairos cleanup         Clean up old session data
"""

import argparse
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def format_cost(cost: float) -> str:
    """Format cost with appropriate precision."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1:
        return f"${cost:.3f}"
    elif cost < 100:
        return f"${cost:.2f}"
    else:
        return f"${cost:,.2f}"


def format_duration(seconds: float) -> str:
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


def cmd_status(args):
    """Show status of current or recent session."""
    from kairos.storage import SessionStorage

    storage = SessionStorage()
    sessions = storage.list_sessions(limit=1)

    if not sessions:
        print("No sessions found.")
        print("\nTo start tracking, run in a Jupyter notebook:")
        print("  import kairos")
        print("  kairos.auto_track()")
        return

    session = sessions[0]
    data = storage.load_session(session["session_id"])

    print("\n" + "=" * 60)
    print("  KAIROS - Most Recent Session")
    print("=" * 60)
    print(f"  Session ID:     {session['session_id']}")
    print(f"  Started:        {session['start_time']}")
    print(f"  Provider:       {session.get('cloud_provider', 'local').upper()}")
    if session.get('instance_type'):
        print(f"  Instance:       {session['instance_type']}")
    print("-" * 60)
    print(f"  Total Cost:     {format_cost(session.get('total_cost_usd', 0))}")
    print(f"  Total Runtime:  {format_duration(session.get('total_duration_seconds', 0))}")
    print(f"  Cells:          {session.get('cell_count', 0)}")
    print(f"  Efficiency:     {session.get('efficiency_score', 0):.0f}/100")
    print("=" * 60 + "\n")


def cmd_history(args):
    """List session history."""
    from kairos.storage import SessionStorage

    storage = SessionStorage()

    # Parse date filters
    start_date = None
    end_date = None

    if args.since:
        try:
            if args.since.endswith("d"):
                days = int(args.since[:-1])
                start_date = datetime.now() - timedelta(days=days)
            elif args.since.endswith("w"):
                weeks = int(args.since[:-1])
                start_date = datetime.now() - timedelta(weeks=weeks)
            else:
                start_date = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"Invalid date format: {args.since}")
            return

    sessions = storage.list_sessions(
        limit=args.limit,
        start_date=start_date,
        end_date=end_date,
    )

    if not sessions:
        print("No sessions found.")
        return

    # Get totals
    totals = storage.get_total_cost(start_date=start_date, end_date=end_date)

    print("\n" + "=" * 80)
    print("  KAIROS - Session History")
    print("=" * 80)
    print(f"  {'Session ID':<12} {'Date':<20} {'Cost':<12} {'Duration':<12} {'Cells':<8} {'Eff':<6}")
    print("-" * 80)

    for session in sessions:
        start = session['start_time'][:16] if session['start_time'] else "N/A"
        cost = format_cost(session.get('total_cost_usd', 0))
        duration = format_duration(session.get('total_duration_seconds', 0))
        cells = session.get('cell_count', 0)
        eff = f"{session.get('efficiency_score', 0):.0f}"

        print(f"  {session['session_id']:<12} {start:<20} {cost:<12} {duration:<12} {cells:<8} {eff:<6}")

    print("-" * 80)
    print(f"  Total: {totals['session_count']} sessions | "
          f"{format_cost(totals['total_cost_usd'])} | "
          f"{format_duration(totals['total_duration_seconds'])} | "
          f"{totals['total_cells']} cells")
    print("=" * 80 + "\n")


def cmd_export(args):
    """Export session data."""
    from kairos.storage import SessionStorage

    storage = SessionStorage()

    if args.format == "csv":
        output = storage.export_csv(
            output_path=args.output,
            session_ids=args.sessions.split(",") if args.sessions else None,
        )
    else:
        output = storage.export_json(
            output_path=args.output,
            session_ids=args.sessions.split(",") if args.sessions else None,
        )

    print(f"Exported to: {output}")


def cmd_config(args):
    """View or set configuration."""
    from kairos.config import KairosConfig, DEFAULT_GPU_PRICING

    if args.show_pricing:
        print("\n" + "=" * 60)
        print("  KAIROS - GPU Pricing (USD/hour)")
        print("=" * 60)

        for provider, instances in DEFAULT_GPU_PRICING.items():
            print(f"\n  {provider.upper()}:")
            for instance, price in sorted(instances.items(), key=lambda x: x[1], reverse=True):
                print(f"    {instance:<25} ${price:.2f}/hr")

        print("=" * 60 + "\n")
        return

    if args.set:
        # Parse key=value
        try:
            key, value = args.set.split("=", 1)
            config_path = Path(".kairos") / "config.json"
            config_path.parent.mkdir(exist_ok=True)

            # Load existing config
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
            else:
                config_data = {}

            # Update
            config_data[key] = value
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            print(f"Set {key} = {value}")
        except ValueError:
            print("Invalid format. Use: kairos config --set key=value")
        return

    # Show current config
    config = KairosConfig.from_env()
    print("\n" + "=" * 60)
    print("  KAIROS - Current Configuration")
    print("=" * 60)
    print(f"  Cloud Provider:   {config.cloud_provider.value}")
    print(f"  Instance Type:    {config.instance_type or 'auto-detect'}")
    print(f"  Hourly Rate:      ${config.get_hourly_rate():.2f}")
    print(f"  Alert Threshold:  {f'${config.alert_threshold_usd:.2f}' if config.alert_threshold_usd else 'Not set'}")
    print(f"  Auto-detect GPU:  {config.auto_detect_gpu}")
    print("=" * 60)
    print("\nEnvironment Variables:")
    print("  KAIROS_CLOUD_PROVIDER  - aws, gcp, azure, or local")
    print("  KAIROS_INSTANCE_TYPE   - Instance type string")
    print("  KAIROS_HOURLY_RATE     - Custom hourly rate")
    print("  KAIROS_ALERT_THRESHOLD - Cost alert threshold")
    print()


def cmd_cleanup(args):
    """Clean up old session data."""
    from kairos.storage import SessionStorage

    storage = SessionStorage()

    if args.all:
        confirm = input("Delete ALL session data? This cannot be undone. [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

        # Delete everything
        import shutil
        storage_path = Path(".kairos")
        if storage_path.exists():
            shutil.rmtree(storage_path)
            print("All session data deleted.")
        else:
            print("No session data found.")
        return

    deleted = storage.cleanup_old_sessions(days=args.days)
    print(f"Cleaned up {deleted} sessions older than {args.days} days.")


def cmd_summary(args):
    """Show aggregate summary across all sessions."""
    from kairos.storage import SessionStorage

    storage = SessionStorage()

    # Parse date filters
    start_date = None
    if args.since:
        try:
            if args.since.endswith("d"):
                days = int(args.since[:-1])
                start_date = datetime.now() - timedelta(days=days)
            elif args.since.endswith("w"):
                weeks = int(args.since[:-1])
                start_date = datetime.now() - timedelta(weeks=weeks)
            elif args.since.endswith("m"):
                months = int(args.since[:-1])
                start_date = datetime.now() - timedelta(days=months * 30)
        except ValueError:
            pass

    totals = storage.get_total_cost(start_date=start_date)

    period = f"last {args.since}" if args.since else "all time"

    print("\n" + "=" * 60)
    print(f"  KAIROS - Summary ({period})")
    print("=" * 60)
    print(f"  Total Sessions:    {totals['session_count']}")
    print(f"  Total Cost:        {format_cost(totals['total_cost_usd'])}")
    print(f"  Total Runtime:     {format_duration(totals['total_duration_seconds'])}")
    print(f"  Total Cells:       {totals['total_cells']}")
    print(f"  Avg Efficiency:    {totals['avg_efficiency_score']:.1f}/100")

    # Calculate rates
    if totals['total_duration_seconds'] > 0:
        hours = totals['total_duration_seconds'] / 3600
        effective_rate = totals['total_cost_usd'] / hours if hours > 0 else 0
        print("-" * 60)
        print(f"  Effective Rate:    {format_cost(effective_rate)}/hour")

        if totals['session_count'] > 0:
            avg_session_cost = totals['total_cost_usd'] / totals['session_count']
            print(f"  Avg Session Cost:  {format_cost(avg_session_cost)}")

    print("=" * 60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kairos",
        description="Kairos - AI/ML Cost Intelligence Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kairos status              View most recent session
  kairos history             List all sessions
  kairos history --since 7d  Sessions from last 7 days
  kairos export --format csv Export to CSV
  kairos summary --since 30d Monthly summary
  kairos config --pricing    Show GPU pricing
  kairos cleanup --days 30   Delete old sessions

For more info: https://usekairos.ai
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="View current/recent session status")
    status_parser.set_defaults(func=cmd_status)

    # History command
    history_parser = subparsers.add_parser("history", help="List session history")
    history_parser.add_argument("--limit", "-n", type=int, default=20, help="Number of sessions to show")
    history_parser.add_argument("--since", "-s", help="Filter by date (e.g., 7d, 2w, 2024-01-01)")
    history_parser.set_defaults(func=cmd_history)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export session data")
    export_parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv", help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--sessions", help="Comma-separated session IDs to export")
    export_parser.set_defaults(func=cmd_export)

    # Config command
    config_parser = subparsers.add_parser("config", help="View or set configuration")
    config_parser.add_argument("--set", help="Set a config value (key=value)")
    config_parser.add_argument("--pricing", "--show-pricing", dest="show_pricing", action="store_true", help="Show GPU pricing")
    config_parser.set_defaults(func=cmd_config)

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show aggregate summary")
    summary_parser.add_argument("--since", "-s", help="Time period (e.g., 7d, 30d, 3m)")
    summary_parser.set_defaults(func=cmd_summary)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old session data")
    cleanup_parser.add_argument("--days", "-d", type=int, default=30, help="Delete sessions older than N days")
    cleanup_parser.add_argument("--all", action="store_true", help="Delete ALL session data")
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Run command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
