"""
Persistent storage for Kairos sessions.

Provides local file-based storage for session data, enabling:
- Session persistence across notebook restarts
- Historical session analysis
- Data export for reporting

Storage formats:
- JSON: Human-readable, easy to inspect
- SQLite: Efficient for large datasets and queries
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
from dataclasses import asdict
import threading

from kairos.cost_calculator import CellExecution, SessionSummary

logger = logging.getLogger(__name__)


class SessionStorage:
    """
    File-based storage for Kairos session data.

    Stores session data in JSON format for easy inspection and SQLite
    for efficient querying of historical data.
    """

    DEFAULT_PATH = ".kairos"

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize storage.

        Args:
            base_path: Base directory for storage (default: .kairos/)
        """
        self.base_path = Path(base_path or self.DEFAULT_PATH)
        self._ensure_directories()
        self._db_lock = threading.Lock()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "sessions").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)

    @property
    def db_path(self) -> Path:
        """Path to SQLite database."""
        return self.base_path / "kairos.db"

    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        cloud_provider TEXT,
                        instance_type TEXT,
                        hourly_rate REAL,
                        total_cost REAL DEFAULT 0,
                        total_duration REAL DEFAULT 0,
                        cell_count INTEGER DEFAULT 0,
                        avg_gpu_utilization REAL DEFAULT 0,
                        efficiency_score REAL DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Cells table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cells (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        cell_id TEXT NOT NULL,
                        cell_number INTEGER,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        duration REAL,
                        cost REAL,
                        gpu_utilization_avg REAL,
                        gpu_utilization_max REAL,
                        memory_used_avg INTEGER,
                        memory_used_max INTEGER,
                        tags TEXT,
                        error TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    )
                """)

                # Indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cells_session
                    ON cells(session_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_start
                    ON sessions(start_time)
                """)

                conn.commit()
            finally:
                conn.close()

    def save_session(self, tracker: "KairosTracker") -> str:
        """
        Save a complete session.

        Args:
            tracker: KairosTracker instance

        Returns:
            Session ID
        """
        self._init_db()

        session_id = tracker.session_id
        summary = tracker._calculator.get_summary()
        config = tracker.config

        # Save to JSON file
        session_data = {
            "session_id": session_id,
            "start_time": summary.start_time.isoformat(),
            "end_time": summary.end_time.isoformat() if summary.end_time else None,
            "config": config.to_dict(),
            "summary": summary.to_dict(),
            "cells": tracker.get_cell_costs(),
            "saved_at": datetime.now().isoformat(),
        }

        json_path = self.base_path / "sessions" / f"{session_id}.json"
        with open(json_path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        # Save to SQLite
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Upsert session
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (
                        session_id, start_time, end_time, cloud_provider,
                        instance_type, hourly_rate, total_cost, total_duration,
                        cell_count, avg_gpu_utilization, efficiency_score,
                        metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    summary.start_time.isoformat(),
                    summary.end_time.isoformat() if summary.end_time else None,
                    summary.cloud_provider,
                    summary.instance_type,
                    summary.hourly_rate,
                    summary.total_cost_usd,
                    summary.total_duration_seconds,
                    summary.cell_count,
                    summary.avg_gpu_utilization,
                    summary.efficiency_score,
                    json.dumps(config.to_dict()),
                    datetime.now().isoformat(),
                ))

                # Save cells
                for cell_data in tracker.get_cell_costs():
                    cursor.execute("""
                        INSERT OR REPLACE INTO cells (
                            session_id, cell_id, cell_number, start_time, end_time,
                            duration, cost, gpu_utilization_avg, gpu_utilization_max,
                            memory_used_avg, memory_used_max, tags, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        cell_data["cell_id"],
                        cell_data.get("cell_number"),
                        cell_data["start_time"],
                        cell_data.get("end_time"),
                        cell_data.get("duration_seconds"),
                        cell_data.get("cost_usd"),
                        cell_data.get("gpu_utilization_avg"),
                        cell_data.get("gpu_utilization_max"),
                        cell_data.get("memory_used_avg_mb"),
                        cell_data.get("memory_used_max_mb"),
                        json.dumps(cell_data.get("tags", [])),
                        cell_data.get("error"),
                    ))

                conn.commit()
            finally:
                conn.close()

        logger.debug(f"Session {session_id} saved to {json_path}")
        return session_id

    def save_cell(self, session_id: str, cell: CellExecution) -> None:
        """
        Save a single cell execution (for incremental saves).

        Args:
            session_id: Session identifier
            cell: Cell execution data
        """
        self._init_db()

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Ensure session exists
                cursor.execute("""
                    INSERT OR IGNORE INTO sessions (session_id, start_time)
                    VALUES (?, ?)
                """, (session_id, datetime.now().isoformat()))

                # Insert cell
                cursor.execute("""
                    INSERT INTO cells (
                        session_id, cell_id, cell_number, start_time, end_time,
                        duration, cost, gpu_utilization_avg, gpu_utilization_max,
                        memory_used_avg, memory_used_max, tags, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    cell.cell_id,
                    cell.cell_number,
                    cell.start_time.isoformat(),
                    cell.end_time.isoformat() if cell.end_time else None,
                    cell.duration_seconds,
                    cell.cost_usd,
                    cell.gpu_utilization_avg,
                    cell.gpu_utilization_max,
                    cell.memory_used_avg_mb,
                    cell.memory_used_max_mb,
                    json.dumps(cell.tags),
                    cell.error,
                ))

                # Update session stats
                cursor.execute("""
                    UPDATE sessions SET
                        cell_count = cell_count + 1,
                        total_cost = total_cost + ?,
                        total_duration = total_duration + ?,
                        updated_at = ?
                    WHERE session_id = ?
                """, (
                    cell.cost_usd,
                    cell.duration_seconds,
                    datetime.now().isoformat(),
                    session_id,
                ))

                conn.commit()
            finally:
                conn.close()

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary or None
        """
        # Try JSON first
        json_path = self.base_path / "sessions" / f"{session_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        # Fall back to SQLite
        self._init_db()
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                session_row = cursor.fetchone()

                if not session_row:
                    return None

                cursor.execute("""
                    SELECT * FROM cells WHERE session_id = ?
                    ORDER BY start_time
                """, (session_id,))
                cell_rows = cursor.fetchall()

                return {
                    "session_id": session_row["session_id"],
                    "start_time": session_row["start_time"],
                    "end_time": session_row["end_time"],
                    "config": json.loads(session_row["metadata"]) if session_row["metadata"] else {},
                    "summary": {
                        "total_cost_usd": session_row["total_cost"],
                        "total_duration_seconds": session_row["total_duration"],
                        "cell_count": session_row["cell_count"],
                        "avg_gpu_utilization": session_row["avg_gpu_utilization"],
                        "efficiency_score": session_row["efficiency_score"],
                        "cloud_provider": session_row["cloud_provider"],
                        "instance_type": session_row["instance_type"],
                        "hourly_rate": session_row["hourly_rate"],
                    },
                    "cells": [
                        {
                            "cell_id": row["cell_id"],
                            "cell_number": row["cell_number"],
                            "start_time": row["start_time"],
                            "end_time": row["end_time"],
                            "duration_seconds": row["duration"],
                            "cost_usd": row["cost"],
                            "gpu_utilization_avg": row["gpu_utilization_avg"],
                            "gpu_utilization_max": row["gpu_utilization_max"],
                            "memory_used_avg_mb": row["memory_used_avg"],
                            "memory_used_max_mb": row["memory_used_max"],
                            "tags": json.loads(row["tags"]) if row["tags"] else [],
                            "error": row["error"],
                        }
                        for row in cell_rows
                    ],
                }
            finally:
                conn.close()

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all sessions with optional filtering.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            start_date: Filter sessions after this date
            end_date: Filter sessions before this date

        Returns:
            List of session summaries
        """
        self._init_db()

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = "SELECT * FROM sessions WHERE 1=1"
                params = []

                if start_date:
                    query += " AND start_time >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND start_time <= ?"
                    params.append(end_date.isoformat())

                query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [
                    {
                        "session_id": row["session_id"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "total_cost_usd": row["total_cost"],
                        "total_duration_seconds": row["total_duration"],
                        "cell_count": row["cell_count"],
                        "cloud_provider": row["cloud_provider"],
                        "instance_type": row["instance_type"],
                        "efficiency_score": row["efficiency_score"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get total costs across all sessions.

        Args:
            start_date: Filter sessions after this date
            end_date: Filter sessions before this date

        Returns:
            Cost summary dictionary
        """
        self._init_db()

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = """
                    SELECT
                        COUNT(*) as session_count,
                        SUM(total_cost) as total_cost,
                        SUM(total_duration) as total_duration,
                        SUM(cell_count) as total_cells,
                        AVG(efficiency_score) as avg_efficiency
                    FROM sessions WHERE 1=1
                """
                params = []

                if start_date:
                    query += " AND start_time >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND start_time <= ?"
                    params.append(end_date.isoformat())

                cursor.execute(query, params)
                row = cursor.fetchone()

                return {
                    "session_count": row["session_count"] or 0,
                    "total_cost_usd": row["total_cost"] or 0,
                    "total_duration_seconds": row["total_duration"] or 0,
                    "total_cells": row["total_cells"] or 0,
                    "avg_efficiency_score": row["avg_efficiency"] or 0,
                }
            finally:
                conn.close()

    def export_csv(
        self,
        output_path: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Export session data to CSV.

        Args:
            output_path: Output file path (default: exports/kairos_export_<timestamp>.csv)
            session_ids: Specific sessions to export (default: all)

        Returns:
            Path to exported file
        """
        import csv

        self._init_db()

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.base_path / "exports" / f"kairos_export_{timestamp}.csv")

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                query = """
                    SELECT
                        s.session_id, s.start_time as session_start,
                        s.cloud_provider, s.instance_type, s.hourly_rate,
                        c.cell_id, c.cell_number, c.start_time as cell_start,
                        c.end_time as cell_end, c.duration, c.cost,
                        c.gpu_utilization_avg, c.memory_used_max, c.tags, c.error
                    FROM sessions s
                    LEFT JOIN cells c ON s.session_id = c.session_id
                """

                if session_ids:
                    placeholders = ",".join("?" * len(session_ids))
                    query += f" WHERE s.session_id IN ({placeholders})"
                    cursor.execute(query, session_ids)
                else:
                    cursor.execute(query)

                rows = cursor.fetchall()

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "session_id", "session_start", "cloud_provider", "instance_type",
                        "hourly_rate", "cell_id", "cell_number", "cell_start", "cell_end",
                        "duration_seconds", "cost_usd", "gpu_utilization", "memory_mb",
                        "tags", "error"
                    ])

                    for row in rows:
                        writer.writerow([
                            row["session_id"], row["session_start"], row["cloud_provider"],
                            row["instance_type"], row["hourly_rate"], row["cell_id"],
                            row["cell_number"], row["cell_start"], row["cell_end"],
                            row["duration"], row["cost"], row["gpu_utilization_avg"],
                            row["memory_used_max"], row["tags"], row["error"]
                        ])

            finally:
                conn.close()

        logger.info(f"Exported to {output_path}")
        return output_path

    def export_json(
        self,
        output_path: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Export session data to JSON.

        Args:
            output_path: Output file path
            session_ids: Specific sessions to export (default: all)

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.base_path / "exports" / f"kairos_export_{timestamp}.json")

        sessions = []

        if session_ids:
            for sid in session_ids:
                data = self.load_session(sid)
                if data:
                    sessions.append(data)
        else:
            for session_info in self.list_sessions(limit=1000):
                data = self.load_session(session_info["session_id"])
                if data:
                    sessions.append(data)

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "session_count": len(sessions),
            "sessions": sessions,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported to {output_path}")
        return output_path

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its data.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        self._init_db()

        # Delete JSON file
        json_path = self.base_path / "sessions" / f"{session_id}.json"
        if json_path.exists():
            json_path.unlink()

        # Delete from SQLite
        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cells WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
            finally:
                conn.close()

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Delete sessions older than this many days

        Returns:
            Number of sessions deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)

        self._init_db()

        deleted_count = 0

        with self._db_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Get old session IDs
                cursor.execute("""
                    SELECT session_id FROM sessions WHERE start_time < ?
                """, (cutoff.isoformat(),))
                old_sessions = [row["session_id"] for row in cursor.fetchall()]

                for session_id in old_sessions:
                    # Delete JSON file
                    json_path = self.base_path / "sessions" / f"{session_id}.json"
                    if json_path.exists():
                        json_path.unlink()

                    # Delete from DB
                    cursor.execute("DELETE FROM cells WHERE session_id = ?", (session_id,))
                    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    deleted_count += 1

                conn.commit()
            finally:
                conn.close()

        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count
