import sqlite3
import threading
import time


class DataStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    ts INTEGER NOT NULL,
                    value REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sensor_samples (
                    ts INTEGER NOT NULL,
                    sensor TEXT NOT NULL,
                    value REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_ts INTEGER NOT NULL,
                    end_ts INTEGER NOT NULL,
                    mean REAL NOT NULL,
                    std REAL NOT NULL,
                    max REAL NOT NULL,
                    min REAL NOT NULL,
                    duration REAL NOT NULL,
                    slope REAL NOT NULL,
                    change_score REAL NOT NULL,
                    candidate INTEGER NOT NULL,
                    label_appliance TEXT,
                    label_phase TEXT,
                    predicted_appliance TEXT,
                    predicted_phase TEXT,
                    created_ts INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS appliances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    status_entity_id TEXT NOT NULL,
                    power_entity_id TEXT NOT NULL,
                    running_watts REAL DEFAULT 0,
                    last_status TEXT,
                    last_status_ts INTEGER,
                    min_power REAL,
                    mean_power REAL,
                    max_power REAL,
                    created_ts INTEGER NOT NULL
                )
                """
            )
            self.conn.commit()
            self._ensure_appliance_columns()

    def _ensure_appliance_columns(self):
        columns = set()
        cursor = self.conn.execute("PRAGMA table_info(appliances)")
        for row in cursor.fetchall():
            columns.add(row[1])
        to_add = []
        if "min_power" not in columns:
            to_add.append(("min_power", "REAL"))
        if "mean_power" not in columns:
            to_add.append(("mean_power", "REAL"))
        if "max_power" not in columns:
            to_add.append(("max_power", "REAL"))
        for col, col_type in to_add:
            try:
                self.conn.execute(f"ALTER TABLE appliances ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass
        if to_add:
            self.conn.commit()

    def add_sample(self, ts, value):
        with self.lock:
            self.conn.execute(
                "INSERT INTO samples (ts, value) VALUES (?, ?)",
                (ts, value),
            )
            self.conn.commit()

    def add_sensor_sample(self, ts, sensor, value):
        with self.lock:
            self.conn.execute(
                "INSERT INTO sensor_samples (ts, sensor, value) VALUES (?, ?, ?)",
                (ts, sensor, value),
            )
            self.conn.commit()

    def get_latest_sample(self):
        with self.lock:
            cursor = self.conn.execute(
                "SELECT ts, value FROM samples ORDER BY ts DESC LIMIT 1"
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def get_latest_sensor_samples(self):
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT sensor, ts, value FROM sensor_samples
                WHERE ts IN (
                    SELECT MAX(ts) FROM sensor_samples GROUP BY sensor
                )
                """
            )
            rows = cursor.fetchall()
        latest = {}
        for row in rows:
            latest[row["sensor"]] = {"ts": row["ts"], "value": row["value"]}
        return latest

    def get_recent_samples(self, limit=200):
        with self.lock:
            cursor = self.conn.execute(
                "SELECT ts, value FROM samples ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows][::-1]

    def get_recent_sensor_samples(self, sensor=None, limit=200):
        query = "SELECT ts, sensor, value FROM sensor_samples"
        params = []
        if sensor:
            query += " WHERE sensor = ?"
            params.append(sensor)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self.lock:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows][::-1]

    def get_samples_between(self, start_ts, end_ts):
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT ts, value FROM samples
                WHERE ts BETWEEN ? AND ?
                ORDER BY ts ASC
                """,
                (start_ts, end_ts),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def add_segment(self, segment):
        with self.lock:
            cursor = self.conn.execute(
                """
                INSERT INTO segments (
                    start_ts, end_ts, mean, std, max, min,
                    duration, slope, change_score, candidate,
                    label_appliance, label_phase,
                    predicted_appliance, predicted_phase, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment["start_ts"],
                    segment["end_ts"],
                    segment["mean"],
                    segment["std"],
                    segment["max"],
                    segment["min"],
                    segment["duration"],
                    segment["slope"],
                    segment["change_score"],
                    1 if segment.get("candidate") else 0,
                    segment.get("label_appliance"),
                    segment.get("label_phase"),
                    segment.get("predicted_appliance"),
                    segment.get("predicted_phase"),
                    segment["created_ts"],
                ),
            )
            self.conn.commit()
            return cursor.lastrowid

    def list_segments(self, limit=200, unlabeled_only=False, candidate_only=False):
        query = "SELECT * FROM segments"
        conditions = []
        params = []
        if unlabeled_only:
            conditions.append("label_appliance IS NULL")
        if candidate_only:
            conditions.append("candidate = 1")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY start_ts DESC LIMIT ?"
        params.append(limit)
        with self.lock:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def delete_unlabeled_before(self, cutoff_ts):
        with self.lock:
            cursor = self.conn.execute(
                """
                DELETE FROM segments
                WHERE label_appliance IS NULL
                  AND label_phase IS NULL
                  AND end_ts < ?
                """,
                (cutoff_ts,),
            )
            self.conn.commit()
        return cursor.rowcount

    def get_segment(self, segment_id):
        with self.lock:
            cursor = self.conn.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def update_segment_label(self, segment_id, appliance, phase):
        with self.lock:
            self.conn.execute(
                """
                UPDATE segments
                SET label_appliance = ?, label_phase = ?
                WHERE id = ?
                """,
                (appliance, phase, segment_id),
            )
            self.conn.commit()

    def update_segment_prediction(self, segment_id, appliance, phase):
        with self.lock:
            self.conn.execute(
                """
                UPDATE segments
                SET predicted_appliance = ?, predicted_phase = ?
                WHERE id = ?
                """,
                (appliance, phase, segment_id),
            )
            self.conn.commit()

    def list_appliances(self):
        with self.lock:
            cursor = self.conn.execute(
                "SELECT * FROM appliances ORDER BY name ASC"
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def add_appliance(self, name, status_entity_id, power_entity_id):
        created_ts = int(time.time())
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO appliances (
                    name, status_entity_id, power_entity_id, created_ts
                ) VALUES (?, ?, ?, ?)
                """,
                (name, status_entity_id, power_entity_id, created_ts),
            )
            self.conn.commit()

    def get_appliance(self, name):
        with self.lock:
            cursor = self.conn.execute(
                "SELECT * FROM appliances WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def update_appliance_running_watts(self, name, running_watts):
        with self.lock:
            self.conn.execute(
                """
                UPDATE appliances
                SET running_watts = ?
                WHERE name = ?
                """,
                (running_watts, name),
            )
            self.conn.commit()

    def update_appliance_power_stats(self, name, min_power, mean_power, max_power):
        with self.lock:
            self.conn.execute(
                """
                UPDATE appliances
                SET min_power = ?, mean_power = ?, max_power = ?
                WHERE name = ?
                """,
                (min_power, mean_power, max_power, name),
            )
            self.conn.commit()

    def update_appliance_status(self, name, status, ts):
        with self.lock:
            self.conn.execute(
                """
                UPDATE appliances
                SET last_status = ?, last_status_ts = ?
                WHERE name = ?
                """,
                (status, ts, name),
            )
            self.conn.commit()

    def get_labeled_segments(self):
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT * FROM segments
                WHERE label_appliance IS NOT NULL AND label_phase IS NOT NULL
                ORDER BY start_ts DESC
                """
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_label_counts_by_appliance(self):
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT label_appliance AS appliance, COUNT(*) AS count
                FROM segments
                WHERE label_appliance IS NOT NULL
                GROUP BY label_appliance
                """
            )
            rows = cursor.fetchall()
        return {row["appliance"]: row["count"] for row in rows}

    def clear_segment_prediction(self, segment_id):
        with self.lock:
            self.conn.execute(
                """
                UPDATE segments
                SET predicted_appliance = NULL, predicted_phase = NULL
                WHERE id = ?
                """,
                (segment_id,),
            )
            self.conn.commit()
