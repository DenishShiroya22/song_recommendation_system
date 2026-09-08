"""Anonymous recommendation impressions and votes; SQLite locally, PostgreSQL in production."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import json
import os
import uuid

ROOT = Path(__file__).resolve().parent

class FeedbackStore:
    def __init__(self, database_url="", sqlite_path=None):
        self.database_url = database_url
        self.sqlite_path = Path(sqlite_path or ROOT/"data/feedback.sqlite3")
        self.durable = bool(database_url)
        if database_url and not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("Feedback database must use a PostgreSQL connection URL.")
        if not database_url:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            self.execute(conn, """CREATE TABLE IF NOT EXISTS recommendation_impressions (
                request_id TEXT NOT NULL, track_id TEXT NOT NULL, seed_id TEXT NOT NULL,
                session_id TEXT NOT NULL, rank INTEGER NOT NULL, audio_weight REAL NOT NULL,
                audio_score REAL NOT NULL, genre_score REAL NOT NULL, score REAL NOT NULL,
                ranking_version TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(request_id, track_id))""")
            self.execute(conn, """CREATE TABLE IF NOT EXISTS recommendation_votes (
                request_id TEXT NOT NULL, track_id TEXT NOT NULL, vote INTEGER NOT NULL CHECK(vote IN (0,1)),
                updated_at TEXT NOT NULL, PRIMARY KEY(request_id, track_id),
                FOREIGN KEY(request_id, track_id) REFERENCES recommendation_impressions(request_id, track_id))""")

    @contextmanager
    def connection(self):
        if self.database_url:
            import psycopg
            conn = psycopg.connect(self.database_url, connect_timeout=8)
        else:
            conn = sqlite3.connect(self.sqlite_path, timeout=10)
            conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def execute(self, conn, sql, params=()):
        return conn.execute(sql.replace("?", "%s") if self.database_url else sql, params)

    def record_impressions(self, session_id, seed_id, rows, audio_weight, request_id=None):
        request_id = request_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            for rank, row in enumerate(rows, 1):
                self.execute(conn, """INSERT INTO recommendation_impressions
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id, track_id) DO NOTHING""",
                    (request_id, row["track_id"], seed_id, session_id, rank, float(audio_weight),
                     float(row["audio_similarity"]), float(row["genre_similarity"]),
                     float(row["similarity"]), row["ranking_version"], now))
        return request_id

    def vote(self, session_id, request_id, track_id, vote):
        if vote is not None and (type(vote) is not int or vote not in (0, 1)):
            raise ValueError("Vote must be 0, 1, or None.")
        with self.connection() as conn:
            owner = self.execute(conn, """SELECT session_id FROM recommendation_impressions
                WHERE request_id=? AND track_id=?""", (request_id, track_id)).fetchone()
            if owner is None or owner[0] != session_id:
                raise ValueError("Vote does not belong to this browser session.")
            if vote is None:
                self.execute(conn, "DELETE FROM recommendation_votes WHERE request_id=? AND track_id=?",
                             (request_id, track_id))
            else:
                self.execute(conn, """INSERT INTO recommendation_votes VALUES (?, ?, ?, ?)
                    ON CONFLICT(request_id,track_id) DO UPDATE SET vote=excluded.vote, updated_at=excluded.updated_at""",
                    (request_id, track_id, vote, datetime.now(timezone.utc).isoformat()))

    def summary(self):
        # No public endpoint: this aggregate report is for the project owner via CLI.
        with self.connection() as conn:
            rows = self.execute(conn, """SELECT i.ranking_version, i.audio_weight, COUNT(*),
                COUNT(v.vote), COALESCE(SUM(v.vote),0), COUNT(DISTINCT i.session_id)
                FROM recommendation_impressions i LEFT JOIN recommendation_votes v
                ON i.request_id=v.request_id AND i.track_id=v.track_id
                GROUP BY i.ranking_version,i.audio_weight ORDER BY i.audio_weight""").fetchall()
        return [{"ranking_version":version, "audio_weight":weight, "impressions":shown, "ratings":rated,
                 "likes":likes, "dislikes":rated-likes, "sessions":sessions,
                 "rating_coverage":rated/shown if shown else 0,
                 "like_rate_among_rated":likes/rated if rated else None}
                for version,weight,shown,rated,likes,sessions in rows]

if __name__ == "__main__":
    database_url = os.environ.get("FEEDBACK_DATABASE_URL", "")
    secrets_path = ROOT/".streamlit/secrets.toml"
    if not database_url and secrets_path.exists():
        import tomllib
        database_url = tomllib.loads(secrets_path.read_text(encoding="utf-8-sig")).get("feedback_database_url", "")
    print(json.dumps(FeedbackStore(database_url).summary(), indent=2))

