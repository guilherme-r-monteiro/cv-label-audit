import sqlite3
from pathlib import Path

from label_audit.audit import Finding

SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    image TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    label_index INTEGER
);
CREATE INDEX IF NOT EXISTS findings_image ON findings (image);
"""

REVIEW_QUEUE_SQL = """
SELECT image, COUNT(*) AS findings, ROUND(SUM(score), 3) AS priority
FROM findings
GROUP BY image
ORDER BY priority DESC
LIMIT ?
"""

CLASS_BREAKDOWN_SQL = """
SELECT label, kind, COUNT(*) AS findings
FROM findings
GROUP BY label, kind
ORDER BY findings DESC
LIMIT ?
"""


def open_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def save_findings(connection: sqlite3.Connection, image: str, findings: list[Finding]) -> None:
    rows = [(image, f.kind.value, f.label, f.score, *f.box, f.label_index) for f in findings]
    connection.executemany("INSERT INTO findings VALUES (?,?,?,?,?,?,?,?,?)", rows)
    connection.commit()


def review_queue(connection: sqlite3.Connection, limit: int) -> list[tuple[str, int, float]]:
    return connection.execute(REVIEW_QUEUE_SQL, (limit,)).fetchall()


def class_breakdown(connection: sqlite3.Connection, limit: int) -> list[tuple[str, str, int]]:
    return connection.execute(CLASS_BREAKDOWN_SQL, (limit,)).fetchall()
