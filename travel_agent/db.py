"""SQLite storage. One file holds users, their data, and the searchable travel documents.

shortcut: SQLite and its built-in FTS5 search, because the data is small (18 documents);
move to Postgres with pgvector when there are many users or documents need semantic search.
"""
import json
import sqlite3
from pathlib import Path

DATA = Path(__file__).parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS preferences (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS trips (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  flight_id TEXT NOT NULL, date TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), text TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS pending (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  tool TEXT NOT NULL, args TEXT NOT NULL, summary TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'cancelled')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(title, body);
"""


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    # mode=ro makes SQLite itself refuse writes, so a read-only tool cannot write even if its code tried.
    uri = f"file:{Path(path).resolve().as_posix()}" + ("?mode=ro" if readonly else "")
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(path: str) -> sqlite3.Connection:
    """Create the tables and load the travel documents once. Returns a read-write connection."""
    conn = connect(path)
    conn.executescript(SCHEMA)
    if not conn.execute("SELECT 1 FROM docs LIMIT 1").fetchone():
        docs = json.loads((DATA / "docs.json").read_text(encoding="utf-8"))
        conn.executemany("INSERT INTO docs (title, body) VALUES (:title, :body)", docs)
    conn.commit()
    return conn


def new_user(conn: sqlite3.Connection) -> int:
    user_id = conn.execute("INSERT INTO users DEFAULT VALUES").lastrowid
    conn.commit()
    return user_id
