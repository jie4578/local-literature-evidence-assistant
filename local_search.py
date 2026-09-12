"""本地 SQLite FTS5 全文索引，不联网。"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Iterable


class LocalSearchIndex:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.fts5_available = True
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(source_file UNINDEXED, page_number UNINDEXED, text)")
        except sqlite3.OperationalError:
            self.fts5_available = False
            self.conn.execute("CREATE TABLE IF NOT EXISTS pages_fallback (source_file TEXT, page_number INTEGER, text TEXT, content_hash TEXT, UNIQUE(source_file, page_number))")
        self.conn.commit()

    def index_pages(self, pages: Iterable[Any]) -> int:
        page_list = list(pages)
        if not page_list:
            return 0
        source = page_list[0].source_file if hasattr(page_list[0], "source_file") else page_list[0]["source_file"]
        texts = [page.text if hasattr(page, "text") else page["text"] for page in page_list]
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        existing = self.conn.execute("SELECT content_hash FROM indexed_files WHERE source_file = ?", (source,)).fetchone() if self._has_indexed_files() else None
        self._ensure_indexed_files()
        existing = self.conn.execute("SELECT content_hash FROM indexed_files WHERE source_file = ?", (source,)).fetchone()
        if existing and existing[0] == digest:
            return 0
        self.remove_source(source)
        for page in page_list:
            number = page.page_number if hasattr(page, "page_number") else page["page_number"]
            text = page.text if hasattr(page, "text") else page["text"]
            if self.fts5_available:
                self.conn.execute("INSERT INTO pages_fts(source_file,page_number,text) VALUES(?,?,?)", (source, number, text))
            else:
                self.conn.execute("INSERT OR REPLACE INTO pages_fallback VALUES(?,?,?,?)", (source, number, text, hashlib.sha256(text.encode()).hexdigest()))
        self.conn.execute("INSERT OR REPLACE INTO indexed_files VALUES(?,?)", (source, digest))
        self.conn.commit()
        return len(page_list)

    def _has_indexed_files(self) -> bool:
        return bool(self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_files'").fetchone())

    def _ensure_indexed_files(self) -> None:
        self.conn.execute("CREATE TABLE IF NOT EXISTS indexed_files(source_file TEXT PRIMARY KEY, content_hash TEXT NOT NULL)")

    def remove_source(self, source_file: str) -> None:
        if self.fts5_available:
            self.conn.execute("DELETE FROM pages_fts WHERE source_file = ?", (source_file,))
        else:
            self.conn.execute("DELETE FROM pages_fallback WHERE source_file = ?", (source_file,))
        self._ensure_indexed_files()
        self.conn.execute("DELETE FROM indexed_files WHERE source_file = ?", (source_file,))

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        special_pattern = self._statistical_pattern(query)
        if special_pattern is not None:
            return self._search_pattern(special_pattern, limit)
        if self.fts5_available:
            if query.startswith('"') and query.endswith('"'):
                match = query
            else:
                tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
                match = " AND ".join(tokens)
            if not match:
                return []
            try:
                rows = self.conn.execute("SELECT source_file,page_number,text FROM pages_fts WHERE pages_fts MATCH ? LIMIT ?", (match, limit)).fetchall()
            except sqlite3.OperationalError:
                return []
        else:
            terms = [term for term in query.strip('"').split() if term]
            if not terms:
                return []
            condition = " AND ".join("text LIKE ?" for _ in terms)
            try:
                rows = self.conn.execute(f"SELECT source_file,page_number,text FROM pages_fallback WHERE {condition} LIMIT ?", tuple(f"%{term}%" for term in terms) + (limit,)).fetchall()
            except sqlite3.OperationalError:
                return []
        return [{"source_file": row[0], "page_number": row[1], "snippet": row[2]} for row in rows]

    @staticmethod
    def _statistical_pattern(query: str) -> re.Pattern[str] | None:
        normalized = unicodedata.normalize("NFKC", query).strip()
        if re.search(r"(?i)\bp\s*(?:[-\u2010-\u2015\u2212]?\s*value|值)", normalized):
            return re.compile(r"(?i)\bp\s*(?:(?:[-\u2010-\u2015\u2212]?\s*value)|值|[<>=≤≥]\s*0?\.\d+)")
        match = re.search(r"(?i)\bp\s*([<>=≤≥])\s*(0?\.\d+)", normalized)
        if match:
            operator, number = re.escape(match.group(1)), re.escape(match.group(2))
            return re.compile(rf"(?i)\bp\s*{operator}\s*{number}")
        return None

    def _search_pattern(self, pattern: re.Pattern[str], limit: int) -> list[dict[str, Any]]:
        table = "pages_fts" if self.fts5_available else "pages_fallback"
        try:
            rows = self.conn.execute(f"SELECT source_file,page_number,text FROM {table}").fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"source_file": row[0], "page_number": row[1], "snippet": row[2]} for row in rows if pattern.search(row[2])][:limit]

    def clear(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError("清空本地索引前必须明确 confirm=True")
        self.conn.execute("DROP TABLE IF EXISTS indexed_files")
        self.conn.execute("DROP TABLE IF EXISTS pages_fallback")
        self.conn.execute("DROP TABLE IF EXISTS pages_fts")
        self.conn.commit()
        self.conn.close()
