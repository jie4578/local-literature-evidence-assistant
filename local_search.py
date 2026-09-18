"""本地 SQLite FTS5 全文索引，不联网。"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SearchContext:
    """当前批次的本地搜索边界，不包含论文全文或任何凭据。"""

    batch_id: str
    task_dir: str
    db_path: str
    document_sources: tuple[str, ...] = ()
    document_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "task_dir": self.task_dir,
            "db_path": self.db_path,
            "document_sources": list(self.document_sources),
            "document_count": self.document_count,
        }

    @classmethod
    def from_value(cls, value: Any) -> "SearchContext | None":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        batch_id = value.get("batch_id")
        task_dir = value.get("task_dir")
        db_path = value.get("db_path")
        sources = value.get("document_sources", ())
        if not all(isinstance(item, str) and item.strip() for item in (batch_id, task_dir, db_path)):
            return None
        if not isinstance(sources, (list, tuple)):
            return None
        normalized_sources = tuple(
            dict.fromkeys(item.strip() for item in sources if isinstance(item, str) and item.strip())
        )
        count = value.get("document_count", len(normalized_sources))
        if not isinstance(count, int) or count < 0:
            count = len(normalized_sources)
        return cls(
            batch_id=batch_id.strip(),
            task_dir=task_dir.strip(),
            db_path=db_path.strip(),
            document_sources=normalized_sources,
            document_count=count,
        )


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

    def close(self) -> None:
        """关闭 SQLite 连接；可重复调用。"""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self) -> "LocalSearchIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

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

    @staticmethod
    def _source_filter(source_files: Iterable[str] | None) -> tuple[str, list[str]]:
        if source_files is None:
            return "", []
        sources = tuple(
            dict.fromkeys(
                str(source).strip()
                for source in source_files
                if str(source).strip()
            )
        )
        if not sources:
            return " AND 1 = 0", []
        # 仅插入由程序生成的占位符，来源值始终作为 SQL 参数传入。
        placeholders = ", ".join("?" for _ in sources)
        return f" AND source_file IN ({placeholders})", list(sources)

    def search(
        self,
        query: str,
        limit: int = 20,
        source_files: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        special_pattern = self._statistical_pattern(query)
        if special_pattern is not None:
            return self._search_pattern(special_pattern, limit, source_files=source_files)
        if self.fts5_available:
            if query.startswith('"') and query.endswith('"'):
                match = query
            else:
                tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
                match = " AND ".join(tokens)
            if not match:
                return []
            source_clause, source_params = self._source_filter(source_files)
            try:
                rows = self.conn.execute(
                    "SELECT source_file,page_number,text FROM pages_fts "
                    f"WHERE pages_fts MATCH ?{source_clause} LIMIT ?",
                    (match, *source_params, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        else:
            terms = [term for term in query.strip('"').split() if term]
            if not terms:
                return []
            condition = " AND ".join("text LIKE ?" for _ in terms)
            source_clause, source_params = self._source_filter(source_files)
            try:
                rows = self.conn.execute(
                    f"SELECT source_file,page_number,text FROM pages_fallback "
                    f"WHERE {condition}{source_clause} LIMIT ?",
                    tuple(f"%{term}%" for term in terms) + tuple(source_params) + (limit,),
                ).fetchall()
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

    def _search_pattern(
        self,
        pattern: re.Pattern[str],
        limit: int,
        source_files: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        table = "pages_fts" if self.fts5_available else "pages_fallback"
        source_clause, source_params = self._source_filter(source_files)
        try:
            rows = self.conn.execute(
                f"SELECT source_file,page_number,text FROM {table} "
                f"WHERE 1 = 1{source_clause}",
                tuple(source_params),
            ).fetchall()
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
        self.close()
