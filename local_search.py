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
        self.passage_fts5_available = self._ensure_passage_schema()
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

    def _ensure_passage_schema(self) -> bool:
        """懒创建 passage 表；不触碰旧页面表和旧全局索引。"""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS passage_metadata ("
            "passage_id TEXT PRIMARY KEY, source_file TEXT NOT NULL, document_id TEXT, "
            "section TEXT NOT NULL, pdf_page_start INTEGER NOT NULL, pdf_page_end INTEGER NOT NULL, "
            "text TEXT NOT NULL, ordinal INTEGER NOT NULL)"
        )
        if getattr(self, "passage_fts5_available", None) is False:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS passages_fallback (passage_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            return False
        existing_fts = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='passages_fts'"
        ).fetchone()
        if existing_fts:
            return True
        try:
            self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(passage_id UNINDEXED, text)")
            return True
        except sqlite3.OperationalError:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS passages_fallback (passage_id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            return False

    @staticmethod
    def _as_passage_dict(passage: Any) -> dict[str, Any]:
        if hasattr(passage, "to_dict"):
            return dict(passage.to_dict())
        if isinstance(passage, Mapping):
            return dict(passage)
        raise TypeError("passage 必须是映射或提供 to_dict() 的对象")

    def _remove_passages_for_source(self, source_file: str) -> None:
        rows = self.conn.execute(
            "SELECT passage_id FROM passage_metadata WHERE source_file = ?", (source_file,)
        ).fetchall()
        for (passage_id,) in rows:
            if self.passage_fts5_available:
                self.conn.execute("DELETE FROM passages_fts WHERE passage_id = ?", (passage_id,))
            else:
                self.conn.execute("DELETE FROM passages_fallback WHERE passage_id = ?", (passage_id,))
        self.conn.execute("DELETE FROM passage_metadata WHERE source_file = ?", (source_file,))

    def index_passages(self, passages: Iterable[Any]) -> int:
        """写入当前批次的 passage 索引；同来源重建时不产生重复记录。"""
        values = [self._as_passage_dict(item) for item in passages]
        if not values:
            return 0
        self._ensure_passage_schema()
        sources = dict.fromkeys(str(item.get("source_file", "")).strip() for item in values)
        for source in sources:
            if source:
                self._remove_passages_for_source(source)
        inserted = 0
        for item in values:
            required = ("passage_id", "source_file", "section", "text", "pdf_page_start", "pdf_page_end", "ordinal")
            if not all(item.get(field) not in (None, "") for field in required):
                continue
            passage_id = str(item["passage_id"])
            text = str(item["text"])
            if self.passage_fts5_available:
                self.conn.execute("INSERT INTO passages_fts(passage_id,text) VALUES(?,?)", (passage_id, text))
            else:
                self.conn.execute("INSERT OR REPLACE INTO passages_fallback(passage_id,text) VALUES(?,?)", (passage_id, text))
            self.conn.execute(
                "INSERT OR REPLACE INTO passage_metadata "
                "(passage_id,source_file,document_id,section,pdf_page_start,pdf_page_end,text,ordinal) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    passage_id,
                    str(item["source_file"]),
                    item.get("document_id"),
                    str(item["section"]),
                    int(item["pdf_page_start"]),
                    int(item["pdf_page_end"]),
                    text,
                    int(item["ordinal"]),
                ),
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def passage_count(self) -> int:
        self._ensure_passage_schema()
        row = self.conn.execute("SELECT COUNT(*) FROM passage_metadata").fetchone()
        return int(row[0] if row else 0)

    def _ensure_embedding_schema(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS passage_embeddings ("
            "passage_id TEXT NOT NULL, model_fingerprint TEXT NOT NULL, text_hash TEXT NOT NULL, "
            "dimension INTEGER NOT NULL, vector_blob BLOB NOT NULL, "
            "PRIMARY KEY (passage_id, model_fingerprint))"
        )

    def list_passages(
        self,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[dict[str, Any]]:
        """读取当前批次 passage metadata，不读取其他数据库。"""
        self._ensure_passage_schema()
        source_clause, source_params = self._value_filter("m.source_file", source_files)
        normalized_sections = self._normalize_sections(sections)
        section_clause, section_params = self._value_filter("m.section", normalized_sections)
        reference_clause = " AND m.section <> 'References'" if exclude_references else ""
        rows = self.conn.execute(
            "SELECT m.passage_id,m.source_file,m.document_id,m.section,m.pdf_page_start,"
            "m.pdf_page_end,m.text,m.ordinal FROM passage_metadata m WHERE 1 = 1"
            f"{source_clause}{section_clause}{reference_clause} ORDER BY m.passage_id ASC",
            tuple(source_params) + tuple(section_params),
        ).fetchall()
        return [
            {
                "passage_id": row[0],
                "source_file": row[1],
                "document_id": row[2],
                "section": row[3],
                "pdf_page_start": row[4],
                "pdf_page_end": row[5],
                "text": row[6],
                "ordinal": row[7],
            }
            for row in rows
        ]

    def get_embedding(self, passage_id: str, model_fingerprint: str) -> dict[str, Any] | None:
        self._ensure_embedding_schema()
        row = self.conn.execute(
            "SELECT passage_id,model_fingerprint,text_hash,dimension,vector_blob "
            "FROM passage_embeddings WHERE passage_id = ? AND model_fingerprint = ?",
            (passage_id, model_fingerprint),
        ).fetchone()
        if row is None:
            return None
        return {
            "passage_id": row[0],
            "model_fingerprint": row[1],
            "text_hash": row[2],
            "dimension": row[3],
            "vector_blob": bytes(row[4]),
        }

    def upsert_embedding(
        self,
        passage_id: str,
        model_fingerprint: str,
        text_hash: str,
        dimension: int,
        vector_blob: bytes,
    ) -> None:
        self._ensure_embedding_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO passage_embeddings "
            "(passage_id,model_fingerprint,text_hash,dimension,vector_blob) VALUES(?,?,?,?,?)",
            (passage_id, model_fingerprint, text_hash, int(dimension), sqlite3.Binary(vector_blob)),
        )

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
                source.strip()
                for source in source_files
                if isinstance(source, str) and source.strip()
            )
        )
        if not sources:
            return " AND 1 = 0", []
        # 仅插入由程序生成的占位符，来源值始终作为 SQL 参数传入。
        placeholders = ", ".join("?" for _ in sources)
        return f" AND source_file IN ({placeholders})", list(sources)

    @staticmethod
    def _value_filter(column: str, values: Iterable[str] | None) -> tuple[str, list[str]]:
        if values is None:
            return "", []
        normalized = tuple(
            dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip())
        )
        if not normalized:
            return " AND 1 = 0", []
        placeholders = ", ".join("?" for _ in normalized)
        return f" AND {column} IN ({placeholders})", list(normalized)

    @staticmethod
    def _normalize_sections(sections: Iterable[str] | None) -> list[str] | None:
        if sections is None:
            return None
        aliases = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "background": "Introduction",
            "methods": "Methods",
            "method": "Methods",
            "results": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
            "references": "References",
            "unknown": "Unknown",
        }
        return list(
            dict.fromkeys(aliases.get(str(section).strip().casefold(), str(section).strip()) for section in sections)
        )

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

    def search_passages(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[dict[str, Any]]:
        """按确定性词法相关性检索证据段落，不生成或改写原文。"""
        query = unicodedata.normalize("NFKC", (query or "")).strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 200))
        self._ensure_passage_schema()
        normalized_sections = self._normalize_sections(sections)
        source_clause, source_params = self._value_filter("m.source_file", source_files)
        section_clause, section_params = self._value_filter("m.section", normalized_sections)
        reference_clause = " AND m.section <> 'References'" if exclude_references else ""
        filters = f"{source_clause}{section_clause}{reference_clause}"
        special_pattern = self._statistical_pattern(query)
        if special_pattern is not None:
            rows = self.conn.execute(
                "SELECT m.passage_id,m.source_file,m.document_id,m.section,m.pdf_page_start,"
                "m.pdf_page_end,m.text,m.ordinal FROM passage_metadata m WHERE 1 = 1"
                f"{filters}",
                tuple(source_params) + tuple(section_params),
            ).fetchall()
            candidates = [row for row in rows if special_pattern.search(row[6])]
            candidates.sort(key=lambda row: (-row[6].casefold().count(query.casefold()), len(row[6]), row[0]))
            return [self._passage_result(row, rank) for rank, row in enumerate(candidates[:limit], 1)]

        if self.passage_fts5_available:
            if query.startswith('"') and query.endswith('"') and len(query) > 1:
                match = query
            else:
                tokens = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
                # 使用 OR 获取部分命中，再由 BM25/coverage 做确定性排序。
                match = " OR ".join(tokens)
            if not match:
                return []
            rows = self.conn.execute(
                "SELECT m.passage_id,m.source_file,m.document_id,m.section,m.pdf_page_start,"
                "m.pdf_page_end,m.text,m.ordinal,bm25(passages_fts) AS lexical_score "
                "FROM passages_fts JOIN passage_metadata m ON m.passage_id = passages_fts.passage_id "
                "WHERE passages_fts MATCH ?"
                f"{filters} ORDER BY bm25(passages_fts) ASC, m.passage_id ASC LIMIT ?",
                (match, *source_params, *section_params, limit),
            ).fetchall()
            return [self._passage_result(row[:8], rank, lexical_score=row[8]) for rank, row in enumerate(rows, 1)]

        terms = [term for term in query.strip('"').split() if term]
        if not terms:
            return []
        condition = " OR ".join("LOWER(m.text) LIKE LOWER(?)" for _ in terms)
        rows = self.conn.execute(
            "SELECT m.passage_id,m.source_file,m.document_id,m.section,m.pdf_page_start,"
            "m.pdf_page_end,m.text,m.ordinal FROM passage_metadata m WHERE "
            f"({condition}){filters}",
            tuple(f"%{term}%" for term in terms) + tuple(source_params) + tuple(section_params),
        ).fetchall()
        phrase = query.strip('"').casefold()
        def fallback_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
            text = row[6].casefold()
            coverage = sum(term.casefold() in text for term in terms)
            occurrence = sum(text.count(term.casefold()) for term in terms)
            return (-int(phrase in text), -coverage, -occurrence, len(row[6]), row[0])
        rows.sort(key=fallback_key)
        return [self._passage_result(row, rank) for rank, row in enumerate(rows[:limit], 1)]

    @staticmethod
    def _passage_result(row: tuple[Any, ...], rank: int, lexical_score: float | None = None) -> dict[str, Any]:
        return {
            "passage_id": row[0],
            "source_file": row[1],
            "document_id": row[2],
            "section": row[3],
            "pdf_page_start": row[4],
            "pdf_page_end": row[5],
            "text": row[6],
            "ordinal": row[7],
            "rank": rank,
            "lexical_score": lexical_score,
        }

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
        self.conn.execute("DROP TABLE IF EXISTS passage_metadata")
        self.conn.execute("DROP TABLE IF EXISTS passages_fallback")
        self.conn.execute("DROP TABLE IF EXISTS passages_fts")
        self.conn.execute("DROP TABLE IF EXISTS passage_embeddings")
        self.conn.commit()
        self.close()
