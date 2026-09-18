"""可选的本地 SentenceTransformer embedding 后端。

本模块故意不在顶层导入 sentence-transformers，保证 Local Offline lexical
模式在未安装可选依赖时仍可启动。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from local_search import LocalSearchIndex, SearchContext


MAX_SEMANTIC_PASSAGES = 10_000


class OptionalSemanticDependencyError(ImportError):
    """可选 semantic 依赖未安装。"""


class LocalModelPathError(ValueError):
    """模型路径不是已存在的本地目录。"""


class EmbeddingDataError(ValueError):
    """embedding 形状、数值或维度无效。"""


class EmbeddingEncoder(Protocol):
    dimension: int
    model_fingerprint: str
    batch_size: int

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray: ...

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray: ...


def _validate_matrix(
    values: Any,
    *,
    dimension: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise EmbeddingDataError("embedding 必须是非空二维向量矩阵")
    if dimension is not None and array.shape[1] != int(dimension):
        raise EmbeddingDataError(f"embedding 维度不一致：收到 {array.shape[1]}，预期 {dimension}")
    if not np.isfinite(array).all():
        raise EmbeddingDataError("embedding 包含 NaN 或 inf")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise EmbeddingDataError("embedding 包含零向量")
    if normalize:
        array = array / norms[:, None]
    return np.ascontiguousarray(array, dtype=np.float32)


def _model_fingerprint(model_dir: Path, dimension: int, backend_version: str) -> str:
    """仅哈希模型目录元数据和小型配置，不保存或哈希绝对路径/完整权重。"""
    entries: list[dict[str, Any]] = []
    config_hashes: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        stat = path.stat()
        entries.append({"name": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if path.name in {"config.json", "modules.json", "sentence_bert_config.json"} and stat.st_size <= 1_048_576:
            config_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "backend": "sentence-transformers",
        "backend_version": backend_version,
        "dimension": int(dimension),
        "entries": entries,
        "config_hashes": config_hashes,
    }
    return "st_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]


class LocalSentenceTransformerEncoder:
    """只接受本地模型目录的 SentenceTransformer encoder。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        normalize_embeddings: bool = True,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise LocalModelPathError(
                f"模型路径必须是已存在的本地目录：{model_path}；不会根据模型 ID 访问网络。"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalSemanticDependencyError(
                "未安装可选 semantic 依赖。请在需要时使用 requirements-semantic.txt 安装，Local Offline 不受影响。"
            ) from exc
        self.batch_size = max(1, int(batch_size))
        self.query_prefix = query_prefix or ""
        self.normalize_embeddings = bool(normalize_embeddings)
        self._model = SentenceTransformer(
            str(self.model_path),
            device=device,
            local_files_only=True,
            trust_remote_code=False,
        )
        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise EmbeddingDataError("本地模型未提供有效 embedding 维度")
        self.dimension = int(dimension)
        version = getattr(__import__("sentence_transformers"), "__version__", "unknown")
        self.model_fingerprint = _model_fingerprint(self.model_path, self.dimension, str(version))

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        result = self._model.encode(
            values,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return _validate_matrix(result, dimension=self.dimension, normalize=self.normalize_embeddings)

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode([f"{self.query_prefix}{text}" for text in texts])

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        return self._encode(texts)


def _context_db_path(search_context: SearchContext) -> Path:
    task_dir = Path(search_context.task_dir).resolve()
    db_path = Path(search_context.db_path).resolve()
    if (
        not task_dir.is_dir()
        or db_path.name != "local_index.sqlite"
        or db_path.parent != task_dir
        or not db_path.is_file()
    ):
        raise FileNotFoundError("当前批次索引不存在或任务目录无效，无法使用本地 semantic backend。")
    return db_path


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_cached(record: dict[str, Any] | None, *, text_hash: str, dimension: int) -> np.ndarray | None:
    if not record or record.get("text_hash") != text_hash or int(record.get("dimension", 0)) != int(dimension):
        return None
    blob = record.get("vector_blob", b"")
    if len(blob) != int(dimension) * 4:
        return None
    try:
        return _validate_matrix(np.frombuffer(blob, dtype=np.float32), dimension=dimension, normalize=True)[0]
    except (EmbeddingDataError, ValueError, TypeError):
        return None


def build_semantic_index(
    search_context: SearchContext | dict[str, Any],
    encoder: EmbeddingEncoder,
    source_files: Iterable[str] | None = None,
    sections: Iterable[str] | None = None,
    *,
    exclude_references: bool = True,
    max_passages: int = MAX_SEMANTIC_PASSAGES,
) -> dict[str, Any]:
    """为当前批次构建或增量更新 embedding cache。"""
    context = SearchContext.from_value(search_context)
    if context is None:
        raise ValueError("缺少有效的当前批次 SearchContext")
    db_path = _context_db_path(context)
    dimension = int(encoder.dimension)
    fingerprint = str(encoder.model_fingerprint)
    with LocalSearchIndex(db_path) as index:
        passages = index.list_passages(
            source_files=source_files if source_files is not None else context.document_sources,
            sections=sections,
            exclude_references=exclude_references,
        )
        if len(passages) > max_passages:
            raise ValueError(f"当前批次 passage 数量 {len(passages)} 超过 semantic 上限 {max_passages}，未加载向量。")
        missing: list[dict[str, Any]] = []
        cached = 0
        for passage in passages:
            text_hash = _text_hash(passage["text"])
            record = index.get_embedding(passage["passage_id"], fingerprint)
            if _decode_cached(record, text_hash=text_hash, dimension=dimension) is not None:
                cached += 1
            else:
                missing.append(passage)
        encoded = 0
        failed = 0
        warnings: list[str] = []
        batch_size = max(1, int(getattr(encoder, "batch_size", 32)))
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            try:
                vectors = _validate_matrix(
                    encoder.encode_documents([item["text"] for item in batch]),
                    dimension=dimension,
                    normalize=True,
                )
                if len(vectors) != len(batch):
                    raise EmbeddingDataError("encoder 返回数量与输入 passage 数量不一致")
            except Exception as exc:
                failed += len(batch)
                warnings.append(f"embedding 批次失败：{type(exc).__name__}")
                continue
            batch_encoded = 0
            try:
                with index.conn:
                    for passage, vector in zip(batch, vectors):
                        index.upsert_embedding(
                            passage["passage_id"],
                            fingerprint,
                            _text_hash(passage["text"]),
                            dimension,
                            vector.astype(np.float32, copy=False).tobytes(),
                        )
                        encoded += 1
                        batch_encoded += 1
            except Exception as exc:
                failed += len(batch)
                encoded -= batch_encoded
                warnings.append(f"embedding cache 写入失败：{type(exc).__name__}")
        return {
            "total_passages": len(passages),
            "cached": cached,
            "encoded": encoded,
            "failed": failed,
            "dimension": dimension,
            "model_fingerprint": fingerprint,
            "warnings": warnings,
        }


class LocalEmbeddingSemanticRetriever:
    """从当前批次 SQLite cache 读取向量并执行确定性 cosine 检索。"""

    def __init__(
        self,
        search_context: SearchContext | dict[str, Any],
        encoder: EmbeddingEncoder,
        *,
        max_passages: int = MAX_SEMANTIC_PASSAGES,
    ):
        context = SearchContext.from_value(search_context)
        if context is None:
            raise ValueError("缺少有效的当前批次 SearchContext")
        self.search_context = context
        self.encoder = encoder
        self.max_passages = max_passages
        self.last_status = "unavailable"
        self.last_warnings: list[str] = []

    def search(
        self,
        query: str,
        limit: int = 10,
        source_files: Iterable[str] | None = None,
        sections: Iterable[str] | None = None,
        exclude_references: bool = True,
    ) -> list[Any]:
        self.last_status = "unavailable"
        self.last_warnings = []
        try:
            db_path = _context_db_path(self.search_context)
        except FileNotFoundError as exc:
            self.last_warnings = [str(exc)]
            return []
        with LocalSearchIndex(db_path) as index:
            passages = index.list_passages(
                source_files=source_files if source_files is not None else self.search_context.document_sources,
                sections=sections,
                exclude_references=exclude_references,
            )
            if len(passages) > self.max_passages:
                self.last_status = "error"
                self.last_warnings = [f"passage 数量超过 semantic 上限 {self.max_passages}。"]
                return []
            candidates = []
            for passage in passages:
                record = index.get_embedding(passage["passage_id"], str(self.encoder.model_fingerprint))
                vector = _decode_cached(record, text_hash=_text_hash(passage["text"]), dimension=int(self.encoder.dimension))
                if vector is None:
                    self.last_warnings.append(f"已跳过无效或过期向量：{passage['passage_id']}")
                    continue
                candidates.append((passage, vector))
            if not candidates:
                self.last_status = "not_indexed"
                if not self.last_warnings:
                    self.last_warnings = ["当前批次尚未建立有效 semantic index。"]
                return []
            try:
                query_vector = _validate_matrix(
                    self.encoder.encode_queries([query]),
                    dimension=int(self.encoder.dimension),
                    normalize=True,
                )[0]
            except Exception as exc:
                self.last_status = "error"
                self.last_warnings.append(f"query embedding 失败：{type(exc).__name__}")
                return []
            scored = []
            for passage, vector in candidates:
                score = float(np.dot(query_vector, vector))
                if np.isfinite(score):
                    scored.append((score, passage))
            scored.sort(key=lambda item: (-item[0], item[1]["passage_id"]))
            from retrieval.models import RetrievalResult

            results = []
            for rank, (score, passage) in enumerate(scored[: max(1, min(int(limit), 200))], 1):
                results.append(
                    RetrievalResult(
                        passage_id=passage["passage_id"],
                        source_file=passage["source_file"],
                        document_id=passage.get("document_id"),
                        section=passage["section"],
                        pdf_page_start=passage["pdf_page_start"],
                        pdf_page_end=passage["pdf_page_end"],
                        text=passage["text"],
                        rank=rank,
                        retrieval_sources=["semantic"],
                        semantic_rank=rank,
                        semantic_score=score,
                    )
                )
            self.last_status = "available"
            return results
