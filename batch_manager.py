"""Local Offline 批次缓存、身份、暂停和续跑。

批次状态采用两层身份模型：``inputs`` 保留每个上传输入，``documents``
按完整内容 SHA-256 聚合。basename 只用于展示，绝不参与去重或缓存键。
这里没有 Provider 依赖，也不会访问网络。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from local_extractor import extract_local_paper
from storage import atomic_write_json


ProgressCallback = Callable[[float, str], Any]
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_COMPATIBILITY_ERROR = "批次 manifest 版本过旧或结构不兼容，无法安全迁移，请重新开始批次。"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_stem(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^\w\-一-龥]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem or "paper")[:80]


def _fingerprint(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _source_alias(path: Path, occurrence: int = 1) -> str:
    """生成不含绝对路径的稳定展示别名。"""
    parent = path.parent.name
    base = f"{parent}/{path.name}" if parent else path.name
    base = re.sub(r"[\\/]+", "/", base)
    return base if occurrence == 1 else f"{base} #{occurrence}"


def _unreadable_document_id(input_id: str) -> str:
    return f"doc_unreadable_{input_id}"


def _document_id(content_sha256: str | None, input_id: str) -> str:
    return f"doc_{content_sha256}" if content_sha256 else _unreadable_document_id(input_id)


def _new_task_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        task_dir = root / f"local_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
        try:
            task_dir.mkdir()
            return task_dir
        except FileExistsError:
            continue


def _manifest_path(task_dir: Path) -> Path:
    return task_dir / "batch_manifest.json"


def _status_path(task_dir: Path) -> Path:
    return task_dir / "status.json"


def _compatibility_check(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("inputs"), list)
        or not isinstance(manifest.get("documents"), list)
    ):
        raise ValueError(MANIFEST_COMPATIBILITY_ERROR)


def _legacy_papers(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """提供旧调用方只读兼容视图，不再作为身份或状态来源。"""
    return [
        {
            "paper_id": document.get("document_id"),
            "source_file": document.get("display_name"),
            "fingerprint": document.get("content_sha256"),
            "status": document.get("status"),
            "result_file": document.get("result_file"),
            "errors": document.get("errors", []),
        }
        for document in manifest.get("documents", [])
    ]


def _refresh_counts(manifest: dict[str, Any]) -> None:
    inputs = manifest.get("inputs", [])
    documents = manifest.get("documents", [])
    manifest["input_count"] = len(inputs)
    manifest["document_count"] = len(documents)
    manifest["duplicate_count"] = sum(item.get("status") == "duplicate" for item in inputs)
    manifest["completed_count"] = sum(item.get("status") == "completed" for item in documents)
    manifest["failed_count"] = sum(item.get("status") == "failed" for item in documents)
    manifest["skipped_count"] = sum(item.get("status") == "skipped" for item in inputs)
    # 旧测试和少量外部调用方仍读取该视图；核心数据仍是 inputs/documents。
    manifest["papers"] = _legacy_papers(manifest)


def _write_state(task_dir: Path, manifest: dict[str, Any]) -> None:
    """原子保存 manifest 和状态摘要，避免半份 JSON 被续跑读取。"""
    _refresh_counts(manifest)
    atomic_write_json(manifest, _manifest_path(task_dir))
    status = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": manifest.get("task_id"),
        "status": manifest.get("status"),
        "report_mode": manifest.get("report_mode"),
        "input_count": manifest.get("input_count", 0),
        "document_count": manifest.get("document_count", 0),
        "duplicate_count": manifest.get("duplicate_count", 0),
        "completed_count": manifest.get("completed_count", 0),
        "failed_count": manifest.get("failed_count", 0),
        "skipped_count": manifest.get("skipped_count", 0),
        "updated_at": manifest.get("updated_at"),
    }
    atomic_write_json(status, _status_path(task_dir))


def _load_manifest(task_dir: Path) -> dict[str, Any]:
    path = _manifest_path(task_dir)
    if not path.is_file():
        raise ValueError("指定的批次目录缺少 batch_manifest.json，无法续跑。")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("批次状态文件无法读取，无法安全续跑。") from exc
    if not isinstance(manifest, dict):
        raise ValueError(MANIFEST_COMPATIBILITY_ERROR)
    _compatibility_check(manifest)
    return manifest


def _input_records(paths: list[Path], old_inputs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    old_by_alias = {str(item.get("source_alias")): item for item in (old_inputs or [])}
    occurrences: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for path in paths:
        base_alias = _source_alias(path)
        occurrences[base_alias] = occurrences.get(base_alias, 0) + 1
        alias = _source_alias(path, occurrences[base_alias])
        old = old_by_alias.get(alias)
        input_id = str(old.get("input_id")) if old and old.get("input_id") else f"input_{uuid.uuid4().hex[:12]}"
        fingerprint = _fingerprint(path)
        records.append({
            "input_id": input_id,
            "display_name": path.name,
            "source_alias": alias,
            "content_sha256": fingerprint,
            "document_id": _document_id(fingerprint, input_id),
            "status": "pending",
            "duplicate_of": None,
        })
    return records


def _build_documents(inputs: list[dict[str, Any]], old_documents: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    old_by_id = {str(item.get("document_id")): item for item in (old_documents or [])}
    documents: list[dict[str, Any]] = []
    by_doc_id: dict[str, dict[str, Any]] = {}
    for item in inputs:
        document_id = str(item["document_id"])
        if document_id in by_doc_id:
            item["status"] = "duplicate"
            item["duplicate_of"] = by_doc_id[document_id]["input_ids"][0]
            by_doc_id[document_id]["input_ids"].append(item["input_id"])
            continue
        cached = old_by_id.get(document_id, {})
        document = {
            "document_id": document_id,
            "content_sha256": item.get("content_sha256"),
            "display_name": item.get("display_name"),
            "source_alias": item.get("source_alias"),
            "input_ids": [item["input_id"]],
            "status": cached.get("status") if cached.get("status") in {"completed", "failed"} else "pending",
            "result_file": cached.get("result_file") or f"documents/{_safe_stem(item['display_name'])}_{document_id[:18]}.json",
            "errors": list(cached.get("errors") or []),
        }
        documents.append(document)
        by_doc_id[document_id] = document
    return documents


def _set_input_statuses(inputs: list[dict[str, Any]], documents: list[dict[str, Any]]) -> None:
    by_id = {doc["document_id"]: doc for doc in documents}
    for item in inputs:
        document = by_id[item["document_id"]]
        item["status"] = "duplicate" if item.get("duplicate_of") else document.get("status", "pending")


def _attach_identity(paper: dict[str, Any], input_item: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    result = dict(paper)
    result["file_name"] = input_item.get("display_name") or result.get("file_name")
    result["document_id"] = document["document_id"]
    result["content_sha256"] = document.get("content_sha256")
    result["source_alias"] = input_item.get("source_alias")
    return result


@dataclass
class LocalBatchResult:
    task_dir: Path
    papers: list[dict[str, Any]]
    manifest: dict[str, Any]


def run_local_batch(
    pdf_files: Iterable[str | Path],
    *,
    output_root: str | Path = "output",
    report_mode: str | None = None,
    resume_dir: str | Path | None = None,
    pause_after: int | None = None,
    progress: ProgressCallback | None = None,
    extractor: Callable[[str | Path], dict[str, Any]] | None = None,
) -> LocalBatchResult:
    """执行或续跑离线批次；不会初始化任何 AI Provider。"""
    extract_fn = extractor or extract_local_paper
    paths = [Path(raw_path) for raw_path in pdf_files]
    task_dir = Path(resume_dir) if resume_dir else _new_task_dir(output_root)
    old_manifest: dict[str, Any] | None = None
    if resume_dir:
        old_manifest = _load_manifest(task_dir)
        manifest = old_manifest
        if report_mode:
            manifest["report_mode"] = report_mode
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "task_id": task_dir.name,
            "status": "running",
            "report_mode": report_mode or "自动选择",
            "created_at": _now(),
            "updated_at": _now(),
            "inputs": [],
            "documents": [],
        }

    inputs = _input_records(paths, (old_manifest or {}).get("inputs"))
    documents = _build_documents(inputs, (old_manifest or {}).get("documents"))
    manifest["inputs"] = inputs
    manifest["documents"] = documents
    manifest["updated_at"] = _now()
    _set_input_statuses(inputs, documents)
    _write_state(task_dir, manifest)

    path_by_input_id = {item["input_id"]: path for item, path in zip(inputs, paths)}
    input_by_doc: dict[str, dict[str, Any]] = {}
    for item in inputs:
        input_by_doc.setdefault(item["document_id"], item)
    total = len(documents)
    processed_new = 0
    papers: list[dict[str, Any]] = []
    for index, document in enumerate(documents, 1):
        input_item = input_by_doc[document["document_id"]]
        result_path = task_dir / document["result_file"]
        cached: dict[str, Any] | None = None
        if document.get("status") in {"completed", "failed"} and result_path.is_file():
            try:
                value = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    cached = value
            except (OSError, json.JSONDecodeError):
                cached = None
                document["status"] = "pending"
                document["errors"] = []
        if cached is not None:
            papers.append(cached)
            if progress:
                progress(index / max(1, total), f"复用已完成论文 {index}/{total}")
            continue
        if pause_after is not None and processed_new >= max(0, int(pause_after)):
            manifest["status"] = "paused"
            manifest["updated_at"] = _now()
            _set_input_statuses(inputs, documents)
            _write_state(task_dir, manifest)
            return LocalBatchResult(task_dir, papers, manifest)
        document["status"] = "running"
        input_item["status"] = "running"
        manifest["updated_at"] = _now()
        _write_state(task_dir, manifest)
        if progress:
            progress((index - 1) / max(1, total), f"本地解析 {index}/{total}")
        try:
            paper = extract_fn(path_by_input_id[input_item["input_id"]])
        except Exception as exc:
            paper = {
                "file_name": input_item["display_name"],
                "pages": [],
                "chunks": [],
                "sections": {},
                "facts": [],
                "warnings": [],
                "errors": [f"本地解析失败：{type(exc).__name__}"],
            }
        paper = _attach_identity(paper, input_item, document)
        atomic_write_json(paper, result_path)
        papers.append(paper)
        processed_new += 1
        document["status"] = "completed" if not paper.get("errors") else "failed"
        document["errors"] = list(paper.get("errors") or [])
        input_item["status"] = document["status"]
        manifest["status"] = "running"
        manifest["updated_at"] = _now()
        _set_input_statuses(inputs, documents)
        _write_state(task_dir, manifest)

    manifest["status"] = "completed_with_errors" if any(doc.get("status") == "failed" for doc in documents) else "completed"
    manifest["updated_at"] = _now()
    _set_input_statuses(inputs, documents)
    _write_state(task_dir, manifest)
    return LocalBatchResult(task_dir, papers, manifest)


def update_batch_manifest(
    task_dir: str | Path,
    document_updates: dict[str, dict[str, Any]] | None = None,
    report_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """原子更新选择信息和报告相对路径。"""
    target = Path(task_dir)
    manifest = _load_manifest(target)
    by_id = {doc.get("document_id"): doc for doc in manifest.get("documents", [])}
    for document_id, update in (document_updates or {}).items():
        if document_id in by_id:
            by_id[document_id].update(dict(update))
    for document_id, relative_path in (report_paths or {}).items():
        if document_id in by_id:
            by_id[document_id]["individual_report"] = relative_path
    manifest["updated_at"] = _now()
    _write_state(target, manifest)
    return manifest


def resume_local_batch(resume_dir: str | Path, pdf_files: Iterable[str | Path], **kwargs: Any) -> LocalBatchResult:
    return run_local_batch(pdf_files, resume_dir=resume_dir, **kwargs)


def pause_local_batch(task_dir: str | Path) -> Path:
    """显式将已保存的批次标记为暂停，不删除任何中间结果。"""
    target = Path(task_dir)
    manifest = _load_manifest(target)
    manifest["status"] = "paused"
    manifest["updated_at"] = _now()
    _write_state(target, manifest)
    return _manifest_path(target)
