"""大批量本地报告的模式与布局规划。

该模块只做确定性的报告规划，不创建 Provider，也不访问网络。
"""

from __future__ import annotations

from typing import Any


SELECTION_STRATEGY = "metadata_completeness_verified_evidence_section_coverage_quality_document_id"


REPORT_MODE_OPTIONS = (
    "自动选择",
    "仅单篇报告",
    "简洁对比",
    "文献库汇总",
    "不生成 Word，仅导出 Excel/JSON",
)

NO_WORD_MODE = "不生成 Word，仅导出 Excel/JSON"


def resolve_report_mode(requested_mode: str | None, paper_count: int) -> str:
    """根据用户选项和论文数量确定稳定的报告布局。"""
    mode = requested_mode if requested_mode in REPORT_MODE_OPTIONS else "自动选择"
    if mode == "自动选择":
        if paper_count <= 1:
            return "仅单篇报告"
        if paper_count <= 3:
            return "简洁对比"
        return "文献库汇总"
    if mode == "简洁对比" and paper_count <= 1:
        return "仅单篇报告"
    if mode == "文献库汇总" and paper_count <= 1:
        return "仅单篇报告"
    return mode


def report_layout_plan(requested_mode: str | None, paper_count: int) -> dict[str, Any]:
    """返回供界面、批次导出和测试共同使用的布局计划。

    4 篇以上始终使用纵向汇总，避免把论文数量映射为表格列数。
    """
    count = max(0, int(paper_count))
    resolved = resolve_report_mode(requested_mode, count)
    word_enabled = resolved != NO_WORD_MODE
    if not word_enabled:
        layout = "none"
        show_comparison = False
    elif resolved == "仅单篇报告" or count <= 1:
        layout = "single"
        show_comparison = False
    elif count <= 3:
        layout = "matrix"
        show_comparison = True
    else:
        layout = "vertical"
        show_comparison = True
    return {
        "requested_mode": requested_mode if requested_mode in REPORT_MODE_OPTIONS else "自动选择",
        "resolved_mode": resolved,
        "paper_count": count,
        "layout": layout,
        "show_comparison": show_comparison,
        "word_enabled": word_enabled,
        "max_word_papers": min(count, 10) if word_enabled else 0,
        "individual_reports": word_enabled and count > 3,
        "full_data_required": count > 3,
    }


def _selection_score(paper: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """为 Word 展示论文计算可解释、稳定且不依赖模型的分数。"""
    metadata = sum(bool(paper.get(key)) for key in ("title_candidate", "author_candidate", "year_candidate", "doi"))
    verified = sum(1 for item in (paper.get("facts") or []) if item.get("verified") is True)
    sections = paper.get("sections") or {}
    section_count = sum(1 for key, value in sections.items() if key != "references" and isinstance(value, dict) and value.get("text"))
    quality_flags = len(paper.get("text_quality_flags") or [])
    components = {
        "metadata": metadata,
        "verified_evidence": verified,
        "sections": section_count,
        "quality_flags": quality_flags,
    }
    score = metadata * 10 + min(verified, 20) * 2 + min(section_count, 8) - quality_flags * 2
    return score, components


def select_word_papers(papers: list[dict[str, Any]], max_papers: int = 10) -> list[dict[str, Any]]:
    """生成 Word 展示论文选择记录；不把顺序选择描述成学术代表性。"""
    records: list[dict[str, Any]] = []
    for paper in papers:
        document_id = str(paper.get("document_id") or paper.get("content_sha256") or paper.get("file_name") or "")
        failed = bool(paper.get("errors"))
        score, components = _selection_score(paper)
        reason = (
            f"元数据 {components['metadata']}/4；已验证证据 {components['verified_evidence']} 条；"
            f"章节覆盖 {components['sections']} 项；文本质量警告 {components['quality_flags']} 条；"
            "按 document_id 稳定排序。"
        )
        if failed:
            reason = "分析失败，不进入 Word 展示。"
        records.append({
            "document_id": document_id,
            "selection_strategy": SELECTION_STRATEGY,
            "included_in_word": False,
            "selection_rank": None,
            "selection_score": score,
            "selection_reason": reason,
        })
    eligible = [record for record, paper in zip(records, papers) if not paper.get("errors")]
    eligible.sort(key=lambda item: (-int(item["selection_score"]), item["document_id"]))
    for rank, record in enumerate(eligible[:max(0, int(max_papers))], 1):
        record["included_in_word"] = True
        record["selection_rank"] = rank
    return records
