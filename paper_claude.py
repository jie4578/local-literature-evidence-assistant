"""
======================================================================
 🧬 Paper Analyzer Pro v7.0 — 网页版
======================================================================
 用法：
    python paper_claude.py
    
 然后浏览器打开 http://localhost:7860
======================================================================
"""

import fitz
import json
import re
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from openai import OpenAI
import gradio as gr
from paper_pipeline import PipelineConfig, run_batch
from exporters import export_csv, export_excel, export_json, export_word
from local_extractor import extract_local_paper
from local_summary import compare_papers, extractive_summary
from local_search import LocalSearchIndex

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)

# ╔═══════════════════════════════════════════╗
# ║  配置区域                                  ║
# ╚═══════════════════════════════════════════╝

API_BASE_URL = "https://api.deepseek.com"
AI_MODEL = "deepseek-chat"

# ============================================================
#  核心功能函数（复用原有逻辑）
# ============================================================

def get_client(api_key):
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise Exception(f"无法打开 PDF: {e}")
    if doc.is_encrypted:
        doc.close()
        raise Exception("PDF 已加密，无法读取")
    text = ""
    for i in range(doc.page_count):
        page_text = doc[i].get_text()
        if page_text.strip():
            text += page_text + "\n"
    doc.close()
    if not text.strip():
        raise Exception("PDF 内容为空（可能是扫描版图片）")
    return text


def call_ai(client, messages, temperature=0.3, max_retries=2):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                temperature=temperature,
                timeout=60
            )
            return res.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep((attempt + 1) * 3)
            else:
                raise Exception(f"AI 调用失败（已重试{max_retries}次）: {last_error}")


def analyze_single_paper(client, text, paper_name=""):
    truncated = text
    prompt = f"""请以专业科研分析助手的身份，对以下论文进行结构化分析。

论文名称：{paper_name}

请严格按照以下格式输出（** 表示加粗内容）：

**研究背景**
（2-3句话说明研究背景和问题）

**研究方法**
（说明采用的核心方法和技术路线）

**创新点**
（列出2-3个关键创新点）

**结论**
（总结核心发现和结论）

——————————
论文内容：
{truncated}
"""
    messages = [
        {"role": "system", "content": "你是专业的生物医学文献分析助手。请严格使用 **加粗** 标记输出结构标题，输出内容要专业、简洁、有深度。"},
        {"role": "user", "content": prompt}
    ]
    return call_ai(client, messages, temperature=0.3)


def generate_review(client, all_analyses):
    prompt = f"""你是一位顶级的科研综述专家。请根据以下多篇论文的分析结果，撰写一篇高质量的文献综述。

要求：
1. **研究背景** — 该领域的整体研究背景和意义
2. **方法分类** — 各篇论文使用的方法归纳分类
3. **对比分析** — 不同研究的异同点、优劣势对比
4. **研究不足** — 现有研究的共同局限
5. **未来方向** — 该领域未来的研究方向

写作规范：
- 用 **加粗** 标记以上5个标题
- 正文中关键术语适当用 **加粗** 强调
- 语言专业流畅，符合学术综述风格

各篇论文分析：
{all_analyses}
"""
    messages = [
        {"role": "system", "content": "你是顶级的科研综述写作专家。请严格使用 **加粗** 标记小标题和关键术语。"},
        {"role": "user", "content": prompt}
    ]
    return call_ai(client, messages, temperature=0.4)


def add_formatted_paragraph(doc, line):
    heading_keywords = [
        "研究背景", "背景", "方法", "方法分类", "创新点", "创新",
        "结论", "总结", "对比分析", "研究不足", "未来方向",
        "研究方法", "核心结果"
    ]
    clean = line.replace("###", "").replace("##", "").replace("#", "").strip()
    is_heading = False
    for kw in heading_keywords:
        if clean.startswith(kw) or clean == kw:
            is_heading = True
            break
        if clean == f"**{kw}**" or clean.startswith(f"**{kw}**"):
            is_heading = True
            break
        no_num = re.sub(r"^\d+[\.\)、\s]*", "", clean)
        if no_num == f"**{kw}**" or no_num.startswith(f"**{kw}**") or no_num == kw:
            is_heading = True
            break
    if is_heading:
        heading_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
        heading_text = re.sub(r"^\d+[\.\)、\s]*", "", heading_text).strip()
        doc.add_heading(heading_text, level=2)
        return
    if "**" in line:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        parts = re.split(r"(\*\*[^*]+\*\*)", line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                bold_text = part[2:-2]
                if bold_text.strip():
                    run = p.add_run(bold_text)
                    run.bold = True
                    run.font.color.rgb = RGBColor(0, 51, 153)
                    run.font.name = 'Times New Roman'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            else:
                if part.strip():
                    run = p.add_run(part)
                    run.font.name = 'Times New Roman'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    else:
        clean_line = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
        if clean_line:
            p = doc.add_paragraph(clean_line)
            p.paragraph_format.space_after = Pt(2)


def save_docx(title, content, output_path):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.2)
    section.right_margin = Inches(1.2)
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(2)
    h = doc.add_heading(title, level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    time_p = doc.add_paragraph()
    time_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = time_p.add_run(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(128, 128, 128)
    doc.add_paragraph("")
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("———") or line.startswith("==="):
            continue
        add_formatted_paragraph(doc, line)
    doc.save(output_path)
    return output_path


# ============================================================
#  Gradio 处理函数
# ============================================================

def process_local_papers(pdf_files, progress=gr.Progress()):
    """本地离线解析：不创建 DeepSeek 客户端、不访问网络。"""
    if not pdf_files:
        return "❌ 请上传至少一个 PDF 文件", None
    paths = [pdf_file.name if hasattr(pdf_file, "name") else pdf_file for pdf_file in pdf_files]
    task_dir = os.path.join("output", f"local_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    os.makedirs(task_dir, exist_ok=False)
    papers = []
    local_index = LocalSearchIndex(os.path.join("output", "local_index.sqlite"))
    for index, path in enumerate(paths, 1):
        progress(index / max(1, len(paths)), desc=f"本地解析 {index}/{len(paths)}")
        papers.append(extract_local_paper(path))
        local_index.index_pages(papers[-1].get("pages", []))
    comparison = compare_papers(papers)
    export_json(papers, os.path.join(task_dir, "local_papers.json"))
    export_json(comparison, os.path.join(task_dir, "comparison.json"))
    export_csv(comparison, os.path.join(task_dir, "comparison.csv"))
    export_excel(comparison, os.path.join(task_dir, "comparison.xlsx"))
    output_path = export_word(comparison, os.path.join(task_dir, "comparison.docx"))
    lines = ["模式：Local Offline", f"本地结果目录：{task_dir}"]
    lines.append("提示：本地规则提取暂不能稳定区分正文、表格标题和表格内容，重要结果请根据 PDF 页码回查原文。摘取式摘要仅选取原句，不代表 AI 生成或事实核验。")
    for paper in papers:
        lines.append(f"\n📄 {paper['file_name']}：{len(paper.get('pages', []))} 页，{len(paper.get('facts', []))} 条规则证据")
        if paper.get("errors"):
            lines.extend(f"错误：{error}" for error in paper["errors"])
        if paper.get("warnings"):
            lines.extend(f"警告：{warning}" for warning in paper["warnings"])
        summary = extractive_summary(paper)
        if summary["sentences"]:
            lines.append("摘取式摘要：")
            lines.extend(f"- {sentence}" for sentence in summary["sentences"])
    return "\n".join(lines), str(output_path)


def search_local_index(query, limit=20):
    """查询本地 SQLite 索引；不访问网络。"""
    if not query or not query.strip():
        return "请输入关键词或精确短语。"
    rows = LocalSearchIndex(os.path.join("output", "local_index.sqlite")).search(query, limit=limit)
    if not rows:
        return "未找到匹配内容。"
    return "\n\n".join(f"{row['source_file']} · PDF 第 {row['page_number']} 页\n{row['snippet']}" for row in rows)


def process_papers(pdf_files, api_key, mode=None, progress=gr.Progress()):
    """主处理函数：接收上传的PDF，返回综述文本和Word文件"""

    if mode == "Local Offline":
        return process_local_papers(pdf_files, progress)

    # 验证输入 — 优先用网页输入的 Key，没填则尝试环境变量
    if not api_key or not api_key.strip():
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or not api_key.strip():
        return "❌ 请先填写 DeepSeek API Key，或在环境变量中设置 DEEPSEEK_API_KEY", None

    if not pdf_files:
        return "❌ 请上传至少一个 PDF 文件", None

    api_key = api_key.strip()
    client = get_client(api_key)

    log_lines = []
    paths = [pdf_file.name if hasattr(pdf_file, "name") else pdf_file for pdf_file in pdf_files]
    progress(0.05, desc="正在进行分页提取与分段分析...")
    pipeline_result = run_batch(client, paths, output_root="output", config=PipelineConfig())
    success_count = sum(not paper.get("errors") for paper in pipeline_result.papers)
    fail_count = len(pipeline_result.papers) - success_count
    for paper in pipeline_result.papers:
        log_lines.append(f"📄 {paper.get('file_name', '未知文件')}：{paper.get('chunk_count', 0)} 个分块")
    progress(0.85, desc="正在整理结构化文献综述...")
    review = _review_to_text(pipeline_result.final_review)
    log_lines.append(f"\n📝 综述处理完成（成功 {success_count} 篇，存在警告/错误 {fail_count} 篇）")
    log_lines.append(f"📁 中间结果：{pipeline_result.task_dir}")

    # 保存 Word
    progress(0.95, desc="正在生成 Word 文档...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(str(pipeline_result.task_dir), f"文献综述_{timestamp}.docx")
        save_docx("文献综述", review, output_path)
        log_lines.append(f"✅ Word 文档已生成")
    except Exception as e:
        return f"❌ Word 生成失败: {e}", None

    progress(1.0, desc="完成！")

    # 返回综述文本 + Word 文件路径
    result_text = "\n".join(log_lines) + "\n\n" + "="*50 + "\n\n" + review
    return result_text, output_path


def _review_to_text(review):
    """将结构化综述转换为兼容现有文本框和 Word 导出的简洁文本。"""
    labels = {
        "research_theme_overview": "研究主题概述", "major_methods": "主要研究方法",
        "common_conclusions": "共同结论", "different_or_conflicting_conclusions": "不同或冲突结论",
        "research_innovations": "研究创新", "current_limitations": "现有局限",
        "research_gaps": "研究空白", "future_recommendations": "后续研究建议",
        "papers": "涉及的论文清单", "evidence": "可核查证据",
    }
    lines = []
    for key, label in labels.items():
        value = review.get(key)
        if value in (None, [], ""):
            continue
        lines.append(f"**{label}**")
        if isinstance(value, list):
            lines.extend(f"- {item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)}" for item in value)
        else:
            lines.append(str(value))
    if review.get("warnings"):
        lines.extend(["**警告**", *[f"- {item}" for item in review["warnings"]]])
    if review.get("errors"):
        lines.extend(["**错误摘要**", *[f"- {item}" for item in review["errors"]]])
    return "\n".join(lines) or "原文未明确说明。"


# ============================================================
#  Gradio 界面
# ============================================================

def build_ui():
    with gr.Blocks(
        title="🧬 科研文献智能分析系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .header { text-align: center; padding: 20px 0 10px 0; }
        .header h1 { font-size: 2em; color: #1e3a5f; }
        .header p { color: #666; margin-top: 5px; }
        .tip-box { background: #f0f7ff; border-left: 4px solid #3b82f6;
                   padding: 10px 15px; border-radius: 4px; margin: 10px 0; font-size: 0.9em; }
        footer { display: none !important; }
        """
    ) as demo:

        # 顶部标题
        gr.HTML("""
        <div class="header">
            <h1>🧬 科研文献智能分析系统</h1>
            <p>上传 PDF 论文 → 本地解析或可选 AI 分析 → 结构化结果导出</p>
        </div>
        """)

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 配置")

                mode_input = gr.Radio(
                    ["Local Offline", "DeepSeek AI"], value="Local Offline",
                    label="运行模式", info="默认仅使用本地确定性功能，不访问网络"
                )

                api_key_input = gr.Textbox(
                    label="DeepSeek API Key",
                    placeholder="your_api_key_here",
                    type="password",
                    info="在 platform.deepseek.com 获取"
                )

                gr.HTML('<div class="tip-box">💡 可选功能：论文文本将发送至 DeepSeek API。请勿上传涉密、敏感或未授权材料。<br>Local Offline 模式不需要 API Key，也不访问网络。</div>')

                gr.Markdown("### 📂 上传论文")

                pdf_input = gr.File(
                    label="选择 PDF 文件（可多选）",
                    file_count="multiple",
                    file_types=[".pdf"],
                )

                gr.HTML('<div class="tip-box">📌 支持同时上传多篇论文，自动批量分析后生成综合综述</div>')

                submit_btn = gr.Button(
                    "🚀 开始处理",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("### 📖 使用说明")
                gr.Markdown("""
1. 默认选择 Local Offline，无需 API Key
2. 上传一篇或多篇 PDF 论文
3. 点击「开始处理」查看本地解析和结构化对比
4. 如主动选择 DeepSeek AI，再填写 API Key

> ⚠️ 仅支持**文字版 PDF**，扫描版图片 PDF 无法读取
                """)

            # 右侧：输出区
            with gr.Column(scale=2):
                gr.Markdown("### 📊 分析结果")

                result_text = gr.Textbox(
                    label="分析进度 & 综述内容",
                    lines=25,
                    max_lines=40,
                    placeholder="点击「开始分析」后，这里会显示实时进度和综述内容...",
                    
                )

                docx_output = gr.File(
                    label="📥 下载 Word 文档",
                    visible=True,
                )

                local_query = gr.Textbox(label="本地证据搜索", placeholder="关键词或精确短语（仅搜索本地索引）")
                local_search_btn = gr.Button("🔎 搜索本地证据")
                local_search_output = gr.Textbox(label="本地搜索结果", lines=8)

        # 绑定事件
        submit_btn.click(
            fn=process_papers,
            inputs=[pdf_input, api_key_input, mode_input],
            outputs=[result_text, docx_output],
        )
        local_search_btn.click(search_local_index, inputs=[local_query], outputs=[local_search_output])

    return demo


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🧬 科研文献智能分析系统 — 网页版")
    print("="*55)
    print("  启动中，请稍候...")
    print("  启动后浏览器访问: http://localhost:7860")
    print("="*55 + "\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,          # 改成 True 可生成公网链接（临时）
        inbrowser=True,       # 自动打开浏览器
        show_error=True,
    )
