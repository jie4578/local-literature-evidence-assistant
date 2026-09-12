"""
======================================================================
 🧬 Paper Analyzer Pro v6.0 — 文献分析兼职交付版
======================================================================
 功能：
  1. 批量读取 paper/ 文件夹下的 PDF（支持拖拽单文件）
  2. AI 分析每篇论文（研究背景、方法、创新点、结论）
  3. 自动生成文献综述
  4. 输出美观 Word 文档（蓝色加粗 + 标题层级 + 表格对比）
  
 用法：
    python paper_analyzer_pro.py
    python paper_analyzer_pro.py --input "论文.pdf"
    python paper_analyzer_pro.py --input "paper/" --output "我的综述.docx"
    
 兼职交付标准：
   ✅ 异常处理 — 单篇失败自动跳过，不影响整体
   ✅ 进度提示 — 命令行实时显示进度
   ✅ 防覆盖 — 自动在文件名后加时间戳
   ✅ 容错 — PDF 加密/空白/格式错误均能处理
======================================================================
"""

import fitz
import glob
import re
import os
import sys
import argparse
import time
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from openai import OpenAI

# ╔═══════════════════════════════════════════╗
# ║  配置区域 — 兼职时客户可在此修改          ║
# ╚═══════════════════════════════════════════╝

# API 配置（建议用环境变量，防止 Key 泄露）
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_BASE_URL = "https://api.deepseek.com"
AI_MODEL = "deepseek-chat"

# 输出配置
DEFAULT_OUTPUT_DIR = "output"           # 输出文件夹
DEFAULT_OUTPUT_NAME = "文献综述"         # 默认文件名

# ============================================================
#  以下为核心代码，一般无需修改
# ============================================================

def get_client():
    if not API_KEY:
        raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY 环境变量，请先设置 DeepSeek API Key")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def log(msg, level="INFO"):
    """带时间戳的统一日志输出"""
    now = datetime.now().strftime("%H:%M:%S")
    icon_map = {"INFO": "📌", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "PROGRESS": "⏳"}
    icon = icon_map.get(level, "📌")
    print(f"  {icon} [{now}] {msg}")


def read_pdf(file_path):
    """
    读取 PDF 文本内容。
    支持加密检测、空页跳过、编码兼容。
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise Exception(f"无法打开 PDF 文件: {e}")

    # 检查是否加密
    if doc.is_encrypted:
        doc.close()
        raise Exception("PDF 已加密，无法读取内容")

    text = ""
    page_count = doc.page_count
    for i in range(page_count):
        page = doc[i]
        page_text = page.get_text()
        if page_text.strip():
            # 跳过全是空白/图片的页
            text += page_text + "\n"

    doc.close()

    if not text.strip():
        raise Exception("PDF 内容为空（可能是纯图片扫描版）")

    return text


def call_ai(messages, model=AI_MODEL, temperature=0.3, max_retries=2):
    """
    调用 AI API，带重试机制。
    兼职场景下网络/API不稳定很常见。
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            res = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=60
            )
            return res.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = (attempt + 1) * 3
                log(f"API 调用失败，{wait}秒后重试 ({attempt+1}/{max_retries}): {e}", "WARN")
                time.sleep(wait)
            else:
                raise Exception(f"AI API 调用失败（已重试{max_retries}次）: {last_error}")


def analyze_single_paper(text, paper_name=""):
    """
    分析单篇论文，返回结构化分析结果。
    """
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
{text}
"""

    messages = [
        {"role": "system", "content": "你是专业的生物医学文献分析助手。请严格使用 **加粗** 标记输出结构标题，输出内容要专业、简洁、有深度。"},
        {"role": "user", "content": prompt}
    ]

    return call_ai(messages, temperature=0.3)


def generate_review(all_analyses):
    """
    综合所有论文分析结果，生成文献综述。
    """
    prompt = f"""你是一位顶级的科研综述专家。请根据以下多篇论文的分析结果，撰写一篇高质量的文献综述。

要求：
1. **研究背景** — 该领域的整体研究背景和意义
2. **方法分类** — 各篇论文使用的方法归纳分类（最好用对比视角）
3. **对比分析** — 不同研究的异同点、优劣势对比
4. **研究不足** — 现有研究的共同局限
5. **未来方向** — 该领域未来的研究方向和潜在突破点

写作规范：
- 用 **加粗** 标记以上5个标题
- 正文中关键术语/重要结论也适当用 **加粗** 强调
- 语言专业流畅，符合学术综述风格
- 避免"首先、其次、最后"等说教语气

各篇论文分析：
{all_analyses}
"""

    messages = [
        {"role": "system", "content": "你是顶级的科研综述写作专家。请严格使用 **加粗** 标记小标题和关键术语。"},
        {"role": "user", "content": prompt}
    ]

    return call_ai(messages, temperature=0.4)


def add_formatted_paragraph(doc, line):
    """
    将一行文本添加到 Word 文档中，支持：
    - **加粗** → 蓝色加粗
    - 标题行识别 → Heading 样式
    - 编号去除
    """
    # ===== 标题关键词列表 =====
    heading_keywords = [
        "研究背景", "背景", "方法", "方法分类", "创新点", "创新",
        "结论", "总结", "对比分析", "研究不足", "未来方向",
        "核心结果", "局限性", "应用方向", "通俗解释",
        "论文一句话总结", "研究方法", "核心结果"
    ]

    # 去掉 Markdown 标题标记，用于文本判断
    clean = line.replace("###", "").replace("##", "").replace("#", "").strip()

    # ===== 判断是否为标题行 =====
    is_heading = False
    for kw in heading_keywords:
        if clean.startswith(kw) or clean == kw:
            is_heading = True
            break
        # 匹配 **研究背景** 这种模式
        if clean == f"**{kw}**" or clean.startswith(f"**{kw}**"):
            is_heading = True
            break
        # 匹配带编号的 "1. **研究背景**"
        no_num = re.sub(r"^\d+[\.\)、\s]*", "", clean)
        if no_num == f"**{kw}**" or no_num.startswith(f"**{kw}**") or no_num == kw:
            is_heading = True
            break

    if is_heading:
        # 去掉 ** 标记和编号，设为二级标题
        heading_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean)
        heading_text = re.sub(r"^\d+[\.\)、\s]*", "", heading_text).strip()
        doc.add_heading(heading_text, level=2)
        return

    # ===== 处理含 **加粗** 的段落 =====
    if "**" in line:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)

        # 按 **...** 分割，交替处理普通文本和加粗文本
        parts = re.split(r"(\*\*[^*]+\*\*)", line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                bold_text = part[2:-2]
                if bold_text.strip():
                    run = p.add_run(bold_text)
                    run.bold = True
                    run.font.color.rgb = RGBColor(0, 51, 153)  # 深蓝色
                    run.font.name = 'Times New Roman'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            else:
                if part.strip():
                    run = p.add_run(part)
                    run.font.name = 'Times New Roman'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    else:
        # ===== 普通段落 =====
        clean_line = re.sub(r"^\d+[\.\)、\s]*", "", line).strip()
        if clean_line:
            p = doc.add_paragraph(clean_line)
            p.paragraph_format.space_after = Pt(2)


def save_docx(title, content, output_path):
    """
    保存为格式化的 Word 文档。
    支持标题层级、蓝色加粗、统一字体。
    """
    doc = Document()

    # ===== 页面设置 =====
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1.2)
    section.right_margin = Inches(1.2)

    # ===== 全局字体设置 =====
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    # 段落间距
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(2)

    # ===== 封面标题 =====
    h = doc.add_heading(title, level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 生成时间
    time_p = doc.add_paragraph()
    time_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = time_p.add_run(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(128, 128, 128)

    doc.add_paragraph("")  # 空行

    # ===== 逐行处理内容 =====
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过 Markdown 分割线
        if line.startswith("———") or line.startswith("==="):
            continue
        add_formatted_paragraph(doc, line)

    doc.save(output_path)
    return output_path


def find_pdf_files(input_path):
    """智能查找 PDF 文件列表"""
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".pdf"):
            return [input_path]
        else:
            log(f"指定文件不是 PDF 格式: {input_path}", "WARN")
            return []

    if os.path.isdir(input_path):
        pdfs = glob.glob(os.path.join(input_path, "*.pdf"))
        if not pdfs:
            # 尝试子目录
            pdfs = glob.glob(os.path.join(input_path, "**/*.pdf"), recursive=True)
        return sorted(pdfs)

    return []


def generate_output_filename(output_dir, base_name, extension=".docx"):
    """生成不覆盖已有文件的输出路径"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}{extension}"
    return os.path.join(output_dir, filename)


def main():
    # ===== 命令行参数解析 =====
    parser = argparse.ArgumentParser(
        description="🧬 Paper Analyzer Pro — 文献分析工具（兼职交付版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python paper_analyzer_pro.py
  python paper_analyzer_pro.py --input "paper/论文1.pdf" --output "分析报告.docx"
  python paper_analyzer_pro.py --input "paper/" --model "deepseek-chat"
        """
    )
    parser.add_argument("--input", "-i", default="paper",
                        help="输入 PDF 文件路径或文件夹路径（默认: paper/）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出 Word 文件路径（默认自动生成）")
    parser.add_argument("--model", "-m", default=AI_MODEL,
                        help=f"AI 模型名称（默认: {AI_MODEL}）")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}/）")
    args = parser.parse_args()

    # ===== 打印启动信息 =====
    print("")
    print("=" * 60)
    print("  🧬 Paper Analyzer Pro v6.0")
    print("  文献分析 · 综述生成 · Word 美化输出")
    print("=" * 60)
    print("")

    # ===== 查找 PDF =====
    log(f"正在扫描 PDF 文件: {args.input}", "INFO")
    pdf_files = find_pdf_files(args.input)

    if not pdf_files:
        log(f"未找到任何 PDF 文件（路径: {args.input}）", "ERROR")
        log("请确保路径正确，或直接指定 PDF 文件路径", "INFO")
        sys.exit(1)

    log(f"找到 {len(pdf_files)} 篇 PDF 论文", "OK")
    for f in pdf_files:
        print(f"     📄 {os.path.basename(f)}")

    print("")
    log("开始 AI 分析...", "INFO")
    print("")

    # ===== 逐篇分析 =====
    all_analyses = ""
    success_count = 0
    fail_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_path)
        print(f"  ──── [{idx}/{len(pdf_files)}] {pdf_name} ────")

        try:
            # 读取 PDF
            log("读取 PDF 内容...", "PROGRESS")
            text = read_pdf(pdf_path)
            log(f"成功读取 ({len(text)} 字符)", "OK")

            # AI 分析
            log("调用 AI 分析论文...", "PROGRESS")
            result = analyze_single_paper(text, pdf_name)
            log("分析完成", "OK")

            # 收集结果
            separator = f"\n\n─── 📄 论文：{pdf_name} ───\n\n"
            all_analyses += separator + result

            success_count += 1

        except Exception as e:
            log(f"处理失败: {e}", "ERROR")
            fail_count += 1
            # 继续处理下一篇，不影响整体

        print("")  # 空行分隔

    # ===== 生成综述 =====
    if success_count == 0:
        log("所有论文均分析失败，无法生成综述", "ERROR")
        sys.exit(1)

    log(f"分析完成：成功 {success_count} 篇，失败 {fail_count} 篇", "OK")
    print("")

    if success_count >= 1:
        log("正在生成文献综述...", "PROGRESS")
        try:
            review = generate_review(all_analyses)
            log("文献综述生成完成", "OK")
        except Exception as e:
            log(f"综述生成失败: {e}", "ERROR")
            sys.exit(1)

        # ===== 生成 Word =====
        log("正在生成 Word 文档...", "PROGRESS")

        # 确定输出路径
        if args.output:
            output_path = args.output
        else:
            output_name = DEFAULT_OUTPUT_NAME
            output_path = generate_output_filename(args.output_dir, output_name)

        try:
            final_path = save_docx("文献综述", review, output_path)
            log(f"Word 文档已生成", "OK")
            print(f"     📄 {os.path.abspath(final_path)}")
        except Exception as e:
            log(f"Word 生成失败: {e}", "ERROR")
            sys.exit(1)

    # ===== 完成 =====
    print("")
    print("=" * 60)
    print(f"  🎉 全部完成！")
    print(f"  成功分析: {success_count} 篇")
    if fail_count > 0:
        print(f"  跳过失败: {fail_count} 篇")
    print("=" * 60)
    print("")
    log("提示：输出文件为 .docx 格式，可用 Word/WPS 打开", "INFO")
    if fail_count > 0:
        log("提示：失败论文可能为扫描版PDF（纯图片），需要OCR处理", "INFO")
    print("")


if __name__ == "__main__":
    main()
