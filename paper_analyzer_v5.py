import fitz
import glob
import re
import os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from openai import OpenAI
from datetime import datetime

# =========================
# 🔑 AI配置
# =========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")


def get_client():
    if not API_KEY:
        raise RuntimeError("❌ 未检测到 DEEPSEEK_API_KEY 环境变量，请先设置 DeepSeek API Key")
    return OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# =========================
# 📄 读取PDF
# =========================
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# =========================
# 🤖 单篇分析
# =========================
def analyze_paper(text):
    prompt = f"""
请结构化分析论文：

输出：
- 研究背景
- 方法
- 创新点
- 结论

论文内容：
{text}
"""

    res = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是科研分析助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return res.choices[0].message.content


# =========================
# 🧠 生成综述
# =========================
def generate_review(all_text):
    prompt = f"""
请根据以下论文分析，写一篇文献综述：

要求：
1. 研究背景
2. 方法分类
3. 对比分析
4. 研究不足
5. 未来方向

内容：
{all_text}
"""

    res = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是科研综述专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return res.choices[0].message.content


# =========================
# 🧼 Word清洗+美化输出（保留蓝色加粗）
# =========================
def save_docx(title, content):

    doc = Document()

    # ===== 字体统一 =====
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    # ===== 标题 =====
    h = doc.add_heading(title, level=1)
    h.alignment = 1

    doc.add_paragraph(f"生成时间：{datetime.now()}")
    doc.add_paragraph("")

    # ===== 内容处理 =====
    lines = content.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 去除Markdown标题标记（###、##），保留文本用于标题判断
        clean_for_check = line.replace("###", "").replace("##", "").strip()

        # ===== 标题判断 =====
        heading_keywords = [
            "研究背景", "背景", "方法", "方法分类", "创新点", "创新",
            "结论", "总结", "对比分析", "研究不足", "未来方向",
            "核心结果", "局限性", "应用方向", "通俗解释",
            "论文一句话总结", "研究方法", "核心结果"
        ]
        is_heading = False
        for kw in heading_keywords:
            # 如果行开头或行内容匹配关键词（且行较短，像标题）
            if clean_for_check.startswith(kw) or clean_for_check == kw:
                is_heading = True
                break
            # 也匹配 "**研究背景**" 这种带加粗的标题
            if ("**" + kw + "**") in line:
                is_heading = True
                break

        if is_heading:
            heading_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_for_check)
            # 去编号
            heading_text = re.sub(r"^\d+\.\s*", "", heading_text)
            doc.add_heading(heading_text.strip(), level=2)
            continue

        # ===== 处理含 **加粗** 的段落（蓝色加粗） =====
        if "**" in line:
            p = doc.add_paragraph()
            # 按 **...** 分割，交替处理
            parts = re.split(r"(\*\*[^*]+\*\*)", line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    bold_text = part[2:-2]
                    run = p.add_run(bold_text)
                    run.bold = True
                    run.font.color.rgb = RGBColor(0, 51, 153)  # 深蓝色加粗
                else:
                    if part:
                        p.add_run(part)
        else:
            # ===== 普通段落，去编号 =====
            clean_line = re.sub(r"^\d+\.\s*", "", line)
            doc.add_paragraph(clean_line)

    output_file = "文献综述_清洗版.docx"
    doc.save(output_file)

    print("✅ Word已生成:", output_file)


# =========================
# 🚀 主程序
# =========================
if __name__ == "__main__":

    if not API_KEY:
        print("错误：未检测到 DEEPSEEK_API_KEY 环境变量，请先设置 DeepSeek API Key")
        raise SystemExit(1)

    print("=" * 60)
    print("🧬 Paper Analyzer v5.2 清洗+Word美化版")
    print("=" * 60)

    pdf_files = glob.glob("paper/*.pdf")

    if not pdf_files:
        print("❌ paper文件夹没有PDF")
        exit()

    all_text = ""

    for pdf in pdf_files:
        print(f"\n📄 分析: {pdf}")

        text = read_pdf(pdf)
        result = analyze_paper(text)

        all_text += f"\n\n=== {pdf} ===\n{result}\n"

    print("\n🧠 正在生成文献综述...")

    review = generate_review(all_text)

    print("📝 正在生成Word文件...")

    save_docx("文献综述", review)

    print("\n🎉 完成！")
