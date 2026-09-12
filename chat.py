import os
from openai import OpenAI

# 1. 初始化 DeepSeek 客户端
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not API_KEY:
    print("错误：未检测到 DEEPSEEK_API_KEY 环境变量，请先设置 DeepSeek API Key。")
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")

print("🚀 正在连接 DeepSeek V4 Pro 搞钱大模型...")

# 2. 你的问题（在这里修改你想对 AI 说的话）
user_question = "我是生物背景的。客户发给我一个表格，要求我把 A 列里含有 'Ab' 的行全部挑出来求平均值，请帮我写出 Python 代码。"

try:
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": "你是一个精通生物医药数据清洗和Python编程的专家。",
            },
            {"role": "user", "content": user_question},
        ],
        stream=False,
    )

    # 3. 打印出 AI 吐回来的完整完美代码
    print("\n💡 [DeepSeek 的大师级解答]：")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\n❌ 发生错误：{e}")
    print("提示：请检查第7行的 API_KEY 是否复制完整，或者官网10元余额是否到账。")
