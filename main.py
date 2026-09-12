print("AI生物助手启动")
print("=" * 30)

while True:
    # 读取用户输入
    user_input = input("请输入您的问题（输入 '退出' 结束对话）：")
    
    # 判断是否退出
    if user_input == "退出":
        print("AI生物助手已关闭，再见！")
        break
    
    # 简单回复用户
    print(f"您的问题是：{user_input}")
    print("AI生物助手正在思考中...\n")
    print("-" * 30)
