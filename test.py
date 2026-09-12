import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 模拟生成100条抗体筛选数据
np.random.seed(42)
clones = [f"Ab-{i}" for i in range(1, 101)]
affinity = np.random.uniform(1.5, 9.5, 100)  # 模拟亲和力数值
is_selected = affinity > 7.0  # 亲和力大于7的被选中

df = pd.DataFrame(
    {
        "Clone_ID": clones,
        "Target_Binding_Affinity": affinity,
        "Is_Selected": is_selected,
    }
)

# 2. 自动保存到当前文件夹
file_name = "antibody_data.xlsx"
df.to_excel(file_name, index=False)
print(f"✅ 成功！模拟的抗体数据已保存为：{os.path.abspath(file_name)}")

# 3. 用 Seaborn 画一张符合 SCI 论文审稿人审美的漂亮的直方图
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))

# 画柱状图，大于7的标为绿色，小于7的标为灰色
sns.histplot(
    data=df,
    x="Target_Binding_Affinity",
    hue="Is_Selected",
    palette={True: "#2ecc71", False: "#95a5a6"},
    multiple="stack",
    bins=15,
)

plt.title("Antibody Screening Results (Affinity Distribution)", fontsize=14)
plt.xlabel("Target Binding Affinity (Kd)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.axvline(
    x=7.0, color="#e74c3c", linestyle="--", label="Selection Threshold (7.0)"
)
plt.legend()

print("📊 正在打开科研级数据图表...")
plt.show()