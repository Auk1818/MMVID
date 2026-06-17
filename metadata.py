# metadata.py 或直接写在生成脚本中
import numpy as np

# 按 ID 顺序（0~33）填写每个人的身高、体重、BMI
# 注意：原始表格中 ID 是从 1 开始的，我们要转成 0~33
heights = [
    179, 175, 167, 165, 176, 176, 158, 152, 187, 169,
    178, 162, 178, 170, 163, 173, 171, 182, 168, 166,
    182, 183, 160, 164, 177, 172, 184, 180, 163, 165,
    183, 174, 187, 160
]  # 单位：cm

weights = [
    63.8, 70, 58.5, 56.5, 56, 81, 51.5, 41, 75, 60,
    70, 51, 69.5, 87.5, 51, 63, 62, 75, 47.5, 56,
    70, 83, 45, 75, 65.5, 53, 75, 71, 50, 68,
    77.5, 60.5, 74, 45
]  # 单位：kg

bmis = [
    19.9, 22.9, 21.0, 20.8, 18.1, 26.1, 20.6, 17.7, 21.4, 21.0,
    22.1, 19.4, 21.9, 30.3, 19.2, 21.0, 21.2, 22.6, 16.8, 20.3,
    21.1, 24.8, 17.6, 27.9, 20.9, 17.9, 22.2, 21.9, 18.8, 25.0,
    23.1, 20.0, 21.2, 17.6
]
# 扩展成 (680,) 的一维数组
num_repeats = 20  # 每人20组动作

heights_680 = np.repeat(heights, num_repeats)  # shape: (680,)
weights_680 = np.repeat(weights, num_repeats)
bmi_680 = np.repeat(bmis, num_repeats)

print("heights_680.shape:", heights_680.shape)  # 应该是 (680,)
print("Sample: first 25 heights:", heights_680[:25])
# 保存到指定目录
save_dir = "C:/Users/lenovo/Desktop/mission/dataset/ruili34_nature"

np.save(f"{save_dir}/list_all_height_1.npy", heights_680)
np.save(f"{save_dir}/list_all_weight_1.npy", weights_680)
np.save(f"{save_dir}/list_all_bmi_1.npy", bmi_680)

print("✅ .npy files saved successfully!")