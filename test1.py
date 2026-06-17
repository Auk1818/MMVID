import numpy as np
from PIL import Image
import os

# ================== 配置路径 ==================
data_dir = r"C:\Users\lenovo\Desktop\mission\dataset\ruili34_nature"  # .npy 文件所在目录
output_dir = r"C:\Users\lenovo\Desktop\tupian"  # 输出图片文件夹

# 要处理的图像文件列表
image_files = [
    'list_all_image_1.npy',
    'list_all_image_2.npy',
    'list_all_image_3.npy'
]

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# ================== 处理每个 .npy 文件 ==================
for npy_file in image_files:
    file_path = os.path.join(data_dir, npy_file)

    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        continue

    print(f"📦 正在加载: {npy_file}")
    data = np.load(file_path)
    print(f"  数据原始形状: {data.shape}")  # 应该是 (200, 20, 224, 224, 3)

    if data.ndim != 5:
        raise ValueError(f"Expected 5D array (B, T, H, W, C), got {data.ndim}D")

    num_samples, num_frames_per_sample, height, width, channels = data.shape
    total_frames = num_samples * num_frames_per_sample

    print(f"  共 {num_samples} 个视频，每个 {num_frames_per_sample} 帧，总计 {total_frames} 张图像")

    # 提取前缀用于命名
    prefix = npy_file.replace('.npy', '')

    # 逐个样本、逐帧导出
    for sample_idx in range(num_samples):
        for frame_idx in range(num_frames_per_sample):
            img_array = data[sample_idx, frame_idx]  # shape: (224, 224, 3)

            # 确保是 uint8 类型
            if img_array.dtype != np.uint8:
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)

            # 转为 PIL 图像
            img = Image.fromarray(img_array)

            # 保存为 PNG，命名格式：list_all_image_1_sample000_frame00
            filename = f"{prefix}_sample{sample_idx:03d}_frame{frame_idx:02d}.png"
            img.save(os.path.join(output_dir, filename))

            # 打印进度（每 100 帧）
            current_frame = sample_idx * num_frames_per_sample + frame_idx
            if current_frame % 100 == 0 or current_frame == total_frames - 1:
                print(f"💾 已保存: {filename}")

print(f"\n🎉 所有图像已成功导出到:")
print(f"   {output_dir}")