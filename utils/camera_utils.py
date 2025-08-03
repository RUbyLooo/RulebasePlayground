import numpy as np
# -------------------相机内参计算-------------------
fovy = 58  # 单位：度
height = 480
width = 640

# 计算焦距（以像素为单位）
f = height / (2 * np.tan(np.radians(fovy) / 2))

# 主点（图像中心）
cx = (width - 1)/ 2
cy = (height - 1)/ 2

# 内参矩阵 K
K = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
])

print("相机内参矩阵（焦距以像素为单位）:\n",K)