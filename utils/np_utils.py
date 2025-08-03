import numpy as np

# path = np.array([[1, 2, 3], [4, 5, 6]])
# new_row = np.array([7, 8, 9])
# path = np.vstack([path, new_row])
# print(path)
# # 输出：
# # [[1 2 3]
# #  [4 5 6]
# #  [7 8 9]]

#
path = np.array([[1, 2, 3]])
new_path = np.array([[7, 8, 9], [4, 5, 6]])  # 使用二维列表
combined_path = np.vstack([path, new_path])

print(combined_path)
