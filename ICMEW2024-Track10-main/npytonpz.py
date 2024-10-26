import numpy as np

# 加载 test_A_joint.npy 文件
input_file = 'data/test_B_bone_motion.npy'
output_file = 'data/test_B_bone_motion.npz'

# 加载数据
data = np.load(input_file)

# 检查数据的形状
print(f"Loaded data shape: {data.shape}")

# 保存为 npz 格式
np.savez(output_file, x_test=data)

print(f"Data saved to {output_file}")

# 验证保存的文件
loaded_data = np.load(output_file)
print(f"Keys in the saved file: {loaded_data.files}")
print(f"Shape of x_test in the saved file: {loaded_data['x_test'].shape}")