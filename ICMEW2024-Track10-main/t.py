import numpy as np

# file_path = 'B-MIX_csv3.npy'
file_path = '/home/axzc/ICMEW2024-Track10-main/Model_inference/Mix_GCN/dataset/save_2d_pose/train_bone.npz'
data = np.load(file_path)

# 打印文件中的所有键
print(f"Keys in the data file: {data.files}")

# 假设你知道正确的键名，例如 'y_test'
key_name = 'y_test'  # 根据实际情况修改

if key_name in data:
    array_data = data[key_name]

    print(f"The shape of the data in '{key_name}' is: {array_data.shape}")
    print(f"Data type: {array_data.dtype}")
    print(f"Total number of elements: {array_data.size}")
    print(f"Dimensions: {array_data.ndim}")

    # 打印前几行数据
    print("\nFirst few predictions:")
    print(array_data[:5])

    # 打印一些基本统计信息
    print("\nStatistics:")
    print(f"Min values: {array_data.min(axis=0)}")
    print(f"Max values: {array_data.max(axis=0)}")
    print(f"Mean values: {array_data.mean(axis=0)}")
else:
    print(f"Key '{key_name}' not found in the data file.")