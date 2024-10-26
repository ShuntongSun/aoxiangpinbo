import numpy as np
import os
import argparse

def main(dataset_path: str) -> None:
    print("加载数据集...")

    # 加载训练集
    train_joint_path = os.path.join(dataset_path, 'data/train_joint_motion.npy')
    train_label_path = os.path.join(dataset_path, 'data/train_label.npy')
    train_joint = np.load(train_joint_path)
    train_label = np.load(train_label_path)

    print(f"训练集关节数据形状: {train_joint.shape}")
    print(f"训练集标签形状: {train_label.shape}")

    # 加载测试集
    test_joint_path = os.path.join(dataset_path, 'data/test_A_joint_motion.npy')
    test_label_path = os.path.join(dataset_path, 'data/test_A_label.npy')
    test_joint = np.load(test_joint_path)
    test_label = np.load(test_label_path)

    print(f"测试集关节数据形状: {test_joint.shape}")
    print(f"测试集标签数据形状: {test_label.shape}")

    # 创建保存目录
    save_dir = '../Model_inference/Mix_GCN/dataset/save_2d_pose'
    os.makedirs(save_dir, exist_ok=True)

    # 保存处理后的数据

    np.savez(os.path.join(save_dir, 'train_jM.npz'), x_train=train_joint, y_train=train_label)
    np.savez(os.path.join(save_dir, 'test_jM.npz'), x_test=test_joint, y_test=test_label)

    print("数据处理完成！")

def get_parser():
    parser = argparse.ArgumentParser(description='处理2D姿势数据集')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../',
        help='包含train_joint.npy, train_label.npy和test_joint.npy的目录路径')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.dataset_path)