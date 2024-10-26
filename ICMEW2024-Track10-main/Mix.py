import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

# from Process_data.tools.convert_insta import motion


def ensemble_results(confidences_list, weights=None):
    """
    对多个模型的置信度进行加权集成
    :param confidences_list: 包含多个模型置信度的列表
    :param weights: 每个模型对应的权重，如果为None则使用平均权重
    :return: 集成后的置信度
    """
    if weights is None:
        weights = [1 / len(confidences_list)] * len(confidences_list)

    assert len(confidences_list) == len(weights), "模型数量和权重数量必须相同"
    assert np.isclose(sum(weights), 1), "权重之和必须为1"

    return sum(conf * w for conf, w in zip(confidences_list, weights))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_with_confidence(r1):
    all_confidences = []
    for i in tqdm(range(len(r1))):
        _, r11 = r1[i]
        confidences = softmax(r11)
        all_confidences.append(confidences)
    return np.array(all_confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'csv1', 'csv2',
                                 'csv3'},
                        help='the work folder for storing results')
    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--JM-dir',
                        help='Directory containing "epoch1_test_score.pkl" for JM eval results')
    parser.add_argument('--BM-dir',
                        help='Directory containing "epoch1_test_score.pkl" for BM eval results')
    parser.add_argument('--weights', nargs='+', type=float,
                        help='Weights for each model (must sum to 1)')
    # parser.add_argument('--output-dir', default='./results',
    #                     help='Directory to save the ensemble results')

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'csv3' in arg.dataset:
        npz_data = np.load('/home/axzc/ICMEW2024-Track10-main/data/test_joint_B.npz')
        data = npz_data['x_test']  # 假设V3.npz中只有x_test，没有y_test
    else:
        raise NotImplementedError("Only csv3 dataset is supported for this script.")

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1_file:
        r1_joint = list(pickle.load(r1_file).items())
    # 加载 bone 结果
    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r1_file:
        r1_bone = list(pickle.load(r1_file).items())

    # 加载joint+motion 结果
    with open(os.path.join(arg.JM_dir, 'epoch1_test_score.pkl'), 'rb') as r1_file:
        r1_JM = list(pickle.load(r1_file).items())
        # 加载bonemotion结果
    with open(os.path.join(arg.BM_dir, 'epoch1_test_score.pkl'), 'rb') as r1_file:
        r1_BM = list(pickle.load(r1_file).items())
        # 进行预测
    joint_confidences = predict_with_confidence(r1_joint)
    bone_confidences = predict_with_confidence(r1_bone)
    JM_confidences = predict_with_confidence(r1_JM)
    BM_confidences = predict_with_confidence(r1_BM)

    confidences = ensemble_results([joint_confidences, bone_confidences, JM_confidences, BM_confidences], arg.weights)
    # 保存为.npy文件
    output_file = f'B-MIX_{arg.dataset}.npy'
    np.save(output_file, confidences)
    print(f"Confidences saved to {output_file}")

    # 打印一些统计信息
    print(f"Shape of confidences: {confidences.shape}")
    print(f"Min confidence: {confidences.min():.4f}")
    print(f"Max confidence: {confidences.max():.4f}")
    print(f"Mean confidence: {confidences.mean():.4f}")

    # 获取预测类别和最高置信度
    predictions = np.argmax(confidences, axis=1)
    max_confidences = np.max(confidences, axis=1)

    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Average max confidence: {np.mean(max_confidences):.4f}")
    print(f"Max confidence range: {np.min(max_confidences):.4f} - {np.max(max_confidences):.4f}")