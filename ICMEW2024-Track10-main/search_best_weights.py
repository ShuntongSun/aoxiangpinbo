import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback
from sklearn.metrics import accuracy_score


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


def weights_to_simplex(weights):
    """将任意非负权重转换为和为1的权重"""
    weights = np.array(weights)
    weights = np.maximum(weights, 0)  # 确保所有权重非负
    sum_weights = np.sum(weights)
    if sum_weights > 0:
        return weights / sum_weights
    else:
        return np.ones_like(weights) / len(weights)

def objective(weights):
    # 确保权重和为1
    weights = weights_to_simplex(weights)
    confidences = ensemble_results([joint_confidences, bone_confidences, JM_confidences,BM_confidences], weights)
    predictions = np.argmax(confidences, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    return -accuracy  # 我们要最大化准确率，所以返回负值


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
    parser.add_argument('--optimize', action='store_true', help='Use Bayesian optimization to find best weights')
    parser.add_argument('--n-calls', type=int, default=50, help='Number of calls for Bayesian optimization')
    # parser.add_argument('--output-dir', default='./results',
    #                     help='Directory to save the ensemble results')

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'csv3' in arg.dataset:
        npz_data = np.load('/home/axzc/ICMEW2024-Track10-main/data/test_A_joint.npz')
        data = npz_data['x_test']  # 假设V3.npz中只有x_test，没有y_test
        npz_label = np.load('/home/axzc/ICMEW2024-Track10-main/data/test_A_label.npz')
        true_labels = npz_label['y_test']  # 加载真实标签
    else:
        raise NotImplementedError("Only csv3 dataset is supported for this script.")

    with open(os.path.join(arg.joint_dir, 'epoch53_test_score.pkl'), 'rb') as r1_file:
        r1_joint = list(pickle.load(r1_file).items())
    # 加载 bone 结果
    with open(os.path.join(arg.bone_dir, 'epoch56_test_score.pkl'), 'rb') as r1_file:
        r1_bone = list(pickle.load(r1_file).items())

    # 加载joint+motion 结果
    with open(os.path.join(arg.JM_dir, 'epoch57_test_score.pkl'), 'rb') as r1_file:
        r1_JM = list(pickle.load(r1_file).items())

    #加载bonemotion结果
    with open(os.path.join(arg.BM_dir, 'epoch55_test_score.pkl'), 'rb') as r1_file:
        r1_BM = list(pickle.load(r1_file).items())
    # 进行预测
    joint_confidences = predict_with_confidence(r1_joint)
    bone_confidences = predict_with_confidence(r1_bone)
    JM_confidences = predict_with_confidence(r1_JM)
    BM_confidences = predict_with_confidence(r1_BM)

    if arg.optimize:
        # 定义搜索空间为3维
        space = [Real(0, 1, name=f'weight_{i}') for i in range(4)]

        @use_named_args(space)
        def objective_wrapper(**params):
            weights = list(params.values())
            return objective(weights)

        def print_results(res):
            current_weights = weights_to_simplex(res.x_iters[-1])
            current_accuracy = -res.func_vals[-1]  # 将负值转换回准确率
            print(f"Iteration: {len(res.x_iters)}")
            print(f"Current weights: {current_weights}")
            print(f"Current accuracy: {current_accuracy:.4f}")
            print("-----------------------------")

        res = gp_minimize(
            objective_wrapper,
            space,
            n_calls=arg.n_calls,
            random_state=0,
            callback=[VerboseCallback(n_total=arg.n_calls), print_results]
        )

        best_weights = weights_to_simplex(res.x)
        best_accuracy = -res.fun  # 将最小化的目标函数值转换回准确率
        print(f"\nOptimization completed.")
        print(f"Best weights found: {best_weights}")
        print(f"Best accuracy: {best_accuracy:.4f}")
    else:
        best_weights = weights_to_simplex(arg.weights) if arg.weights else None

    # 使用最佳权重或提供的权重进行集成
    confidences = ensemble_results([joint_confidences, bone_confidences, JM_confidences,BM_confidences], best_weights)
    output_file = f'Joint-Bone-Motion-mix_{arg.dataset}.npy'
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

    # 计算并打印最终的准确率
    final_accuracy = accuracy_score(true_labels, predictions)
    print(f"Final accuracy: {final_accuracy:.4f}")
