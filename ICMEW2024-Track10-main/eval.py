import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='/home/axzc/ICMEW2024-Track10-main/Joint-Bone-Motion-mix_csv3.npy')

if __name__ == "__main__":

    args = parser.parse_args()

    # load label and pred
    label =np.load('data/test_A_label.npy')

    pred = np.load(args.pred_path).argmax(axis=1)

    correct = (pred == label).sum()

    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))