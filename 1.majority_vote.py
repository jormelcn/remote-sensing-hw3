import utils
from plot_utils import color_true_map, save_image
from constants import indian_pines_colors, indian_pines_labels
from constants import pavia_university_colors, pavia_university_labels
import numpy as np

def majority_vote(img, size):
    out = img.copy()
    padding = (size -1)//2
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            ri, re = max(0, i-padding), min(i+padding+1, img.shape[0])
            ci, ce = max(0, j-padding), min(j+padding+1, img.shape[1])
            values = list(img[ri:re, ci:ce].reshape(-1))
            out[i,j] = max(map(lambda val: (values.count(val), val), set(values)))[1]
    return out

def process_dataset(dataset, colors):
    y = np.load("./data/" + dataset + '_labels.npy')
    pred = np.load("./data/" + dataset + '_clasification.npy')
    test_mask = np.load("./data/" + dataset + '_test_mask.npy').reshape(y.shape)
    mv_sizes = [3, 5, 9]
    mv_pred = [majority_vote(pred, size) for size in mv_sizes]
    mv_scores = [utils.balanced_score(y[test_mask], p[test_mask]) for p in mv_pred]
    mv_cm = [utils.confusion_matrix(y[test_mask], p[test_mask]) for p in mv_pred]
    keys = ["_mv_{}".format(size) for size in mv_sizes]
    utils.save_json(dict(zip(keys, mv_scores)) , dataset + "_mv_scores")
    for i in range(len(mv_sizes)):
        utils.save_csv(mv_cm[i], dataset + keys[i] + "_cm")
        color_map = color_true_map(mv_pred[i], labels_colors=colors)
        save_image(color_map, dataset + keys[i] + "_clasification")

process_dataset("indian_pines", indian_pines_colors)
process_dataset("pavia_university", pavia_university_colors)
