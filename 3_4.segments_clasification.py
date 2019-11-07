import utils
from plot_utils import color_true_map, save_image
from constants import indian_pines_colors, pavia_university_colors
from constants import barr_s2_colors, barr_hy_colors, guav_l8_colors
import numpy as np

def classify_segments(cmap, segments):
    out = np.empty(cmap.shape, dtype=np.uint8)
    seg_labels = range(1, len(np.bincount(segments.reshape(-1))))
    back_mask = segments == 0
    out[back_mask] = cmap[back_mask]
    for seg in seg_labels:
        seg_mask = segments == seg
        out[seg_mask] = np.argmax(np.bincount(cmap[seg_mask]))
    return out

def process_dataset(dataset, colors):
    y = np.load("./data/" + dataset + '_labels.npy')
    pred = np.load("./data/" + dataset + '_clasification.npy')
    segments = np.load("./results/" + dataset + '_segments.npy')
    test_mask = np.load("./data/" + dataset + '_test_mask.npy').reshape(y.shape)
    sc_pred = classify_segments(pred, segments)
    sc_score = utils.balanced_score(y[test_mask], sc_pred[test_mask]) 
    sc_cm = utils.confusion_matrix(y[test_mask], sc_pred[test_mask])
    utils.save_json({"sc": sc_score} , dataset + "_sc_score")
    utils.save_csv(sc_cm, dataset + "_sc_cm")
    color_map = color_true_map(sc_pred, labels_colors=colors)
    save_image(color_map, dataset + "_sc_clasification")

process_dataset("indian_pines", indian_pines_colors)
process_dataset("pavia_university", pavia_university_colors)
process_dataset("barr_s2", barr_s2_colors)
process_dataset("barr_hy", barr_hy_colors)
process_dataset("guav_l8", guav_l8_colors)
