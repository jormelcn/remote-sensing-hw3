import utils
from plot_utils import plot_true_map, plot_img, plot_hist, save_image, plot_endmembers
from constants import indian_pines_colors, indian_pines_labels
from skimage.morphology import binary_opening, area_opening
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# ip_lab = np.load("./data/indian_pines_labels.npy")
# ip_map = np.load("./data/indian_pines_clasification.npy")
# ip_test_mask = np.load("./data/indian_pines_test_mask.npy")
# ip_map_mv = utils.majority_vote(ip_map, 3)

# org_test_score = utils.balanced_score(
#     utils.flat(ip_lab)[ip_test_mask],
#     utils.flat(ip_map)[ip_test_mask] )

# mv_test_score = utils.balanced_score(
#     utils.flat(ip_lab)[ip_test_mask],
#     utils.flat(ip_map_mv)[ip_test_mask] )

# print("Original Test Score:\t", org_test_score)
# print("Majory Vote Test Score:\t", mv_test_score)


# plot_true_map(
#     "Test Map IP_LAB", 
#     ip_lab, 
#     labels_names=indian_pines_labels,
#     labels_colors=indian_pines_colors)
# plot_true_map(
#     "Test Map IP_CLA", 
#     ip_map, 
#     labels_names=indian_pines_labels,
#     labels_colors=indian_pines_colors)
# plot_true_map(
#     "Test Map IP_MV", 
#     ip_map_mv, 
#     labels_names=indian_pines_labels,
#     labels_colors=indian_pines_colors)


# from fcmeans import FCM

# ip = np.load("./data/indian_pines.npy")
# ip_pca_model = utils.load_model("indian_pines_pca")
# ip_pca = ip_pca_model.transform(utils.flat(ip)).reshape((ip.shape[0], ip.shape[1], -1))

# l8 = np.load("./data/guav_l8.npy")
# s2 = np.load("./data/barr_s2.npy")

# segments = utils.segmentation(ip_pca, 5)
# segments = utils.segmentation2(l8, 8)
# n_segments = len(np.bincount(segments.reshape(-1)))
# print(segments.dtype)
# print(n_segments)

# plot_true_map(
#     "Test L8 fuzzy Segments color", 
#     segments 
# )

# plot_img(
#     "Test L8 fuzzy Segments", 
#     segments,
#     cmap='gray')

# s2 = np.load("./data/barr_s2.npy")

# s2_flat = utils.flat(s2)
# mean = s2_flat.mean(axis=0)[np.newaxis]
# mag = cdist(s2_flat, mean, metric='euclidean')/np.max(s2_flat)
# ang = cdist(s2_flat, mean, metric='cosine')
# corr= cdist(s2_flat, mean, metric='correlation')

# plot_hist("Test S2_Angle_Hist", mag + ang + corr)

# def process_dataset(dataset):
#     y = np.load("./data/" + dataset + '_labels.npy')
#     pred = np.load("./data/" + dataset + '_clasification.npy')
#     test_mask = np.load("./data/" + dataset + '_test_mask.npy').reshape(y.shape)
#     score = utils.balanced_score(y[test_mask], pred[test_mask])
#     utils.save_json({"original": score}, dataset + "_original_score")


# process_dataset("indian_pines")
# process_dataset("pavia_university")


# X=utils.equi_dist(3, 3)
# A = utils.cdist(X, X)

# dist = []
# for i in range(A.shape[0]):
#     for j in range(i+1, A.shape[1]):
#         dist.append(A[i, j])
# print("dispersión:", np.std(dist))

# rows, cols = 30, 30
# colors = utils.generate_colors(rows * cols)
# cdf = pd.DataFrame(data=colors, columns=["R", "G", "B"])
# cdf_sort = cdf.sort_values(by=["R", "G", "B"])
# colors_img = cdf_sort.values.reshape((rows, cols, 3))
# save_image(colors_img, "Test Colors")

endmembers = np.load("./results/barr_hy_seg_endmembers.npy")
plot_endmembers("barr_hy_seg_endmember", endmembers)

# errors = np.load("./error.npy")

# elbow = utils.elbow(errors)

# print("Elbow", elbow)
