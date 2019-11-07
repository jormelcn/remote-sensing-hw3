import utils
from plot_utils import color_true_map, save_image
import numpy as np
from skimage import filters
from skimage.measure import label
from fcmeans import FCM
from skimage.morphology import binary_opening, binary_closing

def segmentation(img, n_clusters, sigma=0.3):
    fcm = FCM(n_clusters=n_clusters, max_iter=10000, m=2)
    fcm.fit(utils.flat(img))
    abun = fcm.u.reshape((img.shape[0], img.shape[1], n_clusters))
    masks = np.empty(abun.shape, dtype=bool)
    for i in range(n_clusters):
        thresh = filters.threshold_otsu(abun[:,:,i])
        filters.gaussian(abun[:,:,i],sigma=sigma, output=abun[:,:,i])
        masks[:,:,i] = abun[:,:,i] > thresh
    masks[masks.sum(axis=2) > 1,:] = 0
    label_imgs = [np.zeros(img.shape, dtype=np.uint8)]
    for i in range(n_clusters):
        binary_opening(masks[:,:,i], out=masks[:,:,i])
        label_img = label(masks[:,:,i])
        label_img[label_img > 0] += np.max(label_imgs[-1])
        label_imgs.append(label_img)
    return np.dstack(label_imgs).sum(axis=2)

def process_dataset(dataset, n_clusters, pca_enabled=False):
    X = np.load("./data/" + dataset + '.npy')
    if pca_enabled:
        pca = utils.load_model(dataset + '_pca')
        X = pca.transform(utils.flat(X)).reshape(X.shape[0], X.shape[1], -1)
    segments = segmentation(X, n_clusters)
    np.save("./results/" + dataset + "_segments.npy", segments)
    color_segments = color_true_map(segments, back_color=[1,1,1])
    save_image(color_segments, dataset + "_segments")
    print("Segments:", len(np.bincount(segments.reshape(-1)))-1)

process_dataset("indian_pines", 6, pca_enabled=True)
process_dataset("pavia_university", 6, pca_enabled=True)
process_dataset("cupride", 6, pca_enabled=True)
process_dataset("barr_s2", 6, pca_enabled=False)
process_dataset("barr_hy", 4, pca_enabled=True)
process_dataset("guav_l8", 4, pca_enabled=False)
