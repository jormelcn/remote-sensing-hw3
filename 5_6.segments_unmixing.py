import utils
import numpy as np
from plot_utils import color_abundance_map, plot_error, save_image, plot_endmembers
import matplotlib.pyplot as plt

def unmix_segments(X_img, segments, seg_labels, n_endmembers):
    X = np.empty((len(seg_labels), X_img.shape[2]))
    for seg in seg_labels:
        X[seg,:] = X_img[segments == seg,:].mean(axis=0)
    X = X/2**16
    endmembers = utils.vca(X[1:,:], n_endmembers)
    abundances = utils.unmixing(X, endmembers)
    return X, endmembers, abundances

def error_curve(dataset, X_img, segments, seg_labels):
    errors = []
    for n_endmembers in range(2, 21):
        X, endmembers, abundances = unmix_segments(X_img, segments, seg_labels, n_endmembers)
        errors.append(((X - abundances@endmembers)**2).mean()**0.5)
    elbow = utils.elbow(errors) + 2
    plot_error(dataset + "_unmixing_error", 2, errors, "Numero de Endmembers", "RMSE", elbow = elbow)
    return elbow

def process_dataset(dataset):
    X_img = np.load("./data/" + dataset + '.npy')
    segments = np.load("./results/" + dataset + '_segments.npy')
    seg_labels = range(len(np.bincount(segments.reshape(-1))))
    n_endmembers = error_curve(dataset, X_img, segments, seg_labels)

    _, endmembers, abundances = unmix_segments(X_img, segments, seg_labels, n_endmembers)
    plot_endmembers(dataset + "_seg_endmember", endmembers)
    np.save("./results/" + dataset + "_seg_endmembers.npy", endmembers)
    np.save("./results/" + dataset + "_seg_abundances.npy", abundances)
    abun_map = np.empty((X_img.shape[0], X_img.shape[1], n_endmembers))
    for seg in seg_labels:
        abun_map[segments == seg,:] = abundances[seg,:]
    for i in range(n_endmembers):
        abun_color = color_abundance_map(abun_map[:,:,i], [255, 255, 255])
        save_image(abun_color, dataset + "_seg_abundances_" + str(i))

process_dataset("barr_hy")
process_dataset("cupride")

