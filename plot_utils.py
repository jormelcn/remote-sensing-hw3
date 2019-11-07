import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import matplotlib.image as image
from scipy.spatial.distance import cdist

def equi_dist(N, dims, alpha = 0.1, tol=1e-5):
    X = np.random.rand(N, dims)
    grad = np.ones(X.shape)
    while np.max(grad) > tol:
        A = cdist(X,X, metric='euclidean')
        for i in range(N):
            grad[i,:] = (8/N**2)*(A[i:i+1,:] - 1)@(X[i,:] - X)
        X -= alpha*N*grad
    return X

def generate_colors(N):
    vertex = equi_dist(N, 3)
    _min, _max = np.min(vertex), np.max(vertex)
    return (255*((vertex - _min)/(_max - _min))).astype(np.uint8)

def save_image(img, name, folder='./graphics', cmap=None):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    image.imsave(folder+'/'+ name + '.png', img, cmap=cmap)

def color_abundance_map(abundance_map, color):
    out = np.empty((abundance_map.shape[0], abundance_map.shape[1], 3), dtype=np.uint8)
    out[:,:,0] = abundance_map*color[0]
    out[:,:,1] = abundance_map*color[1]
    out[:,:,2] = abundance_map*color[2]
    return out

def color_true_map(img_labels, back_color=None, labels_colors=None, color_seed=None):
    nan_index = np.isnan(img_labels)
    labels = range(len(np.bincount(img_labels[np.logical_not(nan_index)])))
    labels_colors = generate_colors(len(labels)) if labels_colors is None else labels_colors
    if not back_color is None:
        labels_colors[0,:] = back_color
    img = np.empty((img_labels.shape[0], img_labels.shape[1], 3), dtype=np.uint8)
    for i in range(len(labels)):
        index = img_labels == labels[i]
        img[index,:] = labels_colors[i]
    img[nan_index,:] = [0, 0, 0]
    return img

def plot_true_map(title, img_labels, folder='./graphics', labels_names=None, labels_colors=None, color_seed=None):
    img = color_true_map(img_labels, labels_colors, color_seed)
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis('off')
    if not (labels_names is None):
        patches = [mpatches.Patch(color = labels_colors[i], label=labels_names[i]) for i in range(len(labels_names))]
        plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+'/'+ title + '.png')
    plt.close()

def plot_img(title, img, folder='./graphics', cmap=None):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+'/'+ title + '.png')
    plt.close()

def plot_hist(title, values, folder='./graphics'):
    plt.figure()
    hist, bin_edges = np.histogram(values, bins=int(1.7**np.log(len(values))))
    plt.plot((bin_edges[0:-1] + bin_edges[1:])/2, hist)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+'/'+ title + '.png')
    plt.close()
    return hist, bin_edges

def plot_error(title, start, errors, xlabel, ylabel, elbow=None, folder='./graphics'):
    plt.figure()
    plt.plot(range(start, len(errors)+start), errors)
    if not (elbow is None):
        plt.plot([elbow, elbow], [0, np.max(errors)])
    plt.xticks(range(start, len(errors)+start))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+'/'+ title + '.png')
    plt.close()

def plot_endmembers(title, endmembers, folder='./graphics'):
    rows = endmembers.shape[0]//2 + endmembers.shape[0]%2
    _, axes = plt.subplots(rows, 2, sharex=True, sharey=True)
    for i in range(endmembers.shape[0]):   
        axes[i//2, i%2].plot(range(endmembers.shape[1]), endmembers[i,:])
        axes[i//2, i%2].grid()
    axes[rows-1, 0].set_xlabel("Banda")
    axes[rows-1, 1].set_xlabel("Banda")
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.savefig(folder+'/'+ title + '.png')
    plt.close()
