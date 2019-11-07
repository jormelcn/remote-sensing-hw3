# pylint:disable=unsubscriptable-object
import numpy as np
import joblib
import json
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import confusion_matrix as _confusion_matrix
from VCA import vca as _vca

def flat(img):
    if len(img.shape) == 3:
        return img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
    elif len(img.shape) == 2:
        return img.reshape((img.shape[0]*img.shape[1],))

def random_mask_split(mask, N):
    index = np.array(range(len(mask)), dtype=int)
    available_index = index[mask]
    np.random.shuffle(available_index)
    mask_B = mask.copy()
    mask_B[available_index[0:N]] = False
    mask_A = np.logical_and(mask, np.logical_not(mask_B)) 
    return mask_A, mask_B

def balanced_train_test_mask(y, y_mask, test_ratio):
    N = y[y_mask].shape[0]
    labels = np.unique(y[y_mask])
    mask = np.logical_and(y_mask, (np.expand_dims(y,1) == labels).T).T
    count = np.sum(mask, axis=0)
    s_labels = np.argsort(count)
    test_N,  c_test_N = int(N*test_ratio), 0
    train_N, c_train_N = N - test_N, 0
    train_mask, test_mask = np.zeros(y.shape, dtype=bool), np.zeros(y.shape, dtype=bool)
    for i in range(len(labels)):
        j = s_labels[i]
        if i == len(labels) - 1:
            train_n = train_N - c_train_N
        else:    
            train_n = count[j]-int(count[j]*test_ratio)
        test_n = count[j] - train_n
        label_train_mask, label_test_mask = random_mask_split(mask[:,j], train_n)    
        np.logical_or(train_mask, label_train_mask, out=train_mask)
        np.logical_or(test_mask, label_test_mask, out=test_mask)
        c_train_N += train_n
        c_test_N += test_n
    return train_mask, test_mask

def balanced_score(y, y_pred):
    labels = np.unique(y)
    index = (np.expand_dims(y,1) == labels)
    comp = y == y_pred
    labels_scores = [comp[index[:,i]].mean() for i in range(len(labels))]
    return np.mean(labels_scores)

def pca(X, variance):
    pre_model = PCA(X.shape[1], copy=True)
    pre_model.fit(X)
    cum_var_ratio =  np.cumsum(pre_model.explained_variance_ratio_)
    n_components = np.nonzero(cum_var_ratio > variance)[0][0] + 1
    model = PCA(n_components, copy=True)
    model.fit(X)
    return model

def svc(X, y, C, gamma):
    model = SVC(C=C, kernel='rbf', gamma=gamma, decision_function_shape='ovo')
    model.fit(X, y)
    return model

def load_model(name, folder='./models'):
    return joblib.load(folder + '/' + name + '.joblib')

def save_model(model, name, folder='./models'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    joblib.dump(model, folder + '/' + name + '.joblib')

def save_csv(data, name, folder='./results'):
    df = pd.DataFrame(data=data)
    df.to_csv(folder + "/" + name + '.csv', sep = ';')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + "/" + name + '.csv', 'r') as f:
        text = f.read()
    text = text.replace('.', ',')
    with open(folder + "/" + name + '.csv', 'w') as f:
        f.write(text)

def confusion_matrix(y, y_pred):
    cm = _confusion_matrix(y, y_pred)
    cm = (cm.T/cm.sum(axis=1)).T
    return cm

def save_json(_dict, name, folder='./results'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + '/' + name + '.json', 'w') as f:
        f.write(json.dumps(_dict))

def max_scale(X):
    _min, _max = np.min(X), np.max(X)
    return (X - _min)/(_max - _min)

def elbow(_y_):
    y = np.array(_y_)
    y = y*y.shape[0]/np.max(y)
    start = np.argmax(_y_)
    d_ang = np.empty(len(y)-start-2)
    for i in range(start + 1, len(y)-2):
        ang1 = np.arctan2(y[i] - y[start], i)
        ang2 = np.arctan2(y[-1] - y[i], len(y)-1 - i)
        d_ang[i-1-start] = abs(ang1 - ang2)
    return np.argmax(d_ang) + 1 + start

def vca(X, n_endmembers):
    enmb, _, _ = _vca(X.T, n_endmembers)
    return enmb.T

def mse(a, b, buff=None, axis=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    np.square(buff, out=buff)
    return buff.mean(axis=axis)

def grad_mse(a,b, buff=None):
    buff = np.empty(a.shape) if buff is None else buff
    np.subtract(a,b, out=buff)
    return buff

def unmixing(X, endmembers, alpha=0.001, error=mse, grad_error=grad_mse, C=10, max_iter=100000, tol=1e-5, sum_1=False):
    R, S = X.T, endmembers.T
    a = np.random.rand(S.shape[1], R.shape[1])/S.shape[1]
    grad_a = np.empty(a.shape)
    grad_e = np.empty(R.shape)
    buff_e = np.empty(R.shape)
    buff_a = np.empty(a.shape)
    buff_a2 = np.empty(a.shape)
    ones = np.ones(a.shape)
    buff_aS = np.empty(a.shape[1])
    _R_ = np.empty(R.shape)
    history = np.empty((max_iter, ))
    for i in range(max_iter):
        np.dot(S,a, out=_R_)
        history[i] = error(_R_, R, buff_e)
        grad_error(_R_, R, grad_e)
        np.dot(S.T, grad_e, out=grad_a)
        
        np.sum(a, axis=0, out=buff_aS)
        np.subtract(buff_aS, ones, out=buff_a)
        if not sum_1:
            np.greater(buff_aS, ones, out=buff_a2)
            np.multiply(buff_a2, buff_a, out=buff_a)
        np.multiply(C, buff_a, out=buff_a)
        np.add(grad_a, buff_a, out=grad_a)
        np.multiply(alpha, grad_a, out=grad_a)
        # Update a
        np.subtract(a, grad_a, out=a)
        mag_grad_a = np.max(np.abs(grad_a))
        print("grad-a: {:.15f}".format(mag_grad_a), end='\r')
        if mag_grad_a < tol:
            return a.T
    print("Warning: Not convergence")
    return a.T
