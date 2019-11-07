import utils
import numpy as np
from plot_utils import save_image, color_true_map
from constants import indian_pines_colors, pavia_university_colors
from constants import barr_s2_colors, barr_hy_colors, guav_l8_colors
from constants import cupride_colors

def generate_work_data(dataset, labels, colors, parameters, pca_enabled=False):
    X_img = np.load('./data/' + dataset + '.npy')
    y_img = np.load('./data/' + dataset + '_labels.npy')    
    save_image(color_true_map(y_img, labels_colors=colors), dataset + "_labels")
    
    X = utils.flat(X_img)
    y = utils.flat(y_img)
    train_ratio, val_ratio = 0.1, 0.1
    test_ratio = 1 - (train_ratio + val_ratio)
    tv_mask, test_mask = utils.balanced_train_test_mask(y, np.isin(y, labels), test_ratio)
    train_mask, val_mask = utils.balanced_train_test_mask(y, tv_mask, val_ratio/(val_ratio+train_ratio))

    np.save("./data/" + dataset + "_train_mask.npy", train_mask)
    np.save("./data/" + dataset + "_val_mask.npy", val_mask)
    np.save("./data/" + dataset + "_test_mask.npy", test_mask)

    if pca_enabled:
        pca = utils.pca(X[tv_mask,:], 0.99)
        utils.save_model(pca, dataset + '_pca')
        train = pca.transform(X[train_mask,:])
        test = pca.transform(X[test_mask])
        flat = pca.transform(X)
    else:
        train = X[train_mask,:]
        test = X[test_mask,:]
        flat = X

    svc = utils.svc(train, y[train_mask], parameters["C"], parameters["gamma"])
    utils.save_model(svc, dataset + '_svc')
    test_pred = svc.predict(test)
    np.save("./data/" + dataset + "_test_pred.npy", test_pred)
    classification = svc.predict(flat).reshape(y_img.shape).astype(np.uint8)
    np.save("./data/" + dataset + "_clasification.npy", classification)
    save_image(color_true_map(classification, labels_colors=colors), dataset + "_clasification")

    score = utils.balanced_score(y[test_mask], test_pred)
    utils.save_json({"original": score}, dataset + "_original_score")
    print("Test Score:", score)

exit()  # For prevent accidents

generate_work_data(
    "indian_pines", 
    np.array(range(17)), 
    indian_pines_colors,
    {"C": 50, "gamma": 0.00000025}, 
    pca_enabled=True)

generate_work_data(
    "pavia_university", 
    np.array(range(10)),
    pavia_university_colors, 
    {"C": 100, "gamma": 0.0000038}, 
    pca_enabled=True)

generate_work_data(
    "barr_s2", 
    np.array(range(1,9)), 
    barr_s2_colors,
    {"C": 5, "gamma": 0.0000016}, 
    pca_enabled=False)

generate_work_data(
    "barr_hy", 
    np.array(range(1,8)), 
    barr_hy_colors,
    {"C": 5, "gamma": 0.0000001}, 
    pca_enabled=True)

generate_work_data(
    "guav_l8", 
    np.array(range(1,5)), 
    guav_l8_colors,
    {"C": 1, "gamma": 0.0000004}, 
    pca_enabled=False)

generate_work_data(
    "cupride", 
    np.array(range(11)), 
    cupride_colors,
    {"C": 1, "gamma": 0.000001}, 
    pca_enabled=True)
