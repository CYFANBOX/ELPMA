import numpy as np
import scipy.io as scio


def import_dataset(data_path):
    print('\nImporting dataset ...\n')
    data = scio.loadmat(data_path + 'paper_similarity.mat')
    pse = np.array(data['pse_fusion'])
    miRNA = np.array(data['miRNA_fusion'])
    interaction = np.array(data['interaction'])
    pse_name = data['B_id']
    miRNA_name = data['A_id']
    unknown = []
    for x in range(interaction.shape[0]):
        for y in range(interaction.shape[1]):
            if interaction[x, y] == 0:
                unknown.append((x, y))
    return unknown, pse, miRNA, pse_name, miRNA_name


def base_preds_probs(X_test, trained_clfs):
    prob_list = []
    pred_list = []
    X_test = np.array(X_test)
    for clf in trained_clfs:
        pred = clf.predict(X_test)
        pred_list.append(pred)
        prob = clf.predict_proba(X_test)
        prob_list.append(prob[:, 1])
    pred_list = np.array(pred_list)
    prob_list = np.array(prob_list)
    base_preds = []
    base_probs = []
    for i in range(len(pred_list[0])):
        base_preds.append(pred_list[:, i])
    for i in range(len(prob_list[0])):
        base_probs.append(prob_list[:, i])
    return base_preds, base_probs


def soft_voting(base_probs):
    pred_final = []
    prob_final = []
    for prob in base_probs:
        mean_prob = np.mean(prob)
        prob_final.append(mean_prob)
        if mean_prob > 0.5:
            pred_final.append(1)
        else:
            pred_final.append(0)
    return pred_final, prob_final
