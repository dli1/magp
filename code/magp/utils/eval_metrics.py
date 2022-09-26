#coding=utf-8

import numpy as np
from collections import OrderedDict
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, roc_auc_score
from gpflow.utilities import parameter_dict
from magp.utils.common import *


def prob_to_class(y, threshold):
    y = np.array(y).flatten()
    y = np.where(y >= threshold, RELEVANT, NON_RELEVANT)
    return y


def accprf(ref, pred):
    ref = np.array(ref, dtype=np.int)
    pred = np.array(pred, dtype=np.int)

    acc = accuracy_score(ref, pred)
    p = precision_score(ref, pred, pos_label=RELEVANT)
    r = recall_score(ref, pred, pos_label=RELEVANT)
    f1 = f1_score(ref, pred, pos_label=RELEVANT)

    return acc, p, r, f1


def taurho(ref, pred):
    tau, p_value = kendalltau(ref, pred)
    rho, p_value = pearsonr(ref, pred)
    return tau, rho


def task_difficulty(crowd_y, mask_y, groundtruth_y):

    assert crowd_y.shape[0] == groundtruth_y.shape[0]

    task_difficulty = []
    gold_y = groundtruth_y.flatten()
    for y, mask, in zip(crowd_y, mask_y):
        correct_num = sum([1 for gi, yi, mi in zip(gold_y, y, mask) if (mi == 1 and gi == yi)])
        anno_num = sum(mask)
        task_difficulty.append(1 - float(correct_num)/anno_num)
    task_difficulty = np.array(task_difficulty)
    return task_difficulty


def worker_competence(crowd_y, mask_y, groundtruth_y):

    assert crowd_y.shape[0] == groundtruth_y.shape[0]

    worker_competence_list = []
    gold_y = groundtruth_y.flatten()
    for y, mask, in zip(crowd_y.T, mask_y.T):
        correct_num = sum([1 for gi, yi, mi in zip(gold_y, y, mask) if (mi == 1 and gi == yi)])
        anno_num = sum(mask)
        worker_competence_list.append(float(correct_num) / anno_num)
    worker_competence_list = np.array(worker_competence_list)
    return worker_competence_list



def crowd_sourcing_data_statistics(crowd_y, mask_y, ref_y):

    assert crowd_y.shape[0] == ref_y.shape[0]

    task_num = crowd_y.shape[0]
    worker_num = crowd_y.shape[1]
    judgement_num = mask_y.sum()

    rel_num = sum([1 for item in ref_y if item == 1])
    nonrel_num = sum([1 for item in ref_y if item == 0])

    statis_dct = {'task_num': task_num,
                  'worker_num': worker_num,
                  'judgement_num': judgement_num,
                  'rel_num': rel_num,
                  'nonrel_num': nonrel_num,
                  }

    return statis_dct


def evaluate_label_for_gpla(csd_data, model, measure):
    mean, var = model.predict_y(csd_data['input_features'])
    pred_y = prob_to_class(mean.numpy().flatten(), 0.5)
    pred_y_score = mean.numpy().flatten()
    ref_y = csd_data['gold_labels'].flatten()

    # tn, fp, fn, tp
    if measure == 'tn':
        return confusion_matrix(ref_y, pred_y).ravel()[0]

    if measure == 'fp':
        return confusion_matrix(ref_y, pred_y).ravel()[1]

    if measure == 'fn':
        return confusion_matrix(ref_y, pred_y).ravel()[2]

    if measure == 'tp':
        return confusion_matrix(ref_y, pred_y).ravel()[3]

    if measure == 'acc':
        return accuracy_score(ref_y, pred_y)
    if measure == 'posi_f1':
        return f1_score(ref_y, pred_y, pos_label=RELEVANT)
    if measure == 'nega_f1':
        return f1_score(ref_y, pred_y, pos_label=NON_RELEVANT)
    if measure == 'auc':
        return roc_auc_score(ref_y, pred_y_score)


def evaluate_likelihood_param_for_gpla(csd_data, model, param_name):
    try:
        ref_value = csd_data[param_name].flatten()
        pred_value = parameter_dict(model.likelihood)['.'+param_name].numpy().flatten()
        mse = mean_squared_error(ref_value, pred_value)
        return mse
    except:
        return -1


def evaluate_label_for_la(csd_data, pred_data, measure):

    ref_y = csd_data['gold_labels'].flatten()
    pred_y = pred_data['pred_y'].flatten()
    pred_y_score = pred_data['pred_y_score'].flatten()

    # tn, fp, fn, tp
    if measure == 'tn':
        return confusion_matrix(ref_y, pred_y).ravel()[0]

    if measure == 'fp':
        return confusion_matrix(ref_y, pred_y).ravel()[1]

    if measure == 'fn':
        return confusion_matrix(ref_y, pred_y).ravel()[2]

    if measure == 'tp':
        return confusion_matrix(ref_y, pred_y).ravel()[3]

    if measure == 'acc':
        return accuracy_score(ref_y, pred_y)
    if measure == 'posi_f1':
        return f1_score(ref_y, pred_y, pos_label=RELEVANT)
    if measure == 'nega_f1':
        return f1_score(ref_y, pred_y, pos_label=NON_RELEVANT)
    if measure == 'auc':
        return roc_auc_score(ref_y, pred_y_score)


def evaluate_likelihood_param_for_la(csd_data, pred_data, param_name):
    try:
        ref_value = csd_data[param_name].flatten()
        pred_value = pred_data[param_name].flatten()
        mse = mean_squared_error(ref_value, pred_value)
        return mse
    except:
        return -1


def evaluation(pred_dct, csd_data, result_folder):

    task_ids = csd_data['task_ids']
    crowd_y = csd_data['crowd_labels']
    mask_y = csd_data['crowd_labels_mask']
    ref_y = csd_data['gold_labels']

    pred_y = pred_dct['pred_y']
    pred_y_score = pred_dct['pred_y_score']

    # metrics
    # metrics for latent true labels
    cm = confusion_matrix(ref_y, pred_y)
    acc = accuracy_score(ref_y, pred_y)
    auc = roc_auc_score(ref_y, pred_y_score)
    posi_p = precision_score(ref_y, pred_y, pos_label=RELEVANT)
    posi_r = recall_score(ref_y, pred_y, pos_label=RELEVANT)
    posi_f1 = f1_score(ref_y, pred_y, pos_label=RELEVANT)
    nega_p = precision_score(ref_y, pred_y, pos_label=NON_RELEVANT)
    nega_r = recall_score(ref_y, pred_y, pos_label=NON_RELEVANT)
    nega_f1 = f1_score(ref_y, pred_y, pos_label=NON_RELEVANT)

    # metrics for task difficulty & worker competence
    task_tau, task_rho = -1, -1
    worker_tau, worker_rho = -1, -1
    # if 'pred_td' in pred_dct:
    #     pred_td = pred_dct['pred_td']
    #     ref_td = task_difficulty(crowd_y, mask_y, ref_y)
    #     task_tau, task_rho = taurho(ref_td, pred_td)
    # else:
    #     task_tau, task_rho = -1, -1
    #
    # if 'pred_ac' in pred_dct:
    #     pred_ac = pred_dct['pred_ac']
    #     ref_ac = worker_competence(crowd_y, mask_y, ref_y)
    #     worker_tau, worker_rho = taurho(ref_ac, pred_ac)
    # else:
    #     worker_tau, worker_rho = -1, -1

    with open(os.path.join(result_folder, 'eval.json'), 'w') as f:
        dct = {
            'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1],
            'acc': acc,
            'auc': auc,
            'posi_p': posi_p,
            'posi_r': posi_r,
            'posi_f1': posi_f1,
            'nega_p': nega_p,
            'nega_r': nega_r,
            'nega_f1': nega_f1,
            'task_tau': task_tau,
            'task_rho': task_rho,
            'worker_tau': worker_tau,
            'worker_rho': worker_rho
        }

        f.write('{}\n'.format(json.dumps(dct, cls=NpEncoder)))

    # predictions
    with open(os.path.join(result_folder, 'pred_labels.csv'), 'w') as f:
        for id, ref, pred in zip(task_ids, ref_y, pred_y):
            f.write('{},{},{}\n'.format(id, ref, pred))
    with open(os.path.join(result_folder, 'pred_dct.json'), 'w') as f:
        json.dump(pred_dct, f, cls=NpEncoder)

    return