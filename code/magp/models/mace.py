#coding=utf-8

import subprocess
import pandas as pd

from magp.utils.common import *


def mace_aggregation(train_y, train_y_mask, mid_dir, alpha=0.5, beta=0.5):
    qid = 0
    os.chdir(mid_dir)

    # make_input_file_for_mace
    table = np.zeros(shape=train_y.shape)
    for i in range(train_y.shape[0]):   # doc
        for j in range(train_y.shape[1]):  # annotator
            if train_y_mask[i][j] == 0:  # unlabelled slot
                table[i][j] = None
            else:
                table[i][j] = train_y[i][j]
    pd.DataFrame(table).to_csv(os.path.join(mid_dir, '{}.input.csv'.format(qid)), sep=',', header=False, index=False)

    # call mace in command
    cmd = 'java -jar {}/MACE.jar --alpha {} --beta {} --iterations 100 --entropies --prefix {} {}/{}.input.csv'.format(MACE_JAR_PATH, alpha, beta, qid, mid_dir, qid)
    subprocess.check_output(cmd, shell=True)

    # extract prediction
    with open(os.path.join(mid_dir, '{}.prediction').format(qid)) as f:
        pred_y = np.array([float(line.strip()) for line in f])

    # extract task difficulty
    with open(os.path.join(mid_dir, '{}.entropies').format(qid)) as fin:
        pred_td = np.array([float(item) for item in fin.readlines() if item.strip()])

    # extract annotator capability
    with open(os.path.join(mid_dir, '{}.competence').format(qid)) as fin:
        pred_ac = np.array([float(item) for item in fin.readlines()[0].strip().split('\t')])

    pred_dct = {
        'pred_y': pred_y,
        'pred_y_score': pred_y,
        'task_difficulty': pred_td,
        'worker_competence': pred_ac
    }
    return pred_dct

if __name__ == '__main__':

    pass
