#coding=utf-8

import numpy as np
from collections import Counter
from magp.utils.common import RELEVANT, NON_RELEVANT


def mv_aggregation(train_y, train_y_mask):

    pred_y = []
    for y, mask in zip(train_y, train_y_mask):
        ym = [yi for yi, mi in zip(y, mask) if mi == 1]
        assert len(ym) != 0
        counter = Counter(ym)
        if counter[RELEVANT] >= counter[NON_RELEVANT]:
            label = RELEVANT
        else:
            label = NON_RELEVANT
        pred_y.append(label)
    pred_y = np.array(pred_y, dtype=np.int)
    pred_dct = {'pred_y': pred_y, 'pred_y_score': pred_y}
    return pred_dct


