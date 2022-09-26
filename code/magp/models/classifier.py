#coding=utf-8

import numpy as np
from sklearn.linear_model import LogisticRegression


def classifier_aggregation(train_x, train_crowd_y, train_mask_y, train_ref_y,
                           test_x, test_crowd_y, test_mask_y, random_seed):

    # features: mean, std, median, max, min
    def extract_crowd_feature(crowd_y, crowd_y_mask):
        crowd_features = []
        for y, ymask in zip(crowd_y, crowd_y_mask):
            labels = [yy for yy, yymask in zip(y, ymask) if yymask == 1]
            mean = np.mean(labels)
            std = np.std(labels)
            median = np.median(labels)
            max = np.max(labels)
            min = np.min(labels)

            crowd_features.append((mean, std, median, max, min))
        crowd_features = np.array(crowd_features)
        return crowd_features

    train_crowd_feature = extract_crowd_feature(train_crowd_y, train_mask_y)
    new_train_x = np.concatenate([train_x, train_crowd_feature], axis=1)

    test_crowd_feature = extract_crowd_feature(test_crowd_y, test_mask_y)
    new_test_x = np.concatenate([test_x, test_crowd_feature], axis=1)

    model = LogisticRegression(max_iter=10000, random_state=random_seed)
    model.fit(new_train_x, train_ref_y.flatten())

    pred_y = model.predict(new_test_x)
    pred_dct = {'pred_y': pred_y.flatten()}
    return pred_dct