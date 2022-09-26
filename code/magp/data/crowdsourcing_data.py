#coding=utf-8

import pickle
import pandas as pd
from collections import defaultdict
from magp.common import *


class CrowdSourcingData(object):
    def __init__(self, data_name, task_acc=None, task_anno_num=None, worker_acc=None, worker_anno_num=None,
                 enable_feature=True, feature_name='lexical.rank.onehot.clueweb09.bert',
                 csd_file='crowdsourcing_data.csv'):
        self.data_name = data_name
        self.enable_feature = enable_feature
        self.csd_df = self._init_crowdsoucing_data(data_name, csd_file, task_acc, task_anno_num, worker_acc, worker_anno_num)
        if enable_feature:
            self.feature_dct, self.active_dims = self._init_feature(data_name, feature_name)
        else:
            self.feature_dct, self.active_dims = None, None

    def _init_crowdsoucing_data(self, data_name, csd_file, task_acc, task_anno_num, worker_acc, worker_anno_num):
        csd_file = os.path.join(DATA_DIR, data_name, csd_file)
        csd_df = pd.read_csv(csd_file, sep=',', names=COLUMNS, header=0, dtype=COLUMN_TYPES)

        if (task_acc is not None) or (task_anno_num is not None) or (worker_acc is not None) or (worker_anno_num is not None):
            csd_df['correctlabel'] = csd_df['gold'] == csd_df['label']
            csd_df['correctlabel'] = csd_df['correctlabel'].astype(int)

            task_acc_df = csd_df.groupby(by=['topicID', 'docID'], as_index=False).agg({'correctlabel': 'mean'})
            task_acc_df.rename(columns={'correctlabel': 'task_acc'}, inplace=True)

            worker_acc_df = csd_df.groupby(by=['workerID'], as_index=False).agg({'correctlabel': 'mean'})
            worker_acc_df.rename(columns={'correctlabel': 'worker_acc'}, inplace=True)

            task_anno_num_df = csd_df.groupby(by=['topicID', 'docID'], as_index=False).agg({'label': 'count'})
            task_anno_num_df.rename(columns={'label': 'task_anno_num'}, inplace=True)

            worker_anno_num_df = csd_df.groupby(by=['workerID'], as_index=False).agg({'label': 'count'})
            worker_anno_num_df.rename(columns={'label': 'worker_anno_num'}, inplace=True)

            csd_df = pd.merge(csd_df, task_acc_df, on=['topicID', 'docID'])
            csd_df = pd.merge(csd_df, task_anno_num_df, on=['topicID', 'docID'])
            csd_df = pd.merge(csd_df, worker_acc_df, on=['workerID'])
            csd_df = pd.merge(csd_df, worker_anno_num_df, on=['workerID'])

            if task_acc:
                csd_df = csd_df[(csd_df['task_acc'] >= task_acc[0]) & (csd_df['task_acc'] < task_acc[1])]
            if task_anno_num:
                csd_df = csd_df[(csd_df['task_anno_num'] >= task_anno_num[0]) & (csd_df['task_anno_num'] < task_anno_num[1])]
            if worker_acc:
                csd_df = csd_df[(csd_df['worker_acc'] >= worker_acc[0]) & (csd_df['worker_acc'] < worker_acc[1])]
            if worker_anno_num:
                csd_df = csd_df[(csd_df['worker_anno_num'] >= worker_anno_num[0]) & (csd_df['worker_anno_num'] < worker_anno_num[1])]
            print('csd_df total len: {}'.format(len(csd_df)))
        return csd_df

    def _init_feature(self, data_name, feature_name):
        qid_did_list = self.csd_df[['topicID', 'docID']].drop_duplicates().values  # here we must separate qid and did

        active_dims = {}
        feature_dct = defaultdict(list)
        mdir = os.path.join(DATA_DIR, data_name, 'feature')
        feature_files = FEATURE_INFO[feature_name]['feature_files']
        for infile in feature_files:

            with open(os.path.join(mdir, infile), 'rb') as f:
                dct = pickle.load(f)

                # www version: it contains a bug
                # feature_num = len(list(dct.values())[0])
                # value = list(range(feature_num))

                if len(feature_dct.keys()) == 0:
                    existing_feature_num = 0
                else:
                    existing_feature_num = len(list(feature_dct.values())[0])
                new_feature_num = len(list(dct.values())[0])
                value = list(range(existing_feature_num, existing_feature_num + new_feature_num))

                if 'lexical' in infile:
                    key = 'rbf'
                elif 'rank' in infile:
                    key = 'rbf'
                elif 'bert' in infile:
                    key = 'linear'
                else:
                    raise NotImplementedError

                if key in active_dims:
                    active_dims[key].extend(value)
                else:
                    active_dims[key] = value

                for qid, did in qid_did_list:
                    if qid.strip() + did.strip() in dct:
                        feature_dct[qid.strip() + did.strip()].extend(dct[qid.strip() + did.strip()])
                    else:
                        print('{} not in {}\'s keys'.format(qid.strip() + did.strip(), infile))

        return feature_dct, active_dims

    def get_active_dims(self):
        return self.active_dims

    def get_qids(self):
        return self.csd_df['topicID'].unique()

    def split_topics(self, i_fold, k_folds):
        assert 0 <= i_fold < k_folds

        # use this for future
        # topics = list(self.csd_df['topicID'].unique().flatten())

        # now keep consistent with the splits in the experiment
        if self.data_name == 'cs2010':
            topics = CS2010_TOPICS
        elif self.data_name == 'cs2011':
            topics = CS2011_TOPICS
        else:
            raise ValueError

        one_fold_num = int(len(topics)/k_folds)

        train_qids = topics[one_fold_num * i_fold: one_fold_num * (i_fold + 1)]
        test_qids = [item for item in topics if item not in train_qids]

        return train_qids, test_qids



    def get_data(self, target_qids=None):

        if target_qids is not None and isinstance(target_qids, list):
            csd_df = self.csd_df.loc[self.csd_df.topicID.isin(target_qids)]
        else:
            csd_df = self.csd_df

        tasks = [qid.strip() + '--' + did.strip() for (qid, did) in csd_df[['topicID', 'docID']].drop_duplicates(
            subset=['topicID', 'docID']).values]
        workers = list(csd_df['workerID'].unique())

        # make arrays
        row_num, col_num = len(tasks), len(workers)
        crowd_labels = np.zeros(shape=(row_num, col_num), dtype=np.float)
        crowd_labels_mask = np.zeros(shape=(row_num, col_num), dtype=np.float)
        gold_labels = np.zeros(shape=(row_num), dtype=np.int)

        for _, row in csd_df.iterrows():
            task = row['topicID'].strip() + '--' + row['docID'].strip()
            worker = row['workerID']
            i = tasks.index(task)
            j = workers.index(worker)

            crowd_labels[i][j] = row['label']  # crowd
            crowd_labels_mask[i][j] = 1  # 1
            gold_labels[i] = row['gold']  # gold

        if self.enable_feature:
            feature_num = len(list(self.feature_dct.items())[0][1])
            input_features = np.zeros(shape=(row_num, feature_num), dtype=np.float)
            for _, row in csd_df.iterrows():
                qiddid = row['topicID'].strip() + row['docID'].strip()
                task = row['topicID'].strip() + '--' + row['docID'].strip()
                i = tasks.index(task)
                input_features[i] = self.feature_dct[qiddid]  # feature
        else:
            input_features = None

        data = {'task_ids': tasks,
                'worker_ids': workers,
                'input_features': input_features,
                'crowd_labels': crowd_labels,
                'crowd_labels_mask': crowd_labels_mask,
                'gold_labels': gold_labels,
                'active_dims': self.get_active_dims(),
                }

        # for baselines ds, glad, mace, zc
        csd_df['task'] = csd_df['topicID'] + '--' +csd_df['docID']
        crowd_tuples = csd_df[['task', 'workerID', 'label']].drop_duplicates().values
        data['crowd_tuples'] = crowd_tuples

        return data
