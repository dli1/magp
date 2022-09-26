
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import seaborn as sns
sns.color_palette()
from matplotlib import cm, pyplot as plt


def sample_true_label(N, K=2, dim=10, noise=(5.0, 10.0)):
    """
    generate features from mixtured Gaussians
    """
    task_ids = list(range(N))
    task_features = []
    task_labels = []
    task_difficulties = []

    for _ in task_ids:
        # sample a class
        k = np.random.randint(K)

        # sample a point from k-th gaussian distribution
        rv = multivariate_normal(mean=np.eye(dim)[k], cov=np.eye(dim)*0.5)
        item_feature = rv.rvs(1)
        pdf = rv.pdf(item_feature)

        task_labels.append(k)
        task_features.append(item_feature)
        task_difficulties.append(pdf)

    scaler = MinMaxScaler(feature_range=(noise[0], noise[1]))
    task_difficulties = scaler.fit_transform(np.reshape(task_difficulties, (-1, 1)))
    task_difficulties = task_difficulties.flatten()

    return task_ids, np.array(task_features), np.array(task_labels), task_difficulties


# def sample_item_difficulty(N):
#     """
#     alpha -> + infinite: easy task
#     alpha -> 0: difficult task
#     """
#     alphas = np.random.normal(loc=1.0, scale=1.0, size=N)**2
#     return alphas
#

def glad_sample_worker_competence(M, noise=1.0, power=-1.5):
    """
    sampling_probs: beta
    beta -> + infinite: good worker
    beta -> 0 : only guessing
    beta -> - infinite: bad worker, only giving wrong answers

    power=-1.5: assume 100 workers, only 10% of the workers give 82% of the labels
    """
    worker_ids = np.array(range(M))
    worker_competences = np.random.normal(loc=noise, scale=1.0, size=M)
    sampling_probs = np.array([(i+1)**power for i in range(M)])
    sampling_probs = sampling_probs/sampling_probs.sum()
    return worker_ids, worker_competences, sampling_probs


def glad_sample_crowd_label(K, true_label, item_difficulty, worker_competence):

    sigma = np.minimum(-item_difficulty*worker_competence, 100)  # avoid "overflow encountered in exp" error
    cp = 1.0 / (1.0 + np.exp(sigma))  # correct prob
    wp = (1-cp)/(K-1)  # wrong probs

    p = np.full(K, wp)
    p[true_label] = cp
    label = np.random.choice(a=K, p=p, size=1)[0]
    return label


def glad_generate_simulated_data(K, N, M, P, task_noise=(5.0, 10.0), worker_noise=1.0, random_seed=0):
    np.random.seed(random_seed)

    data = {}

    # ground truth
    task_ids, task_features, task_labels, task_difficulties = sample_true_label(N, K, noise=task_noise)
    # item_difficulties = sample_item_difficulty(N)
    worker_ids, worker_competences, sampling_probs = glad_sample_worker_competence(M, noise=worker_noise)
    wc_dct = {wid: wc for wid, wc in zip(worker_ids, worker_competences)}
    data['task_ids'] = task_ids
    data['task_difficulty'] = task_difficulties
    data['input_features'] = task_features
    data['active_dims'] = {'rbf': list(range(task_features.shape[1]))}
    data['gold_labels'] = task_labels

    # crowd sample tuple
    sampled_workers = set()
    crowd_samples = []
    for task_id, true_label in zip(task_ids, task_labels):
        item_difficulty = task_difficulties[task_id]
        sample_worker_ids = np.random.choice(a=worker_ids, p=sampling_probs, size=P, replace=False)  # sample workers
        for worker_id in sample_worker_ids:
            sampled_workers.add(worker_id)
            worker_competence = worker_competences[worker_id]
            crowd_label = glad_sample_crowd_label(K, true_label, item_difficulty, worker_competence)
            crowd_samples.append((task_id, worker_id, crowd_label))
    data['crowd_tuples'] = crowd_samples
    data['worker_ids'] = list(sampled_workers)
    data['worker_competence'] = np.array([wc_dct[wid] for wid in data['worker_ids']])

    # make crowd sample array
    Mnew = len(data['worker_ids'])
    crowd_labels = np.zeros(shape=(N, Mnew), dtype=np.float)
    crowd_labels_mask = np.zeros(shape=(N, Mnew), dtype=np.float)

    for task_id, worker_id, crowd_label in crowd_samples:
        i = data['task_ids'].index(task_id)
        j = data['worker_ids'].index(worker_id)
        crowd_labels[i][j] = crowd_label  # crowd
        crowd_labels_mask[i][j] = 1  # 1
    data['crowd_labels'] = crowd_labels
    data['crowd_labels_mask'] = crowd_labels_mask

    return data


def mace_sample_worker_competence(M, K, power=-1.5, worker_spamming_noise=(0.0, 0.1)):
    worker_ids = np.array(range(M))

    sampling_probs = np.array([(i+1)**power for i in range(M)])
    sampling_probs = sampling_probs/sampling_probs.sum()

    worker_spamming = np.random.uniform(low=worker_spamming_noise[0], high=worker_spamming_noise[1], size=M)  # assume most workers are not spamming
    worker_labelling_dist = np.full(shape=(M, K), fill_value=1.0/K)

    return worker_ids, sampling_probs, worker_spamming, worker_labelling_dist


def mace_sample_crowd_label(K, true_label, spamming, labelling_dist):
    is_spamming = np.random.binomial(n=1, p=spamming, size=1)[0]
    if not is_spamming:
        label = true_label
    else:
        label = np.random.choice(a=K, p=labelling_dist, size=1)[0]
    return label


def mace_generate_simulated_data(K, N, M, P, worker_spamming_noise=(0.0, 0.1), random_seed=0):

    np.random.seed(random_seed)

    data = {}

    # ground truth
    task_ids, task_features, task_labels, _ = sample_true_label(N, K)
    worker_ids, sampling_probs, worker_spamming, worker_labelling_dist = mace_sample_worker_competence(M, K, worker_spamming_noise=worker_spamming_noise)  # assume 80% of the items are annotated by only 10% of the workers
    ws_dct = {wid: ws for wid, ws in zip(worker_ids, worker_spamming)}
    wld_dct = {wid: list(wld) for wid, wld in zip(worker_ids, worker_labelling_dist)}

    data['task_ids'] = task_ids
    data['input_features'] = task_features
    data['active_dims'] = {'rbf': list(range(task_features.shape[1]))}
    data['gold_labels'] = task_labels

    # crowd sample tuple
    sampled_workers = set()
    crowd_samples = []
    for task_id, true_label in zip(task_ids, task_labels):
        sample_worker_ids = np.random.choice(a=worker_ids, p=sampling_probs, size=P, replace=False)  # sample workers
        for worker_id in sample_worker_ids:
            sampled_workers.add(worker_id)
            spamming = worker_spamming[worker_id]
            labelling_dist = worker_labelling_dist[worker_id]
            crowd_label = mace_sample_crowd_label(K, true_label, spamming, labelling_dist)
            crowd_samples.append((task_id, worker_id, crowd_label))
    data['crowd_tuples'] = crowd_samples
    data['worker_ids'] = list(sampled_workers)
    data['worker_labelling_dist'] = np.array([list(wld_dct[wid]) for wid in data['worker_ids']])
    data['worker_spamming'] = np.array([ws_dct[wid] for wid in data['worker_ids']])

    # make crowd sample array
    Mnew = len(data['worker_ids'])
    crowd_labels = np.zeros(shape=(N, Mnew), dtype=np.float)
    crowd_labels_mask = np.zeros(shape=(N, Mnew), dtype=np.float)

    for task_id, worker_id, crowd_label in crowd_samples:
        i = data['task_ids'].index(task_id)
        j = data['worker_ids'].index(worker_id)
        crowd_labels[i][j] = crowd_label  # crowd
        crowd_labels_mask[i][j] = 1  # 1
    data['crowd_labels'] = crowd_labels
    data['crowd_labels_mask'] = crowd_labels_mask

    return data


def zc_sample_worker_competence(M, power=-1.5, noise=(0.5, 1.0)):
    worker_ids = np.array(range(M))

    sampling_probs = np.array([(i+1)**power for i in range(M)])
    sampling_probs = sampling_probs/sampling_probs.sum()

    worker_competences = np.random.uniform(low=noise[0], high=noise[1], size=M)
    return worker_ids, sampling_probs, worker_competences


def zc_sample_crowd_label(K, true_label, worker_competence):
    wrong_labels = list(range(K))
    wrong_labels.remove(true_label)
    is_correct_label = np.random.binomial(n=1, p=worker_competence, size=1)[0]
    if is_correct_label:
        label = true_label
    else:
        label = np.random.choice(a=wrong_labels, size=1)[0]
    return label



def zc_generate_simulated_data(K, N, M, P, worker_competence_noise, random_seed=0):
    np.random.seed(random_seed)

    data = {}
    # ground truth
    task_ids, task_features, task_labels, _ = sample_true_label(N, K)
    worker_ids, sampling_probs, worker_competences = zc_sample_worker_competence(M, noise=worker_competence_noise)  # assume 80% of the items are annotated by only 10% of the workers
    wc_dct = {wid: wc for wid, wc in zip(worker_ids, worker_competences)}

    data['task_ids'] = task_ids

    data['input_features'] = task_features
    data['active_dims'] = {'rbf': list(range(task_features.shape[1]))}
    data['gold_labels'] = task_labels

    # crowd sample tuple
    sampled_workers = set()
    crowd_samples = []
    for task_id, true_label in zip(task_ids, task_labels):
        sample_worker_ids = np.random.choice(a=worker_ids, p=sampling_probs, size=P, replace=False)  # sample workers
        for worker_id in sample_worker_ids:
            sampled_workers.add(worker_id)
            worker_competence = worker_competences[worker_id]
            crowd_label = zc_sample_crowd_label(K, true_label, worker_competence)
            crowd_samples.append((task_id, worker_id, crowd_label))
    data['crowd_tuples'] = crowd_samples
    data['worker_ids'] = list(sampled_workers)
    data['worker_competence'] = np.array([wc_dct[wid] for wid in data['worker_ids']])

    # make crowd sample array
    Mnew = len(data['worker_ids'])
    crowd_labels = np.zeros(shape=(N, Mnew), dtype=np.float)
    crowd_labels_mask = np.zeros(shape=(N, Mnew), dtype=np.float)

    for task_id, worker_id, crowd_label in crowd_samples:
        i = data['task_ids'].index(task_id)
        j = data['worker_ids'].index(worker_id)
        crowd_labels[i][j] = crowd_label  # crowd
        crowd_labels_mask[i][j] = 1  # 1
    data['crowd_labels'] = crowd_labels
    data['crowd_labels_mask'] = crowd_labels_mask


    return data


def ds_sample_worker_competence(M, K, power=-1.5, worker_correct_rate=(0.7, 0.9)):
    worker_ids = np.array(range(M))

    sampling_probs = np.array([(i+1)**power for i in range(M)])
    sampling_probs = sampling_probs/sampling_probs.sum()

    worker_confusion_matrixes = []
    for i in range(M):
        correct_p = np.random.uniform(low=worker_correct_rate[0], high=worker_correct_rate[1], size=1)[0]
        incorrect_p = (1 - correct_p) / (K-1)
        cf = np.full([K, K], incorrect_p)
        row, col = np.diag_indices_from(cf)
        cf[row, col] = correct_p
        worker_confusion_matrixes.append(cf)
    worker_confusion_matrixes = np.array(worker_confusion_matrixes)

    return worker_ids, sampling_probs, worker_confusion_matrixes


def ds_sample_crowd_label(K, true_label, worker_confusion_matrix):
    return np.random.choice(a=K, p=worker_confusion_matrix[true_label], size=1)[0]


def ds_generate_simulated_data(K, N, M, P, worker_correct_rate=(0.7, 0.8), random_seed=0):

    np.random.seed(random_seed)
    data = {}

    # ground truth
    task_ids, task_features, task_labels, _ = sample_true_label(N, K)
    worker_ids, sampling_probs, worker_confusion_matrixes = ds_sample_worker_competence(M, K, worker_correct_rate=worker_correct_rate)  # assume 80% of the items are annotated by only 10% of the workers
    wcf_dct = {wid: cf for wid, cf in zip(worker_ids, worker_confusion_matrixes)}

    data['task_ids'] = task_ids
    # data['task_difficulties'] = task_difficulties

    data['input_features'] = task_features
    data['active_dims'] = {'rbf': list(range(task_features.shape[1]))}
    data['gold_labels'] = task_labels

    # crowd sample tuple
    sampled_workers = set()
    crowd_samples = []
    for task_id, true_label in zip(task_ids, task_labels):
        sample_worker_ids = np.random.choice(a=worker_ids, p=sampling_probs, size=P, replace=False)  # sample workers
        for worker_id in sample_worker_ids:
            sampled_workers.add(worker_id)
            worker_confusion_matrix = worker_confusion_matrixes[worker_id]
            crowd_label = ds_sample_crowd_label(K, true_label, worker_confusion_matrix)
            crowd_samples.append((task_id, worker_id, crowd_label))

    data['crowd_tuples'] = crowd_samples
    data['worker_ids'] = list(sampled_workers)
    data['worker_confusion_matrix'] = np.array([wcf_dct[wid] for wid in data['worker_ids']])

    # make crowd sample array
    Mnew = len(data['worker_ids'])
    crowd_labels = np.zeros(shape=(N, Mnew), dtype=np.float)
    crowd_labels_mask = np.zeros(shape=(N, Mnew), dtype=np.float)

    for task_id, worker_id, crowd_label in crowd_samples:
        i = data['task_ids'].index(task_id)
        j = data['worker_ids'].index(worker_id)
        crowd_labels[i][j] = crowd_label  # crowd
        crowd_labels_mask[i][j] = 1  # 1
    data['crowd_labels'] = crowd_labels
    data['crowd_labels_mask'] = crowd_labels_mask

    return data


def simulated_data_label_quality(data_dct):
    df1 = pd.DataFrame(data_dct['crowd_tuples'], columns=['taskid', 'workerid', 'crowdlabel'])
    df2 = pd.DataFrame({'taskid': data_dct['task_ids'], 'feature0': [row[0] for row in data_dct['input_features']], \
                        'feature1': [row[1] for row in data_dct['input_features']], 'goldlabel':data_dct['gold_labels']})
    df = pd.merge(df1, df2)
    df['workerid'] = df['workerid'].astype(str)

    # percentage of correctness for each task
    df['correctlabel'] = df['goldlabel'] == df['crowdlabel']
    df['correctlabel'] = df['correctlabel'].astype(int)
    dff = df.groupby(by=['taskid']).mean().reset_index()

    task_mean = dff['correctlabel'].mean()
    task_std = dff['correctlabel'].std()

    # percentage of correctness for each worker
    df['correctlabel'] = df['goldlabel'] == df['crowdlabel']
    df['correctlabel'] = df['correctlabel'].astype(int)
    dff = df.groupby(by=['workerid']).mean().reset_index()

    worker_mean = dff['correctlabel'].mean()
    worker_std = dff['correctlabel'].std()

    return task_mean, task_std, worker_mean, worker_std


def plot_simulated_data(data_dct):
    df1 = pd.DataFrame(data_dct['crowd_tuples'], columns=['taskid', 'workerid', 'crowdlabel'])
    df2 = pd.DataFrame({'taskid': data_dct['task_ids'], 'feature0': [row[0] for row in data_dct['input_features']], \
                        'feature1': [row[1] for row in data_dct['input_features']], 'goldlabel':data_dct['gold_labels']})
    df = pd.merge(df1, df2)
    df['workerid'] = df['workerid'].astype(str)


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8.27*2, 8.27+2))

    # ground truth label for each task
    sns.scatterplot(data=df, x="feature0", y="feature1", hue="goldlabel", alpha=0.2, ax=ax[0][0])
    ax[0][0].set_title('ground truth label for each task')

    # crowd labels for each task
    sns.scatterplot(data=df, x="feature0", y="feature1", hue="crowdlabel", alpha=0.2, ax=ax[0][1])
    ax[0][1].set_title('crowd labels for each task')

    # percentage of correctness for each task
    df['correctlabel'] = df['goldlabel'] == df['crowdlabel']
    df['correctlabel'] = df['correctlabel'].astype(int)
    dff = df.groupby(by=['taskid']).mean().reset_index()

    mean = dff['correctlabel'].mean()
    std = dff['correctlabel'].std()
    sns.barplot(x='taskid', y='correctlabel', data=dff, ax=ax[1][0])
    ax[1][0].set_title('percentage of correctness for each task (mean = {:.2f} std={:.2f})'.format(mean, std))

    # percentage of correctness for each worker
    df['correctlabel'] = df['goldlabel'] == df['crowdlabel']
    df['correctlabel'] = df['correctlabel'].astype(int)
    dff = df.groupby(by=['workerid']).mean().reset_index()

    mean = dff['correctlabel'].mean()
    std = dff['correctlabel'].std()
    sns.barplot(x='workerid', y='correctlabel', data=dff, ax=ax[1][1])
    ax[1][1].set_title('percentage of correctness for each worker (mean = {:.2f} std={:.2f})'.format(mean, std))
    fig.suptitle('K={}, N={}, M={}, P={}'.format(K, N, M, P))
    plt.show()
    return


if __name__ == '__main__':

    K, N, M, P = 2, 1000, 10, 5
    data=glad_generate_simulated_data(K=K, N=N, M=M, P=P, task_noise=(1.4, 2.0), worker_noise=1.0)
    # data=mace_generate_simulated_data(K=K, N=N, M=M, P=P, worker_spamming_noise=(0.25, 0.4))
    # data=zc_generate_simulated_data(K=K, N=N, M=M, P=P, worker_competence_noise=(0.8, 0.85), random_seed=0)
    # data=ds_generate_simulated_data(K=K, N=N, M=M, P=P, worker_correct_rate=(0.8, 0.85), random_seed=0)
    plot_simulated_data(data)
