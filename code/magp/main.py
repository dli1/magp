#coding=utf-8


import random
import logging
import datetime

from magp.models.la import la_aggregation
from magp.models.gp import gp_aggregation, likelihood_aggregation, gpmv_aggregation
from magp.models.mv import mv_aggregation
from magp.models.mace import mace_aggregation

from magp.data.crowdsourcing_data import CrowdSourcingData

from magp.utils.common import *
from magp.utils.eval_metrics import *
from magp.data.simulated_data import *
from magp.utils.config import *

logging.basicConfig(level=logging.INFO)


def run(**kwargs):
    """
    Usage: python -m magp.main run --

    """
    cfg = get_cfg_default()
    cfg.merge_from_list(get_list_from_kwargs(**kwargs))  # update from command line
    cfg.freeze()
    print(cfg)

    data_name=cfg.data_name
    simu_func=cfg.simu_func
    K=cfg.K
    N=cfg.N
    M=cfg.M
    P=cfg.P

    glad_task_noise = cfg.glad_task_noise
    glad_worker_noise = cfg.glad_worker_noise
    mace_worker_spamming_noise = cfg.mace_worker_spamming_noise
    zc_worker_competence_noise = cfg.zc_worker_competence_noise
    ds_worker_correct_rate = cfg.ds_worker_correct_rate

    task_acc = cfg.task_acc
    task_anno_num = cfg.task_anno_num
    worker_acc = cfg.worker_acc
    worker_anno_num = cfg.worker_anno_num

    csd_file = cfg.csd_file
    enable_cv = cfg.enable_cv
    k_folds = cfg.k_folds
    i_fold = cfg.i_fold
    enable_feature = cfg.enable_feature
    feature_name = cfg.feature_name

    model_name=cfg.model_name

    # mean function
    mean_function_name=cfg.mean_function_name  # for gpmv, gpcrowd, gpglad, gpmace, gpzc, gpds only

    # likelihood parameters
    gpcrowd_noise=cfg.gpcrowd_noise  # for gpcrowd, likelihood only
    alpha=cfg.alpha  # for mace only
    beta=cfg.beta  # for mace only
    ds_initquality=cfg.ds_initquality  # for ds only

    # optimization parameters
    epoch_num=cfg.epoch_num
    optimizor_name=cfg.optimizor_name
    learning_rate=cfg.learning_rate
    param_update=cfg.param_update
    threshold=cfg.threshold

    random_seed=cfg.random_seed

    enable_logging=cfg.enable_logging
    logging_freq=cfg.logging_freq
    log_folder=cfg.log_folder
    result_folder=cfg.result_folder

    print(datetime.datetime.now(), 'start.')

    # path
    log_folder = os.path.join(RET_DIR, data_name, log_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    result_folder = os.path.join(RET_DIR, data_name, result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # generate data
    # simulated data
    if data_name == 'simu':
        print(datetime.datetime.now(), 'generate simulated data.')
        if simu_func == 'glad':
            crowdlabeled_data = glad_generate_simulated_data(K, N, M, P, task_noise=glad_task_noise, worker_noise=glad_worker_noise, random_seed=random_seed)
        elif simu_func == 'mace':
            crowdlabeled_data = mace_generate_simulated_data(K, N, M, P, worker_spamming_noise=mace_worker_spamming_noise, random_seed=random_seed)
        elif simu_func == 'zc':
            crowdlabeled_data = zc_generate_simulated_data(K, N, M, P, worker_competence_noise=zc_worker_competence_noise, random_seed=random_seed)
        elif simu_func == 'ds':
            crowdlabeled_data = ds_generate_simulated_data(K, N, M, P, worker_correct_rate=ds_worker_correct_rate, random_seed=random_seed)
        else:
            raise NotImplementedError

        # write simulated data label quality
        with open(os.path.join(result_folder, 'simu_data_quality.txt'), 'w') as f:
            task_mean, task_std, worker_mean, worker_std = simulated_data_label_quality(crowdlabeled_data)
            f.write('{},{},{},{}'.format(task_mean, task_std, worker_mean, worker_std))
    # real data
    else:
        print(datetime.datetime.now(), 'load {} data.'.format(data_name))
        csd = CrowdSourcingData(data_name, task_acc, task_anno_num, worker_acc, worker_anno_num, enable_feature, feature_name, csd_file)
        if enable_cv:
            crowdlabeled_qids, unlabeled_qids = csd.split_topics(i_fold, k_folds)
        else:
            crowdlabeled_qids, unlabeled_qids = None, None

        crowdlabeled_data = csd.get_data(target_qids=crowdlabeled_qids)

    # model
    print(datetime.datetime.now(), 'running model.')
    if model_name == 'mv':
        crowd_y = crowdlabeled_data['crowd_labels']
        mask_y = crowdlabeled_data['crowd_labels_mask']
        pred_dct = mv_aggregation(crowd_y, mask_y)

    elif model_name == 'ds' or model_name == 'glad' or model_name == 'zc':
        crowd_tuples = crowdlabeled_data['crowd_tuples']
        pred_dct = la_aggregation(crowd_tuples=crowd_tuples,
                                  likelihood_name=model_name,
                                  ds_initquality=ds_initquality,
                                  epoch_num=epoch_num,
                                  threshold=threshold,
                                  random_seed=random_seed,
                                  enable_logging=enable_logging,
                                  csd_data=crowdlabeled_data,
                                  log_path=log_folder,
                                  logging_freq=logging_freq)

    elif model_name == 'mace':

        mid_dir = os.path.join(result_folder, model_name, str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+str(random.randint(0,100)) + 'mid')
        if not os.path.exists(mid_dir):
            os.makedirs(mid_dir)
        crowd_y = crowdlabeled_data['crowd_labels']
        mask_y = crowdlabeled_data['crowd_labels_mask']
        pred_dct = mace_aggregation(crowd_y, mask_y, mid_dir, alpha, beta)

    elif (model_name == 'gpcrowd') or (model_name == 'gpglad') or (model_name == 'gpmace') or (model_name == 'gpzc') \
        or (model_name == 'gpds'):

        new_x = crowd_x = crowdlabeled_data['input_features']
        crowd_y = crowdlabeled_data['crowd_labels']
        mask_y = crowdlabeled_data['crowd_labels_mask']
        active_dims = crowdlabeled_data['active_dims']

        if mean_function_name == 'pretrain':
            pretrain_mean_function_path = os.path.join(DATA_DIR, data_name, 'meanfunc', feature_name + '.joblib')
        else:
            pretrain_mean_function_path = None
        pred_dct = gp_aggregation(train_x=crowd_x, train_y=crowd_y, train_y_mask=mask_y, test_x=new_x, active_dims=active_dims,
                                    mean_function_name=mean_function_name, mean_function_path=pretrain_mean_function_path,
                                    likelihood_name=model_name,
                                    gpcrowd_noise=gpcrowd_noise,
                                    epoch_num=epoch_num,
                                    optimizor_name=optimizor_name,
                                    learning_rate=learning_rate,
                                    param_update=param_update,
                                    threshold=threshold,
                                    random_seed=random_seed,
                                  crowdlabeled_data=crowdlabeled_data,
                                  enable_logging=enable_logging, log_path=log_folder, logging_freq=logging_freq)

    elif model_name == 'likelihood':

        crowd_y = crowdlabeled_data['crowd_labels']
        mask_y = crowdlabeled_data['crowd_labels_mask']

        pred_dct = likelihood_aggregation(train_y=crowd_y, train_y_mask=mask_y,
                                          log_path=log_folder,
                                          epoch_num=epoch_num,
                                          noise_level=gpcrowd_noise,
                                          random_seed=random_seed)
    elif model_name == 'gpmv':

        new_x = crowd_x = crowdlabeled_data['input_features']
        crowd_y = crowdlabeled_data['crowd_labels']
        mask_y = crowdlabeled_data['crowd_labels_mask']
        active_dims = crowdlabeled_data['active_dims']

        if mean_function_name == 'pretrain':
            pretrain_mean_function_path = os.path.join(DATA_DIR, data_name, 'meanfunc', feature_name + '.joblib')
        else:
            pretrain_mean_function_path = None

        pred_dct = gpmv_aggregation(train_x=crowd_x, train_y=crowd_y, train_y_mask=mask_y, test_x=new_x, active_dims=active_dims,
                                    mean_function_name=mean_function_name, mean_function_path=pretrain_mean_function_path,
                                    log_path=log_folder,
                                    epoch_num=epoch_num,
                                    noise_level=gpcrowd_noise,
                                    random_seed=random_seed)
    else:
        raise TypeError

    # evaluation
    print(datetime.datetime.now(), 'evaluating model.')
    evaluation(pred_dct, crowdlabeled_data, result_folder)

    # config file
    with open(os.path.join(result_folder, 'cfg.yaml'), 'w') as f:
        f.write(cfg.dump())

    print(datetime.datetime.now(), 'end.')

    return


if __name__ == '__main__':
    # automatically generating command line interfaces (CLIs)
    import fire
    fire.Fire()

    pass