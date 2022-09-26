#coding=utf-8
from yacs.config import CfgNode

cfg = CfgNode()


cfg.data_name = 'cs2010'

# simulated data parameters
cfg.simu_func = 'ds'
cfg.K = 2
cfg.N = 1000
cfg.M = 10
cfg.P = 5
cfg.glad_task_noise = (1.0, 2.0)  # [0, +inf)  MinMaxScaler(feature_range=(noise[0], noise[1]))
cfg.glad_worker_noise = 1.0  # (-inf, +inf)  np.random.normal(loc=noise, scale=1.0, size=M)
cfg.mace_worker_spamming_noise = (0.0, 0.1)  # np.random.uniform(low=worker_spamming_noise[0], high=worker_spamming_noise[1], size=M)
cfg.zc_worker_competence_noise = (0.5, 0.8)  # np.random.uniform(low=noise[0], high=noise[1], size=M)
cfg.ds_worker_correct_rate = (0.6, 0.9)

# real data parameters
cfg.csd_file = 'crowdsourcing_data.csv'
cfg.enable_cv = False
cfg.k_folds = 5
cfg.i_fold = 0
cfg.task_acc = None
cfg.task_anno_num = None
cfg.worker_acc = None
cfg.worker_anno_num = None


cfg.enable_feature = True
cfg.feature_name = 'clueweb09.bert'

cfg.model_name = 'zc'
cfg.mean_function_name = 'zero'  # only for gpmv, gpcrowd, gpglad, gpmace, gpzc, gpds

cfg.gpcrowd_noise = 0.0  # only for gpcrowd, likelihood

# only for mace
cfg.alpha = 0.5
cfg.beta = 0.5

cfg.ds_initquality = 0.7

cfg.epoch_num = 120
cfg.optimizor_name = 'adam'
cfg.learning_rate = 0.001
cfg.param_update = 'VEM'

cfg.threshold = 1e-8

cfg.random_seed = 0

cfg.enable_logging = True
cfg.logging_freq = 5
cfg.log_folder = 'log'
cfg.result_folder = 'result'


def get_cfg_default():
    return cfg.clone()