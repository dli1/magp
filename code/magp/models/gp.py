#coding=utf-8

import joblib
import random
from collections import Counter
from tqdm import tqdm
import scipy.stats as ss
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, Tuple, Union

import gpflow
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.utilities import positive, triangular, set_trainable, to_default_float, parameter_dict
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.likelihoods.utils import inv_probit
from gpflow.quadrature import NDiagGHQuadrature
from gpflow.models.util import data_input_to_tensor
from gpflow.models.training_mixins import InputData, OutputData, InternalDataTrainingLossMixin
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import MeanFunction, Linear, Zero
from gpflow.conditionals import conditional

from gpflow.monitor import (
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
from sklearn.linear_model import LogisticRegression
from magp.models.mv import mv_aggregation
OutputMask = Union[tf.Tensor]
AnnotationData = Tuple[InputData, OutputData, OutputMask]
AnnotationDataWithoutInput = Tuple[OutputData, OutputMask]

from magp.utils.eval_metrics import *

import logging
logging.basicConfig(level=logging.INFO)


def sum_to_one() -> tfp.bijectors.Bijector:
    return tfp.bijectors.SoftmaxCentered()


def min_zero_max_one():
    return tfp.bijectors.Sigmoid()


def acquisition_ei(mean, var):
    fmin = 0

    var = np.maximum(var, default_jitter())

    a = mean -fmin
    s = np.sqrt(var)
    z = a / s

    cdf = ss.norm.cdf(z)
    pdf = ss.norm.pdf(z)
    ei = a * cdf + s * pdf

    return ei


def acquisition_pi(mean, var):
    fmin = 0

    a = mean - fmin
    s = np.sqrt(var)
    z = a / s

    # get the cdf
    cdf = ss.norm.cdf(z)

    return cdf


def acquisition_ucb(mean, var):
    """
    adapted from pybo
    """
    xi = 0.2
    delta = 0.1
    d = len(mean)

    a = xi * 2 * np.log(np.pi ** 2 / 3 / delta)
    b = xi * (4 + d)

    beta = a + b * np.log(d + 1)
    ucb = mean + np.sqrt(beta * var)

    return ucb


def inv_probit_likelihood(F, Y):
    """
	Dan Li @20190117
    :param F: b_i^j
    :param Y: y_i^j
    :return: p(yij | bij)
    """

    p = 0.5 * (1.0 + tf.math.erf(F / np.sqrt(2.0))) * (1 - 2 * default_jitter()) + default_jitter()
    return tf.where(tf.equal(Y, 1), p, 1 - p)


def dense_tensor_to_sparse_tensor(arr):
    idx  = np.where(arr != 0.0)
    return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)


class CrowdScalarLikelihood(ScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num, **kwargs):
        super().__init__(**kwargs)
        self.num_gauss_hermite_points = 20

        self.latent_dim = latent_dim  # dim = 1 in our model
        self.K = label_num
        self.N = task_num
        self.M = worker_num
        assert (self.K != 0) or (self.N != 0) or (self.M != 0)

    def calc_quadrature(self, Fmu, Fvar):
        """
        Gauss-Hermite method, calculate sum_{i=1,N} sum_{k=1,n_gh} w_ik * logp(f_ik)
        """
        quadrature = NDiagGHQuadrature(self.latent_dim, self.num_gauss_hermite_points)
        QuadrFmu, QuadrW = quadrature._build_X_W(Fmu, Fvar)  # x: [n_gh, N, dim], w: [n_gh, 1, 1]

        return QuadrFmu, QuadrW

    def calc_p_zi_fi(self, Fmu):
        """
        [n_gh, N, dim] -> [n_gh, N, dim, K]
        """
        n_gh = self.num_gauss_hermite_points
        K = self.K
        N = self.N
        dim = self.latent_dim

        zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        zi1 = tf.tile(tf.reshape(zi, [1, N, 1, K]), [n_gh, 1, dim, 1])
        Fmu1 = tf.tile(tf.reshape(Fmu, [n_gh, N, dim, 1]), [1, 1, 1, K])
        p_zi_fi = inv_probit_likelihood(Fmu1, zi1)

        return p_zi_fi

    def calc_log_p_yi_fi(self, p_zi_fi, p_yij_zi, Y_mask):
        """
        [n_gh, N, dim, K], [N, M, K] -> [n_gh, N, dim]
        """
        n_gh = self.num_gauss_hermite_points
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim

        p_zi_fi = tf.tile(tf.reshape(p_zi_fi, [n_gh, N, 1, dim, K]), [1, 1, M, 1, 1])  # [n_gh, N, M, dim, K]
        p_yij_zi = tf.tile(tf.reshape(p_yij_zi, [1, N, M, 1, K]), [n_gh, 1, 1, dim, 1])
        Y_mask = tf.tile(tf.reshape(Y_mask, [1, N, M, 1, 1]), [n_gh, 1, 1, dim, K])
        p_y_f = tf.where(tf.equal(Y_mask, 1), p_yij_zi * p_zi_fi, 1)
        p_y_f = tf.reduce_prod(p_y_f, [2])  # multiplication over M
        p_y_f = tf.reduce_sum(p_y_f, [-1])  # [n_gh, N, dim]  sum over z
        log_p_yi_fi = tf.math.log(p_y_f)

        return log_p_yi_fi

    def calc_p_yij_zi(self, Y):
        raise NotImplementedError

    def _scalar_log_prob(self, F, Y, Y_mask):
        """
        log p(Y|F)

        # TODO: need to change dimension
        # p_zi_fi = self.calc_p_zi_fi(F)  # [n_gh, N, dim, K]
        # p_yij_zi = self.calc_p_yij_zi(Y)  # [N, M, K]
        # log_p_yi_fi = self.calc_log_p_yi_fi(p_zi_fi, p_yij_zi, Y_mask)  # [n_gh, N, dim]
        # return log_p_yi_fi

        """
        print("I'm called.")
        return None

    def _variational_expectations(self, Fmu, Fvar, Y, Y_mask):
        """
        calculate ∫log(p(y|f))q(f)df
        注意这里的Fmu, Fvar的每个分量fi，表示变分高斯分布的边际分布的均值和协方差  维度都是 N x dim，N为样本数，dim为latent function个数
        Fmu [N, 1]
        Fvar [N, 1]
        Y: [N, M]
        Y_mask: [N, M]
        """

        Fmu0, W0 = self.calc_quadrature(Fmu, Fvar)  # x: [n_gh, N, dim], w: [n_gh, 1, 1]
        p_zi_fi = self.calc_p_zi_fi(Fmu0)  # [n_gh, N, dim, K]
        p_yij_zi = self.calc_p_yij_zi(Y)  # [N, M, K]
        log_p_yi_fi = self.calc_log_p_yi_fi(p_zi_fi, p_yij_zi, Y_mask)  # [n_gh, N, dim]
        ve = tf.reduce_sum(W0 * log_p_yi_fi, [0, 1])  # ∫log(p(y|f))q(f)df -> w_ik * log_p_y_f, [n_gh, N, dim] -> [dim]
        return ve

    def variational_expectations(self, Fmu, Fvar, Y, Y_mask):
        # check shape
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        _ = tf.broadcast_dynamic_shape(tf.shape(Fmu)[:-1], tf.shape(Y)[:-1])

        ret = self._variational_expectations(Fmu, Fvar, Y, Y_mask)
        return ret

    def calc_ve_zf(self, W, q_z, p_zi_fi):
        n_gh = self.num_gauss_hermite_points
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim

        log_qz_1 = tf.tile(tf.reshape(tf.math.log(q_z), [1, N, dim, 1]), [n_gh, 1, 1, 1])
        log_qz_0 = tf.tile(tf.reshape(tf.math.log(1-q_z), [1, N, dim, 1]), [n_gh, 1, 1, 1])
        log_qz = tf.concat([log_qz_0, log_qz_1], axis=-1)  # [n_gh, N, dim, 2]

        qz_1 = tf.tile(tf.reshape(q_z, [1, N, dim, 1]), [n_gh, 1, 1, 1])
        qz_0 = tf.tile(tf.reshape(1-q_z, [1, N, dim, 1]), [n_gh, 1, 1, 1])
        qz = tf.concat([qz_0, qz_1], axis=-1)  # [N, M, dim, K]

        log_p_zi_fi = tf.math.log(p_zi_fi)  # [n_gh, N, dim, K]
        kl_z = qz * (log_p_zi_fi - log_qz)
        kl_z = tf.reduce_sum(kl_z, [-1])  # [n_gh, N, dim]

        # calculate E_q(f) KL -> w_ik * kl_z
        ve_zf = tf.reduce_sum(W * kl_z, [0, 1])  # [n_gh, N, dim] -> [dim]

        return ve_zf

    def calc_ve_z(self, q_z, Y, Y_mask):
        n_gh = self.num_gauss_hermite_points
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim

        q_zi = tf.tile(tf.reshape(q_z, [N, 1, 1, K]), [1, M, dim, 1])

        p_yij_zi = self.calc_p_yij_zi(Y)
        log_p_yij_zi = tf.math.log(p_yij_zi)
        log_p_yij_zi = tf.tile(tf.reshape(log_p_yij_zi, [N, M, 1, K]), [1, 1, dim, 1])  # [N, M, dim, K]

        Y_mask = tf.tile(tf.reshape(Y_mask, [N, M, 1, 1]), [1, 1, dim, K])

        ve_z = tf.reduce_sum(log_p_yij_zi * q_zi * Y_mask, [0, 1, 3])  # [dim]

        return ve_z

    def _z_f_variational_expectations(self, q_z, Fmu, Fvar, Y, Y_mask):

        # calculate E_q(z) log p(y | z)
        ve_z = self.calc_ve_z(q_z, Y, Y_mask)

        # calculate E_q(f) KL(q(z) || p(z|f)) -> sum_{i=1,N} sum_{k=1,n_gh} w_ik * KL(q(z) || p(z|f))
        Fmu0, W0 = self.calc_quadrature(Fmu, Fvar)

        # step2: calculate p(z_i | f_i)
        p_zi_fi = self.calc_p_zi_fi(Fmu0)

        # step 3: calculate KL = E_q(z) ( log p(z_i | f_i) - log q(z) )
        ve_zf = self.calc_ve_zf(W0, q_z, p_zi_fi)

        ve = ve_z + ve_zf
        return ve

    def z_f_variational_expectations(self, q_z, Fmu, Fvar, Y, Y_mask):
        # check shape
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        _ = tf.broadcast_dynamic_shape(tf.shape(Fmu)[:-1], tf.shape(Y)[:-1])

        ret = self._z_f_variational_expectations(q_z, Fmu, Fvar, Y, Y_mask)
        return ret

    def _predict_mean_and_var(self, Fmu, Fvar):
        """
        Copied from Bernoulli likelihood.
        """
        p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
        return p, p - tf.square(p)


class GLADLikelihood(CrowdScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num,
                 task_difficulty: Union[list, np.ndarray], worker_competence: Union[list, np.ndarray],
                 **kwargs):
        super().__init__(latent_dim, label_num, task_num, worker_num, **kwargs)

        self.task_difficulty = Parameter(np.array(task_difficulty).reshape(1, -1), transform=positive())  # beta: 0 -> difficult task, +inf -> easy task
        self.task_difficulty.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(1.0))
        self.worker_competence = Parameter(np.array(worker_competence).reshape(1, -1))  # alpha -> -inf bad worker, -> 0 random worker, -> +inf good worker
        self.worker_competence.prior = tfp.distributions.Normal(loc=to_default_float(0.0), scale=to_default_float(1.0))

    def calc_p_yij_zi(self, Y):
        """
        [N, M] -> [N, M, K]
        """
        n_gh = self.num_gauss_hermite_points
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim

        zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        zi1 = tf.tile(tf.reshape(zi, [N, 1, K]), [1, M, 1])

        td = tf.tile(tf.reshape(self.task_difficulty, [N, 1]), [1, M])
        ac = tf.tile(tf.reshape(self.worker_competence, [1, M]), [N, 1])
        p_zi_eq_yij = 1.0 / (1.0 + tf.exp(-td*ac))  # [N, M]
        p_zi_eq_yij1 = tf.tile(tf.reshape(p_zi_eq_yij, [N, M, 1]), [1, 1, K])

        Y1 = tf.tile(tf.reshape(Y, [N, M, 1]), [1, 1, K])

        p_yij_zi = tf.where(tf.equal(Y1, zi1), p_zi_eq_yij1, 1 - p_zi_eq_yij1)

        return p_yij_zi


class ZCLikelihood(CrowdScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num,
                 worker_competence: Union[list, np.ndarray],
                 **kwargs):
        super().__init__(latent_dim, label_num, task_num, worker_num, **kwargs)

        self.worker_competence = Parameter(np.array(worker_competence).reshape(1, -1), transform=min_zero_max_one())  # eta \in [0, 1]
        self.worker_competence.prior = tfp.distributions.Uniform(low=to_default_float(0.0), high=to_default_float(1.0))

    def calc_p_yij_zi(self, Y):
        """
        [N, M] -> [N, M, K]
        """
        K = self.K
        N = self.N
        M = self.M

        zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        zi1 = tf.tile(tf.reshape(zi, [N, 1, K]), [1, M, 1])
        p_zi_eq_yij = tf.tile(tf.reshape(self.worker_competence, [1, M, 1]), [N, 1, K])
        Y1 = tf.tile(tf.reshape(Y, [N, M, 1]), [1, 1, K])
        p_yij_zi = tf.where(tf.equal(Y1, zi1), p_zi_eq_yij, 1 - p_zi_eq_yij)

        return p_yij_zi


class MACELikelihood(CrowdScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num,
                 worker_spamming: Union[list, np.ndarray], worker_labelling_dist: Union[list, np.ndarray],
                 **kwargs):
        super().__init__(latent_dim, label_num, task_num, worker_num, **kwargs)

        self.worker_spamming = Parameter(np.array(worker_spamming).reshape(1, worker_num), transform=min_zero_max_one())  # eta \in [0, 1]  -> 0 spamming, -> 1 not spamming
        self.worker_spamming.prior = tfp.distributions.Uniform(low=to_default_float(0.0), high=to_default_float(1.0))
        self.worker_labelling_dist = Parameter(np.array(worker_labelling_dist).reshape(worker_num, label_num), transform=sum_to_one())
        self.worker_labelling_dist.prior = tfp.distributions.Beta(to_default_float(1.0), to_default_float(1.0))

    def calc_p_yij_zi(self, Y):
        """
        [N, M] -> [N, M, K]
        """
        K = self.K
        N = self.N
        M = self.M

        zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        zi1 = tf.tile(tf.reshape(zi, [N, 1, K]), [1, M, 1])

        epsilon = tf.tile(tf.reshape(self.worker_spamming, [1, M, 1]), [N, 1, K])
        chi = tf.tile(tf.reshape(self.worker_labelling_dist, [1, M, K]), [N, 1, 1])
        p_zi_eq_yij = 1 - epsilon + epsilon * chi
        p_zi_neq_yij = epsilon * chi

        Y1 = tf.tile(tf.reshape(Y, [N, M, 1]), [1, 1, K])
        p_yij_zi = tf.where(tf.equal(Y1, zi1), p_zi_eq_yij, p_zi_neq_yij)

        return p_yij_zi


class DSLikelihood(CrowdScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num,
                 worker_confusion_matrix: Union[list, np.ndarray],
                 **kwargs):
        super().__init__(latent_dim, label_num, task_num, worker_num, **kwargs)

        self.worker_confusion_matrix = Parameter(np.array(worker_confusion_matrix).reshape(worker_num, label_num, label_num), transform=sum_to_one())
        self.worker_confusion_matrix.prior = tfp.distributions.Beta(to_default_float(1.0), to_default_float(1.0))

    def calc_p_yij_zi(self, Y):
        """
        [N, M] -> [N, M, K]
        """
        K = self.K
        N = self.N
        M = self.M

        # zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        # zi1 = tf.tile(tf.reshape(zi, [N, 1, 1, 1, K]), [1, M, K, K, 1])
        #
        # Y = tf.tile(tf.reshape(Y, [N, M, 1, 1, 1]), [1, 1, K, K, K])
        #
        # conf_matrix = tf.tile(tf.reshape(self.worker_confusion_matrix, [1, M, K, K, 1]), [N, 1, 1, 1, K])  # [N, M, K, K, K]
        #
        # label_matrix = tf.convert_to_tensor(np.array(list(range(K))*K), dtype=default_float())  # [K, K]
        # label_matrix = tf.tile(tf.reshape(label_matrix, [1, 1, K, K, 1]), [N, M, 1, 1, K])
        #
        # p_yij_zi = tf.where(tf.equal(Y, label_matrix) & tf.equal(zi1, label_matrix), conf_matrix, 1)
        #
        # p_yij_zi = tf.reduce_prod(p_yij_zi, [2, 3])

        zi = tf.transpose(tf.convert_to_tensor(np.array([[i]*N for i in range(K)]), dtype=default_float()))  # [N, K]
        zi1 = tf.tile(tf.reshape(zi, [N, 1, K, 1, 1, ]), [1, M, 1, K, K])

        Y = tf.tile(tf.reshape(Y, [N, M, 1, 1, 1]), [1, 1, K, K, K])

        conf_matrix = tf.tile(tf.reshape(self.worker_confusion_matrix, [1, M, 1, K, K]), [N, 1, K, 1, 1])  # [N, M, K, K, K]

        label_matrix = tf.convert_to_tensor(np.array([list(range(K)) for i in range(K)]), dtype=default_float())  # [K, K]
        label_matrix = tf.tile(tf.reshape(label_matrix, [1, 1, 1, K, K]), [N, M, K, 1, 1])
        z_mask = tf.equal(zi1, label_matrix)  # column

        label_matrix_t = tf.convert_to_tensor(np.array([list(range(K)) for i in range(K)]).T, dtype=default_float())
        label_matrix_t = tf.tile(tf.reshape(label_matrix_t, [1, 1, 1, K, K]), [N, M, K, 1, 1])
        y_mask = tf.equal(Y, label_matrix_t)  # row

        p_yij_zi = tf.where(y_mask & z_mask, conf_matrix, 1)

        p_yij_zi = tf.reduce_prod(p_yij_zi, [-2, -1])


        return p_yij_zi


class GaussianBernoulliLikelihood(CrowdScalarLikelihood):
    def __init__(self, latent_dim, label_num, task_num, worker_num,
                 task_mean: Union[list, np.ndarray], task_var: Union[list, np.ndarray],
                 worker_mean: Union[list, np.ndarray], worker_var: Union[list, np.ndarray],
                 **kwargs):
        super().__init__(latent_dim, label_num, task_num, worker_num, **kwargs)

        self.task_mean = Parameter(np.array(task_mean).reshape(1, -1))
        self.task_var = Parameter(np.array(task_var).reshape(1, -1), transform=positive())
        self.worker_mean = Parameter(np.array(worker_mean).reshape(1, -1))
        self.worker_var = Parameter(np.array(worker_var).reshape(1, -1), transform=positive())

        self.task_mean.prior = tfp.distributions.Normal(loc=to_default_float(0.0), scale=to_default_float(1.0))
        self.task_var.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(1.0))
        self.worker_mean.prior = tfp.distributions.Normal(loc=to_default_float(0.0), scale=to_default_float(1.0))
        self.worker_mean.prior = tfp.distributions.Gamma(to_default_float(1.0), to_default_float(1.0))

    def log_prob(self, F, Y, Y_mask):
        """Calculate log likelihood. 只支持F维度为(N, 1), Y维度为(N, M)

                q(b_i^j) = N(fi, sigma_i^2 + sigma_j^2)

                p(y_i^j=Y|b_i^j) = inv_probit_likelihood

                then this method computes

                \log \prod_{i, j} \int p(y_i^j=Y|b_i^j)q(b_i^j) db_i^j

                """
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim
        n_gh = n_gh_new = self.num_gauss_hermite_points

        Tmu = self.task_mean
        Tvar = self.task_var
        Amu = self.worker_mean
        Avar = self.worker_var

        # step2: calculate logp(f_ik) = log p(yij|fi) = log \int p(bij|fi+mui+muj, sigma_i^2 + sigma_j^2) p(yij|bij) dbij
        # fi+mui+muj
        Fmu2 = tf.tile(tf.reshape(F, (N, 1, dim)), [1, M, 1])  # [N, M, dim]
        Fmu2 += tf.tile(tf.reshape(Tmu, (N, 1, 1)), [1, M, dim])
        Fmu2 += tf.tile(tf.reshape(Amu, (1, M, 1)), [N, 1, dim])

        # sigma_i^2 + sigma_j^2
        Fvar2 = tf.tile(tf.reshape(Tvar, (N, 1, 1)), [1, M, dim])
        Fvar2 += tf.tile(tf.reshape(Avar, (1, M, 1)), [N, 1, dim])

        # quadrature
        Fmu3, W3 = self.calc_quadrature(Fmu2, Fvar2)  # X: [n_gh, N, M, dim], w: [ngh, 1, 1, 1]

        # tile Y and Y_mask
        Y_mask_tiled = tf.tile(tf.reshape(Y_mask, (1, N, M, 1)),
                               [n_gh, 1, 1, dim])  # [n_gh, N, M, dim]
        Y_tiled = tf.tile(tf.reshape(Y, (1, N, M, 1)), [n_gh, 1, 1, dim])  # [n_gh, N, M, dim]

        # log p(yij|fi) = log p(y_ijkl|f_ijkl)
        density = inv_probit_likelihood(F=Fmu3, Y=Y_tiled)
        log_density = tf.math.log(density)

        # w_ijkl * log p(y_ijkl|f_ijkl) * ymask_ijkl
        ve = W3 * log_density * Y_mask_tiled
        ve = tf.reduce_sum(ve, [0, 1, 2])  # [n_gh, N, M, dim] -> [dim]

        return ve

    def _variational_expectations(self, Fmu, Fvar, Y, Y_mask):
        """
        calculate ∫log(p(y|f))q(f)df
        注意这里的Fmu, Fvar的每个分量fi，表示变分高斯分布的边际分布的均值和协方差  维度都是 N x dim，N为样本数，dim为latent function个数
        Fmu [N, 1]
        Fvar [N, 1]
        Y: [N, M]
        Y_mask: [N, M]
        """
        K = self.K
        N = self.N
        M = self.M
        dim = self.latent_dim
        n_gh = n_gh_new = self.num_gauss_hermite_points

        Tmu = self.task_mean
        Tvar = self.task_var
        Amu = self.worker_mean
        Avar = self.worker_var

        Fmu1, W1 = self.calc_quadrature(Fmu, Fvar)  # x: [n_gh, N, dim], w: [n_gh, 1, 1]

        # calculate logp(f_ik) = log p(yij|fi) = log \int p(bij|fi+mui+muj, sigma_i^2 + sigma_j^2) p(yij|bij) dbij
        # fi+mui+muj
        Fmu2 = tf.tile(tf.reshape(Fmu1, (n_gh, N, 1, dim)), [1, 1, M, 1])  # [n_gh, N, M, dim]
        Fmu2 += tf.tile(tf.reshape(Tmu, (1, N, 1, 1)), [n_gh, 1, M, dim])
        Fmu2 += tf.tile(tf.reshape(Amu, (1, 1, M, 1)), [n_gh, N, 1, dim])
        # sigma_i^2 + sigma_j^2
        Fvar2 = tf.tile(tf.reshape(Tvar, (1, N, 1, 1)), [n_gh, 1, M, dim])
        Fvar2 += tf.tile(tf.reshape(Avar, (1, 1, M, 1)), [n_gh, N, 1, dim])

        Fmu3, W3 = self.calc_quadrature(Fmu2, Fvar2)  # X: [n_gh_new, n_gh, N, M, dim], w: [ngh_new, 1, 1, 1, 1]

        # log p(yij|fi) = log p(y_ijkl|f_ijkl)
        Y_tiled = tf.tile(tf.reshape(Y, (1, 1, N, M, 1)), [n_gh_new, n_gh, 1, 1, dim])  # [n_gh_new, n_gh, N, M, dim]
        p_yij_fi = inv_probit_likelihood(F=Fmu3, Y=Y_tiled)
        log_p_yij_fi = tf.math.log(p_yij_fi)

        # w_ijkl * log p(y_ijkl|f_ijkl) * ymask_ijkl
        Y_mask_tiled = tf.tile(tf.reshape(Y_mask, (1, 1, N, M, 1)), [n_gh_new, n_gh, 1, 1, dim])  # [n_gh_new, n_gh, N, M, dim]
        ve = W3 * log_p_yij_fi * Y_mask_tiled
        ve = tf.reduce_sum(ve, [0, 3])  # [n_gh_new, n_gh, N, M, dim] -> [n_gh, N, dim]

        ve = W1 * ve
        ve = tf.reduce_sum(ve, [0, 1])  # [n_gh, N, dim] -> [dim]

        return ve


class LRMeanFunction(MeanFunction):
    """
    Only for cases where latent function number = 1.
    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def __init__(self, model_file):
        MeanFunction.__init__(self)
        if isinstance(model_file, str):
            self.model = joblib.load(model_file)
        else:
            self.model = model_file
        self.rel_col = list(self.model.classes_.flatten()).index(RELEVANT)

    def __call__(self, X):
        row_num = X.shape[0]

        X = np.array(X)
        log_probas = self.model.predict_log_proba(X)
        F = log_probas[:, self.rel_col]
        F.reshape(row_num, -1)  # does not work
        F = tf.convert_to_tensor(F, dtype=default_float())
        F = tf.reshape(F, (row_num, -1))
        return F


class VGPWithAnnotationData(GPModel, InternalDataTrainingLossMixin):
    def __init__(self,
                 data: AnnotationData,
                 kernel: Kernel,
                 likelihood: ScalarLikelihood,
                 mean_function: Optional[MeanFunction] = None):
        self.data = data_input_to_tensor(data)
        X_data, Y_data, _ = self.data
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        num_data = X_data.shape[0]
        self.num_data = num_data

        self.q_mu = Parameter(np.zeros((num_data, self.num_latent_gps)))
        # self.q_mu = Parameter(np.random.randn(num_data, self.num_latent_gps))
        q_sqrt = np.array([np.eye(num_data) for _ in range(self.num_latent_gps)])
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())
        # self.q_z = Parameter(np.full(shape=(num_data, self.num_latent_gps), fill_value=0.5), transform=min_zero_max_one())  # 0.5 for binary labels

    # def log_prior_density(self) -> tf.Tensor:
    #     """
    #     Sum of the log prior probability densities of all (constrained) variables in this model.
    #     """
    #     return to_default_float(0.0)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        r"""
        This method computes the variational lower bound on the likelihood,
            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]
        """

        X_data, Y_data, Y_mask = self.data  # modified @20200902
        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kernel(X_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(K)
        fmean = tf.linalg.matmul(L, self.q_mu) + self.mean_function(X_data)  # [N,N] matmul [N,dim] -> [N, dim]
        q_sqrt_dnn = tf.linalg.band_part(self.q_sqrt, -1, 0)  # [dim, N, N]
        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent_gps, 1, 1]))  # [dim, N, N]
        LTA = tf.linalg.matmul(L_tiled, q_sqrt_dnn)  # [dim, N, N]
        fvar = tf.reduce_sum(tf.square(LTA), 2)  # [dim, N]

        fvar = tf.transpose(fvar)  # [N, dim]

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y_data, Y_mask)  # modified @20200902
        # var_exp = self.likelihood.z_f_variational_expectations(self.q_z, fmean, fvar, Y_data, Y_mask)

        return tf.reduce_sum(var_exp) - KL

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        X_data, _, _ = self.data
        mu, var = conditional(Xnew, X_data, self.kernel, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov, white=True)
        return mu + self.mean_function(Xnew), var

    def predict_z(self):
        return self.q_z


class GPMCWithAnnotationData(GPModel, InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: AnnotationData,
            kernel: Kernel,
            likelihood: ScalarLikelihood,
            mean_function: Optional[MeanFunction] = None,
            num_latent_gps: Optional[int] = None,
    ):
        """
        data is a tuple of X, Y with X, a data matrix, size [N, D] and Y, a data matrix, size [N, R]
        kernel, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        self.data = data_input_to_tensor(data)
        X_data, Y_data, _ = self.data
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        num_data = X_data.shape[0]
        self.num_data = num_data
        self.V = Parameter(np.zeros((self.num_data, self.num_latent_gps)))
        self.V.prior = tfp.distributions.Normal(
            loc=to_default_float(0.0), scale=to_default_float(1.0)

        )

    def log_posterior_density(self) -> tf.Tensor:
        return self.log_likelihood() + self.log_prior_density()

    def _training_loss(self) -> tf.Tensor:
        return -self.log_posterior_density()

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_likelihood()

    def log_likelihood(self) -> tf.Tensor:
        r"""
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y | V, theta).

        """
        X_data, Y_data, Y_mask = self.data  # modified @20220103
        K = self.kernel(X_data)
        L = tf.linalg.cholesky(
            K + tf.eye(tf.shape(X_data)[0], dtype=default_float()) * default_jitter()
        )
        F = tf.linalg.matmul(L, self.V) + self.mean_function(X_data)

        # var_exp = self.likelihood.variational_expectations(fmean, fvar, Y_data, Y_mask)  # modified @20200902
        # var_exp = self.likelihood.z_f_variational_expectations(self.q_z, fmean, fvar, Y_data, Y_mask)

        return tf.reduce_sum(self.likelihood.log_prob(F, Y_data))

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        X_data, Y_data = self.data
        mu, var = conditional(
            Xnew, X_data, self.kernel, self.V, full_cov=full_cov, q_sqrt=None, white=True
        )
        return mu + self.mean_function(Xnew), var


class LIKELIHOOD(GPModel, InternalDataTrainingLossMixin):

    def __init__(
        self,
        data: AnnotationDataWithoutInput,
        likelihood: GaussianBernoulliLikelihood,
        kernel: Optional[Kernel] = None,
        mean_function: Optional[MeanFunction] = None
    ):

        self.data = data_input_to_tensor(data)
        self.num_data = self.data[0].shape[0]
        num_latent_gps = 1
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.F = Parameter(np.zeros((self.num_data, num_latent_gps)))  # 代表隐变量f，每次HMC调用sample方法采样一个V的值。
        self.F.prior = tfp.distributions.Normal(
            loc=to_default_float(0.0), scale=to_default_float(1.0)
        )

    def log_posterior_density(self) -> tf.Tensor:
        return self.log_likelihood() + self.log_prior_density()

    def _training_loss(self) -> tf.Tensor:
        return -self.log_posterior_density()

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_likelihood()

    def log_likelihood(self) -> tf.Tensor:
        r"""
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y | V, theta).
        """
        Y_data, Y_mask = self.data
        return tf.reduce_sum(self.likelihood.log_prob(self.F, Y_data, Y_mask))

    def predict_f(self):
        return self.F


def sgd(opt, training_loss, m):
    opt.minimize(training_loss, m.trainable_variables)
    return


def vem_fixkernelparam(opt, training_loss, m):
    # update kernel params
    set_trainable(m.kernel, False)

    # update q(f)
    set_trainable(m.likelihood, False)
    set_trainable(m.q_mu, True)
    set_trainable(m.q_sqrt, True)
    opt.minimize(training_loss, m.trainable_variables)

    # update likelihood param
    set_trainable(m.likelihood, True)
    set_trainable(m.q_mu, False)
    set_trainable(m.q_sqrt, False)
    opt.minimize(training_loss, m.trainable_variables)

    return


def vem(opt, training_loss, m):
    # update kernel params
    set_trainable(m.kernel, True)
    set_trainable(m.likelihood, False)
    set_trainable(m.q_mu, False)
    set_trainable(m.q_sqrt, False)
    opt.minimize(training_loss, m.trainable_variables)

    # update q(f)
    set_trainable(m.kernel, False)
    set_trainable(m.likelihood, False)
    set_trainable(m.q_mu, True)
    set_trainable(m.q_sqrt, True)
    opt.minimize(training_loss, m.trainable_variables)

    # update likelihood param
    set_trainable(m.kernel, False)
    set_trainable(m.likelihood, True)
    set_trainable(m.q_mu, False)
    set_trainable(m.q_sqrt, False)
    opt.minimize(training_loss, m.trainable_variables)

    return


def gp_aggregation(train_x, train_y, train_y_mask, test_x, active_dims,
                   mean_function_name='zero', mean_function_path=None,
                   likelihood_name='gpcrowd',
                   gpcrowd_noise=0.1,  # only for gpcrowd
                   epoch_num=100,
                   optimizor_name='adam',
                   learning_rate=0.001,
                   param_update='VEM',
                   threshold=1e-4,
                   random_seed=0,
                   crowdlabeled_data=None,
                   enable_logging=True, log_path=None, logging_freq=10):

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # kernel
    if 'rbf' in active_dims.keys() and 'linear' not in active_dims.keys():
        kern = gpflow.kernels.RBF(variance=1.0, lengthscales=[1.0]*len(active_dims['rbf']), active_dims=active_dims['rbf'])  # ARD默认为False, the kernel has a single length-scale.
        kern.variance.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))
        kern.lengthscales.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))

    elif 'linear' in active_dims.keys() and 'rbf' not in active_dims.keys():
        kern = gpflow.kernels.Linear(variance=1.0, active_dims=active_dims['linear'])
        kern.variance.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))

    elif 'rbf' in active_dims.keys() and 'linear' in active_dims.keys():
        kern1 = gpflow.kernels.RBF(variance=1.0, lengthscales=[1.0]*len(active_dims['rbf']), active_dims=active_dims['rbf'])  # ARD默认为False, the kernel has a single length-scale.
        kern1.variance.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))
        kern1.lengthscales.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))

        kern2 = gpflow.kernels.Linear(variance=1.0, active_dims=active_dims['linear'])
        kern2.variance.prior = tfp.distributions.Gamma(to_default_float(1), to_default_float(1))

        kern = kern1 + kern2
    else:
        raise NotImplementedError

    # likelihood
    latent_dim = 1
    K = 2
    N = train_y.shape[0]
    M = train_y.shape[1]
    if likelihood_name == 'gpglad':
        lik = GLADLikelihood(latent_dim=latent_dim, label_num=K, task_num=N, worker_num=M,
                             task_difficulty=np.full(N, 1.0),
                             worker_competence=np.full(M, 1.0))

    elif likelihood_name == 'gpmace':
        lik = MACELikelihood(latent_dim=latent_dim, label_num=K, task_num=N, worker_num=M,
                             worker_spamming=np.full(M, 0.5),   # annotator is spamming half the chance
                             worker_labelling_dist=np.full([M, K], 1.0/K))

    elif likelihood_name == 'gpzc':
        correct_p = 0.7
        lik = ZCLikelihood(latent_dim=latent_dim, label_num=K, task_num=N, worker_num=M,
                           worker_competence=np.full(M, correct_p))   # annotator gives correct label half the chance

    elif likelihood_name == 'gpds':
        correct_p = 0.7
        incorrect_p = (1 - correct_p) / (K-1)
        cf = np.full([K, K], incorrect_p)
        row, col = np.diag_indices_from(cf)
        cf[row, col] = correct_p
        cf = np.array([cf for i in range(M)])
        lik = DSLikelihood(latent_dim=latent_dim, label_num=K, task_num=N, worker_num=M,
                           worker_confusion_matrix=cf)

    elif likelihood_name == 'gpcrowd':
        lik = GaussianBernoulliLikelihood(latent_dim=latent_dim, label_num=K, task_num=N, worker_num=M,
                                          task_mean=[0.0] * N, task_var=[gpcrowd_noise] * N,
                                          worker_mean=[0.0] * M, worker_var=[gpcrowd_noise] * M)
    else:
        raise NotImplementedError

    # mean function
    feature_dim = train_x.shape[1]
    if mean_function_name == 'pretrain':
        mean_function = LRMeanFunction(mean_function_path)
    elif mean_function_name == 'linear':
        mean_function = Linear(A=np.random.randn(feature_dim, 1), b=np.random.randn(1, 1))
    elif mean_function_name == 'zero':
        mean_function = Zero()
    elif mean_function_name == 'logisticregression':
        mv_y = mv_aggregation(train_y, train_y_mask)['pred_y'].reshape(-1, 1)
        lrmodel = LogisticRegression(random_state=random_seed)
        lrmodel.fit(train_x, mv_y)
        mean_function = LRMeanFunction(model_file=lrmodel)
    else:
        raise NotImplementedError

    # model
    m = VGPWithAnnotationData(data=(train_x, train_y, train_y_mask),
                            kernel=kern, likelihood=lik, mean_function=mean_function)

    # log
    model_task = ModelToTensorBoard(log_path, m)
    lml_task = ScalarToTensorBoard(log_path, lambda: m.training_loss(), "training_loss")
    loglikelihood_task = ScalarToTensorBoard(log_path, lambda: m.maximum_log_likelihood_objective(), "log_likelihood")
    logprior_task = ScalarToTensorBoard(log_path, lambda: m.log_prior_density(), "log_prior_density")
    fast_tasks = MonitorTaskGroup([model_task, lml_task, loglikelihood_task, logprior_task], period=1)
    fast_monitor = Monitor(fast_tasks)

    if enable_logging:
        eval_label_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'acc'), 'acc'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'auc'), 'auc'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'posi_f1'), 'posi_f1'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'nega_f1'), 'nega_f1'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'tn'), 'tn'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'fp'), 'fp'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'fn'), 'fn'),
                            ScalarToTensorBoard(log_path, lambda: evaluate_label_for_gpla(crowdlabeled_data, m, 'tp'), 'tp')
                            ]

        if likelihood_name == 'gpglad':
            eval_likelihood_param_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'task_difficulty'), 'task_difficulty_mse'),
                                           ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'worker_competence'), 'worker_competence_mse')]

        elif likelihood_name == 'gpmace':
            eval_likelihood_param_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'worker_spamming'), 'worker_spamming_mse'),
                                           ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'worker_labelling_dist'), 'worker_labelling_dist_mse')]

        elif likelihood_name == 'gpzc':
            eval_likelihood_param_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'worker_competence'), 'worker_competence_mse')]

        elif likelihood_name == 'gpds':
            eval_likelihood_param_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(crowdlabeled_data, m, 'worker_confusion_matrix'), 'worker_confusion_matrix_mse')]

        elif likelihood_name == 'gpcrowd':
            # eval_likelihood_param_tasks = [ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(test_data, m, 'task_mean'), 'task_mean_mse'),
            #                                ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(test_data, m, 'task_var'), 'task_var_mse'),
            #                                ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(test_data, m, 'worker_mean'), 'worker_mean_mse'),
            #                                ScalarToTensorBoard(log_path, lambda: evaluate_likelihood_param_for_gpla(test_data, m, 'worker_var'), 'worker_var_mse')]
            eval_likelihood_param_tasks = []
        else:
            raise NotImplementedError

        slow_tasks = MonitorTaskGroup(eval_label_tasks+eval_likelihood_param_tasks, period=int(logging_freq))
        slow_monitor = Monitor(slow_tasks)


    # optimize
    training_loss = m.training_loss_closure(compile=True)
    if optimizor_name == 'adam':
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizor_name == 'adagrad':
        opt = tf.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizor_name == 'adadelta':
        opt = tf.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizor_name == 'rmsprop':
        opt = tf.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizor_name == 'scipy':
        opt = gpflow.optimizers.Scipy()
    else:
        raise NotImplementedError

    loss = m.training_loss().numpy()
    for epoch in tqdm(range(epoch_num), desc='Epoch'):

        # update params
        if param_update == 'SGD':
            sgd(opt, training_loss, m)
        elif param_update == 'VEM':
            vem(opt, training_loss, m)

        # if stop training
        last_loss = loss
        loss = m.training_loss().numpy()
        print('loss', loss, 'ratio', np.abs((loss-last_loss)/last_loss))
        if (np.abs((loss-last_loss)/last_loss)) < threshold:
            break

        # logging
        if enable_logging:
            fast_monitor(epoch)
            slow_monitor(epoch)

    # predict
    mean, var = m.predict_y(test_x)

    pred_dct = {}
    # for evaluation
    pred_dct['pred_y'] = prob_to_class(mean.numpy().flatten(), 0.5)
    pred_dct['pred_y_score'] = mean.numpy().flatten()

    # for logging
    # label
    pred_dct['pred_mean'] = mean.numpy()
    pred_dct['pred_var'] = var.numpy()
    # parameters
    for model in [m.likelihood, m.kernel]:
        tempdct = parameter_dict(model)
        for key in tempdct:
            pred_dct[key[1:]] = tempdct[key].numpy()
    pred_dct['q_mu'] = m.q_mu.numpy()
    pred_dct['q_sqrt'] = m.q_sqrt.numpy()

    return pred_dct


def likelihood_aggregation(train_y, train_y_mask,
                           log_path=None,
                           epoch_num=100,
                           noise_level=0.1,
                           random_seed=0
                           ):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # likelihood
    task_num = train_y.shape[0]
    worker_num = train_y.shape[1]
    lik = GaussianBernoulliLikelihood(task_mean=[0] * task_num, task_var=[noise_level] * task_num,
                               worker_mean=[0] * worker_num, worker_var=[noise_level] * worker_num)

    # model
    m = LIKELIHOOD(data=(train_y, train_y_mask), kernel=None, likelihood=lik, mean_function=None)

    # log
    model_task = ModelToTensorBoard(log_path, m)
    lml_task = ScalarToTensorBoard(log_path, lambda: m.training_loss(), "training_objective")
    fast_tasks = MonitorTaskGroup([model_task, lml_task], period=1)
    monitor = Monitor(fast_tasks)

    # optimize
    training_loss = m.training_loss_closure(compile=True)
    opt = tf.optimizers.Adam()

    for step in range(epoch_num):
        opt.minimize(training_loss, m.trainable_variables)
        monitor(step)

    # predict
    F = m.predict_f()
    pred_dct = {}
    pred_dct['pred_y'] = prob_to_class(F.numpy().flatten(), 0)
    pred_dct['pred_td'] = list(m.likelihood.task_var.numpy().flatten())
    pred_dct['pred_task_mean'] = list(m.likelihood.task_mean.numpy().flatten())
    pred_dct['pred_ac'] = list(m.likelihood.worker_var.numpy().flatten())
    pred_dct['pred_worker_mean'] = list(m.likelihood.worker_mean.numpy().flatten())

    return pred_dct


def gpmv_aggregation(crowd_x, crowd_y, mask_y, test_x, active_dims,
                     mean_function_name, pretrain_mean_function_path=None,
                     log_path=None,
                     epoch_num=100,
                     random_seed=0
                     ):

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    # majority voting
    aggr_y = []
    for y, mask in zip(crowd_y, mask_y):
        ym = [yi for yi, mi in zip(y, mask) if mi == 1]
        assert len(ym) != 0
        counter = Counter(ym)
        if counter[RELEVANT] >= counter[NON_RELEVANT]:
            label = RELEVANT
        else:
            label = NON_RELEVANT
        aggr_y.append(label)
    aggr_y = np.reshape(np.array(aggr_y, dtype=np.int), (-1, 1))

    # kernel  ARD默认为False, the kernel has a single length-scale (ARD=False).
    if 'rbf' in active_dims.keys() and 'linear' not in active_dims.keys():
        kern = gpflow.kernels.RBF(active_dims=active_dims['rbf'])

    elif 'linear' in active_dims.keys() and 'rbf' not in active_dims.keys():
        kern = gpflow.kernels.Linear(active_dims=active_dims['linear'])

    elif 'rbf' in active_dims.keys() and 'linear' in active_dims.keys():
        kern = gpflow.kernels.RBF(active_dims=active_dims['rbf']) + \
               gpflow.kernels.Linear(active_dims=active_dims['linear'])
    else:
        raise NotImplementedError

    # mean function
    dim = crowd_x.shape[1]
    if mean_function_name == 'pretrain':
        mean_function = LRMeanFunction(pretrain_mean_function_path)
    elif mean_function_name == 'linear':
        mean_function = Linear(A=np.random.randn(dim, 1), b=np.random.randn(1, 1))
    elif mean_function_name == 'zero':
        mean_function = Zero()
    else:
        raise NotImplementedError

    # likelihood
    lik = gpflow.likelihoods.Bernoulli()

    # model
    m = gpflow.models.VGP(data=(crowd_x, aggr_y),
                          kernel=kern, likelihood=lik, mean_function=mean_function, num_latent_gps=1)

    # log
    model_task = ModelToTensorBoard(log_path, m)
    lml_task = ScalarToTensorBoard(log_path, lambda: m.training_loss(), "training_objective")
    fast_tasks = MonitorTaskGroup([model_task, lml_task], period=1)
    monitor = Monitor(fast_tasks)

    # optimize
    training_loss = m.training_loss_closure(compile=True)
    opt = tf.optimizers.Adam()

    for step in range(epoch_num):
        opt.minimize(training_loss, m.trainable_variables)
        monitor(step)

    # predict
    mean, var = m.predict_y(test_x)
    pred_dct = {}
    pred_dct['pred_y'] = prob_to_class(mean, 0.5)
    pred_dct['pred_mean'] = mean.numpy().flatten()
    pred_dct['pred_var'] = var.numpy().flatten()

    return pred_dct


if __name__ == '__main__':

    pass
