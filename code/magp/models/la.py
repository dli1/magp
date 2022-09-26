
import sys
import random
import math
from tqdm import tqdm
import tensorflow as tf
from scipy.optimize import minimize

from magp.utils.eval_metrics import *


class LA(object):
    def __init__(self, crowd_tuples, csd_data, log_path, logging_freq, enable_logging):
        # for optimization
        self.e2wl, self.w2el, self.label_set = self.gete2wlandw2el(crowd_tuples)
        assert RELEVANT in self.label_set and NON_RELEVANT in self.label_set
        self.rel_label, nonrel_label = RELEVANT, NON_RELEVANT
        self.e2lpd = {}

        # for prediction, evaluation, logging
        self.csd_data = csd_data
        self.logging_freq = logging_freq
        self.enable_logging = enable_logging
        if enable_logging:
            self.log_writer = tf.summary.create_file_writer(log_path)

    def gete2wlandw2el(self, crowd_tuples):
        e2wl = {}
        w2el = {}
        label_set=[]

        for line in crowd_tuples:
            example, worker, label = line
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker,label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example,label])

            if label not in label_set:
                label_set.append(label)

        return e2wl,w2el,label_set

    def get_e2truth(self, e2lpd):

        e2truth = {}
        for e in e2lpd:
            if type(e2lpd[e]) == type({}):
                temp = 0
                for label in e2lpd[e]:
                    if temp < e2lpd[e][label]:
                        temp = e2lpd[e][label]

                candidate = []

                for label in e2lpd[e]:
                    if temp == e2lpd[e][label]:
                        candidate.append(label)

                truth = random.choice(candidate)

            else:
                truth = e2lpd[e]

            e2truth[e] = truth

        return e2truth

    def get_e2truth_prob(self, e2lpd):
        e2truth_prob = {}
        for e in e2lpd:
            if type(e2lpd[e]) == type({}):
                rel_prob = e2lpd[e][self.rel_label]
            else:
                rel_prob = e2lpd[e]
            e2truth_prob[e] = rel_prob

        return e2truth_prob

    def predict_label(self):
        pred_dct = {}

        task_ids = self.csd_data['task_ids']

        e2truth = self.get_e2truth(self.e2lpd)
        pred_dct['pred_y'] = np.array([e2truth[task] for task in task_ids])

        e2turth_prob = self.get_e2truth_prob(self.e2lpd)
        pred_dct['pred_y_score'] = np.array([e2turth_prob[task] for task in task_ids])

        return pred_dct

    def predict_params(self):
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    def logging(self, epoch):
        with self.log_writer.as_default():
            tf.summary.scalar('log_likelihood', self.log_likelihood(), epoch)

            pred_label_data = self.predict_label()
            for measure in ['acc', 'auc', 'posi_f1', 'nega_f1', 'tn', 'fp', 'fn', 'tp']:
                value = evaluate_label_for_la(self.csd_data, pred_label_data, measure)
                tf.summary.scalar(measure, value, epoch)

            pred_param_data = self.predict_params()
            for param in pred_param_data.keys():
                value = evaluate_likelihood_param_for_la(self.csd_data, pred_param_data, param)
                tf.summary.scalar(param+'_mse', value, epoch)


class DS(LA):
    def __init__(self, crowd_tuples, csd_data, log_path, logging_freq, enable_logging, initquality=0.7):
        super().__init__(crowd_tuples, csd_data, log_path, logging_freq, enable_logging)

        self.workers = self.w2el.keys()
        self.initalquality = initquality  # default diagonal value for confusion matrix

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}

        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]

                lpd[tlabel] = weight
                total_weight += weight

            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight

            self.e2lpd[example] = lpd

    #M-step
    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0

        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0/len(self.e2lpd)

    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:

                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (1-self.initalquality)*1.0/(len(self.label_set)-1)

                    continue

                for example, label in self.w2el[w]:

                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel]*1.0/w2lweights[w][tlabel]  # this line seems incorrect

        return self.w2cm

    #initialization
    def Init_l2pd(self):
        #uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0/len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1-self.initalquality)/(len(self.label_set)-1)

        return w2cm

    def Run(self, epoch_num, threshold):

        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()

        self.Update_e2lpd()
        llh = self.log_likelihood()

        for epoch in tqdm(range(epoch_num), desc='Epoch'):
            # E-step: update zi
            self.Update_e2lpd()

            # M-step: update confusion matrix and posterior p(zi | Y)
            self.Update_l2pd()
            self.Update_w2cm()

            # if stop training
            last_llh = llh
            llh = self.log_likelihood()
            # q = self.q_function()
            # print('lk', loglikelihood)
            # print('q', q)
            if (np.abs((llh-last_llh)/last_llh)) < threshold:
                break
            if self.enable_logging:
                if int(epoch) % int(self.logging_freq) == 0:
                    self.logging(epoch)

        return

    def log_likelihood(self):
        """
        log p(Y) = sum_i log sum_zi {p(zi) prod_j p(y_ij | zi)}.
        Note p(zi) here means the prior distribution of zi.
        """

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]  # p(y_ij==label | zi==tlabel)
                temp += inner

            lh += math.log(temp)

        return lh

    def q_function(self):
        """
        Q = sum_i sum_zi p(zi|data) log {p(zi) prod_j p(y_ij | zi)}.
        """
        q = 0

        for example, worker_label_set in self.e2wl.items():
            pi = 0
            for tlabel, p_zi_post in self.e2lpd[example].items():

                p_yi_zi = p_zi_prior = self.l2pd[tlabel]
                for worker, label in worker_label_set:
                    p_yi_zi *= self.w2cm[worker][tlabel][label]
                log_p_yi_zi = math.log(p_yi_zi)

                pi += log_p_yi_zi * p_zi_post

            q += pi

        return q

    def predict_params(self):
        pred_dct = {}
        worker_ids = self.csd_data['worker_ids']
        cm = np.zeros(shape=[len(worker_ids), len(self.label_set), len(self.label_set)])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                for k in range(cm.shape[2]):
                    cm[i][j][k] = self.w2cm[worker_ids[i]][self.label_set[j]][self.label_set[k]]
        pred_dct['worker_confusion_matrix'] = cm

        return pred_dct


class ZC(LA):
    def __init__(self, crowd_tuples, csd_data, log_path, logging_freq, enable_logging):
        super().__init__(crowd_tuples, csd_data, log_path, logging_freq, enable_logging)

        self.l2pd = None
        self.e2lpd = None
        self.wm = None

    def InitPj(self):
        l2pd={}
        for label in self.label_set:
            l2pd[label]=1.0/len(self.label_set)
        return l2pd

    def InitWM(self, workers={}):
        wm={}

        if workers=={}:
            workers=self.w2el.keys()
            for worker in workers:
                wm[worker]=0.8
        else:
            for worker in workers:
                if worker not in wm: # workers --> wm
                    wm[worker] = 0.8
                else:
                    wm[worker]=workers[worker]

        return wm

    #E-step
    def ComputeTij(self):
        e2lpd={}
        for e, workerlabels in self.e2wl.items():
            e2lpd[e]={}
            for label in self.label_set:
                e2lpd[e][label]=self.l2pd[label]  #1.0#l2pd[label]

            for worker,label in workerlabels:
                for candlabel in self.label_set:
                    if label==candlabel:
                        e2lpd[e][candlabel]*=self.wm[worker]
                    else:
                        e2lpd[e][candlabel]*=(1-self.wm[worker])*1.0/(len(self.label_set)-1)

            sums=0
            for label in self.label_set:
                sums+=e2lpd[e][label]

            if sums==0:
                for label in self.label_set:
                    e2lpd[e][label]=1.0/len(self.label_set)
            else:
                for label in self.label_set:
                    e2lpd[e][label]=e2lpd[e][label]*1.0/sums

        self.e2lpd = e2lpd
        return


    #M-step
    def ComputePj(self):
        l2pd = {}

        for label in self.label_set:
            l2pd[label]=0
        for e in self.e2lpd:
            for label in self.e2lpd[e]:
                l2pd[label]+=self.e2lpd[e][label]

        for label in self.label_set:
            l2pd[label]=l2pd[label]*1.0/len(self.e2lpd)

        self.l2pd = l2pd
        return


    def ComputeWM(self):
        wm={}
        for worker,examplelabels in self.w2el.items():
            wm[worker]=0.0
            for e,label in examplelabels:
                wm[worker]+=self.e2lpd[e][label]*1.0/len(examplelabels)
        self.wm = wm
        return


    def Run(self, epoch_num, threshold):
        """
        #wm     worker_to_confusion_matrix = {}
        #e2lpd   example_to_softlabel = {}
        #l2pd   label_to_priority_probability = {}
        """

        self.l2pd = self.InitPj()
        self.wm = self.InitWM()


        self.ComputeTij()
        self.ComputePj()
        self.ComputeWM()
        llh = self.log_likelihood()
        for epoch in tqdm(range(epoch_num), desc='Epoch'):
            #E-step
            self.ComputeTij()

            #M-step
            self.ComputePj()
            self.ComputeWM()

            # if stop training
            last_llh = llh
            llh = self.log_likelihood()
            # print(llh)
            if (np.abs((llh-last_llh)/last_llh)) < threshold:
                break
            if self.enable_logging:
                if int(epoch) % int(self.logging_freq) == 0:
                    self.logging(epoch)

        return

    def log_likelihood(self):
        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    p = self.wm[worker] if tlabel == label else 1 - self.wm[worker]   # p(y_ij==label | zi==tlabel)
                    inner *= p
                temp += inner
            try:
                logtemp = math.log(temp)
            except:
                print('temp={}'.format(temp))
                print(self.l2pd, worker_label_set)
                logtemp = 0
            lh += logtemp

        return lh

    def predict_params(self):
        pred_dct = {}
        worker_ids = self.csd_data['worker_ids']
        worker_competence = np.zeros(shape=len(worker_ids))
        for i in range(worker_competence.shape[0]):
            worker_competence[i] = self.wm[worker_ids[i]]
        pred_dct['worker_competence'] = worker_competence

        return pred_dct



class GLAD(LA):
    def __init__(self, crowd_tuples, csd_data, logging_freq, log_path, enable_logging):
        super().__init__(crowd_tuples, csd_data, logging_freq, log_path, enable_logging)

        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()

    def sigmoid(self,x):
        if (-x)>math.log(sys.float_info.max):
            return 0
        if (-x)<math.log(sys.float_info.min):
            return 1

        return 1/(1+math.exp(-x))

    def logsigmoid(self,x):
        # For large negative x, -log(1 + exp(-x)) = x
        if (-x)>math.log(sys.float_info.max):
            return x
        # For large positive x, -log(1 + exp(-x)) = 0
        if (-x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(-x))
        #if (math.isinf(value)):
        #    return x

        return value

    def logoneminussigmoid(self,x):
        # For large positive x, -log(1 + exp(x)) = -x
        if (x)>math.log(sys.float_info.max):
            return -x
        # For large negative x, -log(1 + exp(x)) = 0
        if (x)<math.log(sys.float_info.min):
            return 0

        value = -math.log(1+math.exp(x))
        #if (math.isinf(value)):
        #    return -x

        return value

    def kronecker_delta(self,answer,label):
        if answer==label:
            return 1
        else:
            return 0

    def expbeta(self,beta):
        if beta>=math.log(sys.float_info.max):
            return sys.float_info.max
        else:
            return math.exp(beta)

    #E step
    def Update_e2lpd(self):
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.prior.items():
                weight = math.log(prob)
                for (worker, label) in worker_label_set:
                    logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    weight = weight + delta*logsigma + (1-delta)*(logoneminussigma-math.log(len(self.label_set)-1))

                if weight<math.log(sys.float_info.min):
                    lpd[tlabel] = 0
                else:
                    lpd[tlabel] = math.exp(weight)
                total_weight = total_weight + lpd[tlabel]

            for tlabel in lpd:
                if total_weight == 0:
                    lpd[tlabel] = 1.0/len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel]*1.0/total_weight

            self.e2lpd[example] = lpd

    #M_step
    def gradientQ(self):

        self.dQalpha={}
        self.dQbeta={}

        for example, worker_label_set in self.e2wl.items():
            dQb = 0
            for (worker, label) in worker_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQb = dQb + self.e2lpd[example][tlabel]*(delta-sigma)*self.alpha[worker]*self.expbeta(self.beta[example])
            self.dQbeta[example] = dQb - (self.beta[example] - self.priorbeta[example])

        for worker, example_label_set in self.w2el.items():
            dQa = 0
            for (example, label) in example_label_set:
                for tlabel in self.prior.keys():
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label,tlabel)
                    dQa = dQa + self.e2lpd[example][tlabel]*(delta-sigma)*self.expbeta(self.beta[example])
            self.dQalpha[worker] = dQa - (self.alpha[worker] - self.prioralpha[worker])

    def computeQ(self):

        Q = 0
        # the expectation of examples given priors, alpha and beta
        for worker, example_label_set in self.w2el.items():
            for (example, label) in example_label_set:
                logsigma = self.logsigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                logoneminussigma = self.logoneminussigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                for tlabel in self.prior.keys():
                    delta = self.kronecker_delta(label,tlabel)
                    Q = Q + self.e2lpd[example][tlabel]*(delta*logsigma+(1-delta)*(logoneminussigma-math.log(len(self.label_set)-1)))

        # the expectation of the sum of priors over all examples
        for example in self.e2wl.keys():
            for tlabel, prob in self.prior.items():
                Q = Q + self.e2lpd[example][tlabel] * math.log(prob)
        # Gaussian (standard normal) prior for alpha
        for worker in self.w2el.keys():
            Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))
        # Gaussian (standard normal) prior for beta
        for example in self.e2wl.keys():
            Q = Q + math.log((pow(2*math.pi,-0.5)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))
        return Q

    def optimize_f(self,x):
        # unpack x
        i=0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        return -self.computeQ() #Flip the sign since we want to minimize

    def optimize_df(self,x):
        # unpack x
        i=0
        for worker in self.workers:
            self.alpha[worker] = x[i]
            i = i + 1
        for example in self.examples:
            self.beta[example] = x[i]
            i = i + 1

        self.gradientQ()

        # pack x
        der = np.zeros_like(x)
        i = 0
        for worker in self.workers:
            der[i] = -self.dQalpha[worker] #Flip the sign since we want to minimize
            i = i + 1
        for example in self.examples:
            der[i] = -self.dQbeta[example] #Flip the sign since we want to minimize
            i = i + 1

        return der

    def Update_alpha_beta(self):

        x0=[]
        for worker in self.workers:
            x0.append(self.alpha[worker])
        for example in self.examples:
            x0.append(self.beta[example])

        # res = minimize(self.optimize_f, x0, method='BFGS', jac=self.optimize_df,tol=0.01,
        #       options={'disp': True,'maxiter':100})

        res = minimize(self.optimize_f, x0, method='CG', jac=self.optimize_df,tol=0.01,
                       options={'disp': False,'maxiter':25})

        self.optimize_f(res.x)

    #likelihood
    def log_likelihood(self):
        L = 0

        for example, worker_label_set in self.e2wl.items():
            L_example= 0
            for tlabel, prob in self.prior.items():
                L_label = prob
                for (worker, label) in worker_label_set:
                    sigma = self.sigmoid(self.alpha[worker]*self.expbeta(self.beta[example]))
                    delta = self.kronecker_delta(label, tlabel)
                    L_label = L_label * pow(sigma, delta)*pow((1-sigma)/(len(self.label_set)-1),1-delta)  # p(y_ij==label | zi==tlabel)
                L_example = L_example +L_label
            L = L + math.log(L_example)

        for worker in self.w2el.keys():
            L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.alpha[worker]-self.prioralpha[worker]),2)/2))

        for example in self.e2wl.keys():
            L = L + math.log((1/pow(2*math.pi,1/2)) * math.exp(-pow((self.beta[example]-self.priorbeta[example]),2)/2))

        return L

    #initialization
    def Init_prior(self):
        #uniform probability distribution
        # prior = {}
        # for label in self.label_set:
        #     prior[label] = 1.0/len(self.label_set)

        # solve the problem when P=1, parameter doesn't update
        prior = {}
        for label in self.label_set:
            prior[label] = (1.0-0.01)/len(self.label_set)
        prior[self.label_set[1]] += 0.01

        return prior

    def Init_alpha_beta(self):
        prioralpha={}
        priorbeta={}
        for worker in self.w2el.keys():
            prioralpha[worker]=1.  # debug, original value is 1
        for example in self.e2wl.keys():
            priorbeta[example]=1.  # debug, original value is 1
        return prioralpha,priorbeta

    def get_workerquality(self):
        sum_worker = sum(self.alpha.values())
        norm_worker_weight = dict()
        for worker in self.alpha.keys():
            norm_worker_weight[worker] = self.alpha[worker] / sum_worker
        return norm_worker_weight

    def Run(self, epoch_num, threshold):

        self.prior = self.Init_prior()
        self.prioralpha, self.priorbeta = self.Init_alpha_beta()

        self.alpha=self.prioralpha  # worker competence
        self.beta=self.priorbeta    # task difficulty

        self.Update_e2lpd()
        self.Update_alpha_beta()
        Q = self.computeQ()

        for epoch in tqdm(range(epoch_num), desc='Epoch'):

            # E-step
            self.Update_e2lpd()

            # M-step
            self.Update_alpha_beta()

            lastQ = Q
            Q = self.computeQ()
            if (math.fabs((Q-lastQ)/lastQ)) < threshold:
                break

            if self.enable_logging:
                if int(epoch) % int(self.logging_freq) == 0:
                    self.logging(epoch)

        return

    def predict_params(self):
        pred_dct = {}
        task_ids = self.csd_data['task_ids']
        task_difficulty = np.zeros(shape=len(task_ids))
        for i in range(task_difficulty.shape[0]):
            task_difficulty[i] = self.beta[task_ids[i]]
        pred_dct['task_difficulty'] = task_difficulty

        worker_ids = self.csd_data['worker_ids']
        worker_competence = np.zeros(shape=len(worker_ids))
        for i in range(worker_competence.shape[0]):
            worker_competence[i] = self.alpha[worker_ids[i]]
        pred_dct['worker_competence'] = worker_competence

        return pred_dct


def la_aggregation(crowd_tuples,
                   likelihood_name,
                   ds_initquality=0.7,
                   epoch_num=100,
                   threshold=1e-4,
                   random_seed=0,
                   enable_logging=True,
                   csd_data=None,
                   log_path=None,
                   logging_freq=10):

    random.seed(random_seed)
    np.random.seed(random_seed)

    if likelihood_name == 'ds':
        model = DS(crowd_tuples, csd_data, log_path, logging_freq, enable_logging, ds_initquality)
    elif likelihood_name == 'zc':
        model = ZC(crowd_tuples, csd_data, log_path, logging_freq, enable_logging)
    elif likelihood_name == 'glad':
        model = GLAD(crowd_tuples, csd_data, log_path, logging_freq, enable_logging)
    else:
        raise ValueError

    model.Run(epoch_num, threshold)
    pred_dct1 = model.predict_label()
    pred_dct2 = model.predict_params()
    pred_dct = {**pred_dct1, **pred_dct2}
    # print(accprf(csd_data['gold_labels'].reshape((-1, 1)), pred_dct['pred_y']))

    return pred_dct
