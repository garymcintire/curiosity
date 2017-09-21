import gym
import roboschool
import os
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.engine.topology import Input
from keras.layers import merge
from keras.layers.core import  Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import mpi_util     # If you don't want to use mpi parallel processes, just comment all the lines with mpi_util and this file is all you need to wrap environments in single process mode
import tensorflow as tf


# this makes an inversemodel network, such that action = f(state, stateprime)
#  The first few layers generate the features. The last hidden layer is the feature vector.
# That vector will be output from reset() and step() such the the agent wrapping this will see
# the feature layer and not the original obsrvation layer
# It works a little but not as good as without it.
class inverse_feature_wrapper(object):
    def __init__(self, env, hidlist=[100,50], N = 1000000, xform_space=1):
        self.env = env
        self.hidlist = hidlist
        self.xform_space = xform_space  # true for inverse_wrapper, false for curiousity
        self.N = N      # the last N number of samples to train with
        if xform_space:
            self.observation_space = gym.spaces.Box(np.full(hidlist[-1],float('-inf')), np.full(hidlist[-1],float('inf')) ) # make a box the sixe of the feature layer
        else:
            self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.batch = 0
        self.paths = []
        self.batchpaths = []
        self.path = []
        self.nunactions=env.action_space.shape[0]
        self.numstate=env.observation_space.shape[0]
        self.make_networks()
    def reset(self, *args, **kwargs):
        ob = self.env.reset(*args, **kwargs)
        self.prev_ob = ob
        if self.xform_space:
            return self.features_of(ob)
        else:
            return ob
    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self.save_sars( observation, action, reward, done, info)
        if self.xform_space:
            return self.features_of(observation), reward, done, info   # Convert the ob to a feature vector
        else:
            return observation, reward, done, info
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def trainN(self, N):    # last N steps to train with
        # if mpi_util.rank != 0: return   # only train on rank0
        s = np.concatenate([d['s'] for d in self.paths])
        sprime = np.concatenate([d['sprime'] for d in self.paths])
        actions = np.concatenate([d['actions'] for d in self.paths])
        self.inv.fit([s[-N:],sprime[-N:]],actions[-N:])# train_networks APPARENTLY KERAS .fit CREATES MORE trainable VARIABLES SUCH THAT WE MUST DO IT ON EACH PROCESS AT LEAST ONCE
    def end_of_batch(self, bcast_inv=True):
        s = np.concatenate([d['s'] for d in self.batchpaths])
        sprime = np.concatenate([d['sprime'] for d in self.batchpaths])
        actions = np.concatenate([d['actions'] for d in self.batchpaths])
        d = mpi_util.rank0_accum_batches({'s': s,  'sprime': sprime,  'actions': actions})
        self.paths.append(d)
        self.batchpaths = []
        self.path = []
        if True or self.batch < 100:
            N = self.N
            self.trainN(N)   # last N steps to train with
            if self.batch % 10 == 0: self.trainN(10*N)
            if self.batch % 100 == 0: self.trainN(100*N)
            if bcast_inv: mpi_util.rank0_bcast_wts_keras(self.inv)
        # print(str(mpi_util.rank)+'rank  len of states', len(self.paths[-1]['s']) )
        self.batch += 1

    def make_networks(self):
        self.op = keras.optimizers.Adam(lr=.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.s = Input((self.numstate, ), name='s')
        self.sprime = Input((self.numstate, ), name='sprime')
        fh1 = Dense(self.hidlist[0],activation='tanh', name='Densefh1')
        fh2 = Dense(self.hidlist[1], activation='tanh', name='Densefh2')
        features = fh2(fh1((self.s)))
        self.features = Model(input=[self.s], output=[features], name='features')
        self.features.compile(loss='mse', optimizer=self.op)
        self.features.summary()
        # inv = Dense(self.nunactions, activation='tanh')(merge([fh2(fh1((self.s))), fh2(fh1((self.sprime)))], mode='concat'))
        self.feat_s      = self.features(self.s)
        self.feat_sprime = self.features(self.sprime)
        inv = Dense(self.nunactions, activation='tanh',name='dense_inv1')(merge([self.feat_s, self.feat_sprime], mode='concat', name='inv_merge'))
        self.inv = Model(input=[self.s, self.sprime], output=[inv])
        self.inv.compile(loss='mse', optimizer=self.op)
        self.inv.summary()
    def features_of(self, states):
        ans = self.features.predict(np.array([states]))[0]
        return ans


    def save_sars(self, observation, action, reward, done, info):
        sars = {'s': self.prev_ob,
                    'actions': action,
                    'rewards': reward,
                    'sprime': observation,
                    'dones': done}
        self.path.append(sars)
        self.prev_ob = observation
        if done:
            np.array([d['s'] for d in self.path])
            self.batchpaths.append({
                's': np.array([d['s'] for d in self.path], dtype='f'),
                'sprime': np.array([d['sprime'] for d in self.path], dtype='f'),
                'actions': np.array([d['actions'] for d in self.path], dtype='f'),
                 }
            )
            self.path = []

# this makes intrinsic reward based on finding state spaces that have not been learned yet and going to them
# https://pathak22.github.io/noreward-rl/resources/icml17.pdf   Curiosity-driven Exploration by Self-supervised Prediction
# It return both the original reward and the intrinsic reward in the info return from env.step so the user
#   can mix as they like
class curiosity_wrapper(inverse_feature_wrapper):
    def __init__(self, env, hidlist=[100,50]):
        super().__init__(env, hidlist=hidlist, N=20000, xform_space=0)
        self.intrinsic_coeff = .5       # how much intrinsic stddevs(scaled to orig reward std) to add to the original reward
        self.steps_in_batch = 0
        self.reward_stat = stat(.5)
        self.intrinsic_stat = stat(.5)

    def make_networks(self):
        super().make_networks()
        self.a = Input((self.nunactions, ), name='a')
        forward = Dense(self.hidlist[1],activation='tanh', name='Densefwd2')( Dense(self.hidlist[0],activation='tanh', name='Densefwd1')(merge([self.a, self.feat_s], mode='concat',name='fwd_merge')) )
        self.forward = Model(input=[self.a, self.s], output=[forward], name='forward')
        self.forward.compile(loss='mse', optimizer=self.op)
        self.forward.summary()
        self.trn_mdl = Model(input=[self.a, self.s, self.sprime], output=[forward, self.inv.output], name='trn_mdl')
        self.trn_mdl.compile(loss=['mse','mse'], optimizer=self.op, loss_weights=[.25, 1.]) # num actions is small while num features is different. Should we compensate for the imbalance?
        self.trn_mdl.summary()
        pass

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self.save_sars( observation, action, reward, done, info)
        feat = self.features_of(observation)
        pred = self.forward.predict([action[None], observation[None]])[0]
        intrinsic_reward = np.mean((pred - feat)**2)
        info['orig_reward'] = reward
        info['intrinsic_reward'] = intrinsic_reward
        self.reward_stat.add(reward)
        self.intrinsic_stat.add(intrinsic_reward)
        self.steps_in_batch += 1
        norm_reward = self.reward_stat.norm(reward, demean=False)
        norm_intrinsic = self.intrinsic_stat.norm(intrinsic_reward)
        # print(str(mpi_util.rank)+'norm_reward', norm_reward,'norm_intrinsic', norm_intrinsic)
        cmbd_intrinsic = self.intrinsic_coeff*norm_intrinsic*(self.reward_stat.runstd)
        cmbd_rew = reward
        # cmbd_rew = 1
        cmbd_reward = cmbd_rew + cmbd_intrinsic
        # print('cmbd',cmbd_reward, cmbd_rew, cmbd_intrinsic)
        return observation, cmbd_reward, done, info

    def trainN(self, N):    # last N steps to train with
        if mpi_util.rank != 0: return   # only train on rank0
        s = np.concatenate([d['s'] for d in self.paths])[-N:]
        sprime = np.concatenate([d['sprime'] for d in self.paths])[-N:]
        actions = np.concatenate([d['actions'] for d in self.paths])[-N:]
        targets = self.features.predict(np.array(sprime))
        self.trn_mdl.fit([actions,s,sprime], [targets, actions],) # train_networks APPARENTLY KERAS .fit CREATES MORE trainable VARIABLES SUCH THAT WE MUST DO IT ON EACH PROCESS AT LEAST ONCE

    def end_of_batch(self, bcast_inv=False):
        super().end_of_batch(bcast_inv=False)
        mpi_util.chk_cfg_file(self, filename='curiosity')
        # print(str(mpi_util.rank)+'step_in_batch', self.steps_in_batch)
        print(str(mpi_util.rank) +'orig_reward_mean', self.reward_stat.thismean, 'intrinsic_reward_mean', self.intrinsic_stat.thismean, 'reward_std', self.reward_stat.thisstd, 'intrinsic_std', self.intrinsic_stat.thisstd)
        print(str(mpi_util.rank) +'running_reward_mean', self.reward_stat.runmean, 'intrinsic_reward_mean', self.intrinsic_stat.runmean, 'reward_std', self.reward_stat.runstd, 'intrinsic_std', self.intrinsic_stat.runstd)
        self.reward_stat.eob()      # Note this does not combine the stats of each process. Assume proc0 having only his stats is representative enough
        self.intrinsic_stat.eob()
        # mpi_util.chksum_keras(self.features, str(mpi_util.rank)+'before')
        mpi_util.rank0_bcast_wts_keras(self.forward)
        # mpi_util.chksum_keras(self.features, str(mpi_util.rank)+'after')
        #mpi_util.rank0_bcast_wts_keras(self.forward)

def var( n, sum, sumsqd):
    return (n*sumsqd - sum**2)/(n*(n-1))
class stat (object):
    def __init__(self, coeff):
        self.coeff = coeff
        self.sum = 0
        self.sum2 = 0
        self.n = 0
        self.runmean = 0
        self.runstd = 1.
        self.thismean = 0
        self.thisstd = 0
        pass
    def add(self, x):
        self.n += 1
        self.sum += x
        self.sum2 += x**2
    def eob(self):  # end of batch
        self.thismean = self.sum / self.n
        self.runmean = self.avg(self.runmean, self.thismean)
        self.thisstd = np.sqrt(var(self.n, self.sum, self.sum2))
        self.runstd = self.avg(self.runstd, self.thisstd)
        self.sum = 0
        self.sum2 = 0
        self.n = 0
    def avg(self, old, new):
        return self.coeff*old + (1-self.coeff)*new
    def norm(self, x, demean=True):
        if demean:
            return (x - self.runmean) / self.runstd
        else:
            return (x - self.runmean) / self.runstd  + self.runmean
