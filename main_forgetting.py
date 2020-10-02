# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import os,pickle
import tensorflow as tf
import sys
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data,sample_txt_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.data.data_sampler import sample_eeg_data,sample_diabetic_data,sample_phone_data,sample_aps_data,sample_amazon_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.algorithms.epsilon_greedy import NeuralLinearEpsilonGreedy
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data,sample_wheel2_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.algorithms.neural_linear_sampling_finite_memory import NeuralLinearPosteriorSamplingFiniteMemory
from bandits.algorithms.neural_linear_sampling_online import NeuralLinearPosteriorSamplingOnline
# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')
flags.DEFINE_string(
    'financial_data',
    os.path.join(base_route, data_route, 'raw_stock_contexts'),
    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string(
    'statlog_data',
    os.path.join(base_route, data_route, 'shuttle.trn'),
    'Directory where Statlog data is stored.')
flags.DEFINE_string(
    'adult_data',
    os.path.join(base_route, data_route, 'adult.full'),
    'Directory where Adult data is stored.')
flags.DEFINE_string(
    'covertype_data',
    os.path.join(base_route, data_route, 'covtype.data'),
    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data',
    os.path.join(base_route, data_route, 'USCensus1990.data.txt'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'eeg_data',
    os.path.join(base_route, data_route, 'eeg.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'diabetic_data',
    os.path.join(base_route, data_route, 'diabetic.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'amazon_data_file',
    os.path.join(base_route, data_route, 'Amazon.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'phone_data',
    os.path.join(base_route, data_route, 'samsung.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'aps_data',
    os.path.join(base_route, data_route, 'aps.csv'),
    'Directory where Census data is stored.')

flags.DEFINE_string(
    'positive_data_file',
    os.path.join(base_route, data_route, 'rt-polarity.pos'),
    'Directory where Census data is stored.')

flags.DEFINE_string(
    'negative_data_file',
    os.path.join(base_route, data_route, 'rt-polarity.neg'),
    'Directory where Census data is stored.')

flags.DEFINE_integer("task_id",None,"ID of task")

def sample_data(data_type, num_contexts=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """
  if data_type == '2linear':
    # Create linear dataset
    num_actions = 2
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                               num_actions, num_contexts,
                                               noise_stds, shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name, num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None
  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 2
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts,
                                     shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name, num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1] #54
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'wheel':
    delta = 0.5
    num_actions = 5
    context_dim = 2
    mean_v = [0.1,0.1,0.1,0.1,0.2]
    std_v = [0.1, 0.1, 0.1, 0.1, 0.1]
    mu_large = 0.4
    std_large = 0.1
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'wheel2':
    delta = 0.7
    num_actions = 2
    context_dim = 2
    mean_v = [0.0, 1]
    std_v = [0.1, 0.1]
    mu_large = 2
    std_large = 0.1
    dataset, opt_wheel = sample_wheel2_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'eeg':
    file_name = FLAGS.eeg_data
    num_actions = 5
    num_contexts = min(11500, num_contexts)
    sampled_vals = sample_eeg_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'diabetic':
    file_name = FLAGS.diabetic_data
    num_actions = 3
    num_contexts = min(100000, num_contexts)
    sampled_vals = sample_diabetic_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'phone':
    file_name = FLAGS.phone_data
    num_actions = 6
    num_contexts = min(7767, num_contexts)
    sampled_vals = sample_phone_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'aps':
    file_name = FLAGS.aps_data
    num_actions = 2
    num_contexts = min(76000, num_contexts)
    sampled_vals = sample_aps_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim
  elif data_type == 'txt':
    file_name = [FLAGS.positive_data_file,FLAGS.negative_data_file]
    num_actions = 2
    num_contexts = min(10000, num_contexts)
    sampled_vals = sample_txt_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions),vocab_processor = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor
  elif data_type == 'amazon':
    file_name = FLAGS.amazon_data_file
    num_actions = 5
    num_contexts = min(10000, num_contexts)
    sampled_vals = sample_amazon_data(file_name, num_contexts,shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions),vocab_processor = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def show_predictions(dataset,algos):
    for ii,a in enumerate(algos):
        x = []
        y = []
        r = [[]for _ in range(4)]
        for c in dataset:
            x.append(c[0])
            y.append(c[1])
            context = np.array(c[:2]).reshape((1, 2))
            if ii>0:
                context = a.bnn.sess.run(a.bnn.nn, feed_dict={a.bnn.x: context})
            context = np.append(context,1)
            r[2].append(np.dot(a.mu[0],context.T))
            r[0].append(np.dot(a.mu[1], context.T))
            r[1].append(c[2])
            r[3].append(c[3])
        titles= ['pred a2', 'true a1','pred a1','true a2',]
        fig = plt.figure()
        for i in range(4):
            plt.subplot(2, 2, i)
            plt.scatter(x, y, c=r[i], alpha=0.5)
            plt.title(titles[i])
        fig.suptitle(a.name)
        plt.show()
def show_predictions2(dataset,algos):
    j = 5
    x = []
    y = []
    r = [[] for _ in range(15)]
    for ii,a in enumerate(algos):
        if ii==1:
            continue
        for c in dataset:
            context = np.array(c[:2]).reshape((1, 2))
            if ii>0:
                x.append(c[0])
                y.append(c[1])
                context = a.bnn.sess.run(a.bnn.nn, feed_dict={a.bnn.x: context})
            context = np.append(context,1)
            if j == 5:
                for t in range(5):
                    r[t].append(c[t + 2])
            for t in range(5):
                r[t+j].append(np.dot(a.mu[t],context.T))
        j+=5

    titles= ['r1', 'r2','r3','r4','r5',
             'r1_lin', 'r2_lin', 'r3_lin', 'r4_lin', 'r5_lin',
             'r1_neurallin', 'r2_neurallin', 'r3_neurallin', 'r4_neurallin', 'r5_neurallin',]
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.scatter(x, y, c=r[i], alpha=0.5,linewidths=0,edgecolors=None)
        plt.title(titles[i])
        plt.xticks([], [])
        plt.yticks([], [])

    plt.show()
def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')
def display_final_results(algos, opt_rewards,res, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed.'.format(
    name))
  print('---------------------------------------------------')

  performance_triples = []
  for j, a in enumerate(algos):
      performance_triples.append((a.name, np.mean(res[j]),np.std(res[j])))
  performance_pairs = sorted(performance_triples,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, mean_reward,std_reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10} +- {:10}.'.format(i, name, mean_reward,std_reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def Run(context_dim, num_actions, dataset, algos, opt_rewards, opt_actions, data_type):
    # Run contextual bandit problem
    t_init = time.time()
    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
    h_actions, h_rewards = results

    # Display results
    display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)
    # Append Results
    res=[]
    for j, a in enumerate(algos):
        res.append((np.sum(h_rewards[:, j])))
    return res
def main(argv): #runindex, train freq, data size, data type, output file

  # Problem parameters
  num_contexts = 4000
  tfn=400
  tfe=tfn*2

  Datatypes = ['linear', 'sparse_linear', 'mushroom', 'financial', 'jester',
                   'statlog', 'adult', 'covertype', 'census', 'wheel','eeg','phone','aps']
  data_type = 'statlog'
  l_sizes=[50]
  #data_type = '2linear'
  outdir  ="./"
  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor = sampled_vals
  bs = num_actions * 16
  print(num_actions)
  print(context_dim)
  MEMSIZE = 100

  if not os.path.exists(outdir):
      os.makedirs(outdir)

  filename = outdir + '/' + data_type+'_index_'+'1'+'_train_freq_' + str(tfn)
  # Define hyperparameters and algorithms
  hparams = tf.contrib.training.HParams(num_actions=num_actions)

  hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               a0=6,
                                               b0=6,
                                               lambda_prior=0.25,
                                               initial_pulls=2)

  hparams_txt = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                batch_size=512,
                                                initial_lr=0.1,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=0.25,
                                                )




  hparams_rms = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[50],
                                            batch_size=512,
                                            activate_decay=True,
                                            initial_lr=0.1,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=1000,
                                            buffer_s=-1,
                                            initial_pulls=2,
                                            optimizer='RMS',
                                            reset_lr=True,
                                            lr_decay_rate=0.5,
                                            training_freq=50,
                                            training_epochs=100,
                                            p=0.95,
                                            q=3)

  hparams_dropout = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=[50],
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                optimizer='RMS',
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=50,
                                                training_epochs=100,
                                                use_dropout=True,
                                                keep_prob=0.80)

  hparams_bbb = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[50],
                                            batch_size=512,
                                            activate_decay=True,
                                            initial_lr=0.1,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=1000,
                                            buffer_s=-1,
                                            initial_pulls=2,
                                            optimizer='RMS',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=10,
                                            initial_training_steps=100,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=50,
                                            training_epochs=100)

  hparams_nlinear = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=bs,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=0.25)
  hparams_epsilon = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=bs,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                epsilon=0.1)

  hparams_nlinear2 = tf.contrib.training.HParams(num_actions=num_actions,
                                                 context_dim=context_dim,
                                                 init_scale=0.3,
                                                 activation=tf.nn.relu,
                                                 layer_sizes=[50],
                                                 batch_size=512,
                                                 activate_decay=True,
                                                 initial_lr=0.1,
                                                 max_grad_norm=5.0,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 buffer_s=-1,
                                                 initial_pulls=2,
                                                 reset_lr=True,
                                                 lr_decay_rate=0.5,
                                                 training_freq=10,
                                                 training_freq_network=tfn,
                                                 training_epochs=tfe,
                                                 a0=6,
                                                 b0=6,
                                                 lambda_prior=0.25)

  hparams_nlinear_finite_memory = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=bs,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=MEMSIZE,
                                                mu_prior_flag=1,
                                                sigma_prior_flag=1,
                                                              )

  hparams_nlinear_finite_memory2 = tf.contrib.training.HParams(num_actions=num_actions,
                                                 context_dim=context_dim,
                                                 init_scale=0.3,
                                                 activation=tf.nn.relu,
                                                 layer_sizes=[50],
                                                 batch_size=bs,
                                                 activate_decay=True,
                                                 initial_lr=0.1,
                                                 max_grad_norm=5.0,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 buffer_s=-1,
                                                 initial_pulls=2,
                                                 reset_lr=True,
                                                 lr_decay_rate=0.5,
                                                 training_freq=10,
                                                 training_freq_network=tfn,
                                                 training_epochs=tfe,
                                                 a0=6,
                                                 b0=6,
                                                 lambda_prior=1,
                                                 mem=MEMSIZE,
                                                 mu_prior_flag=1,
                                                 sigma_prior_flag=1,
                                                               )

  hparams_nlinear_finite_memory_no_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=MEMSIZE,
                                                mu_prior_flag=0,
                                                sigma_prior_flag=0,
                                                              )

  hparams_nlinear_finite_memory2_no_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                 context_dim=context_dim,
                                                 init_scale=0.3,
                                                 activation=tf.nn.relu,
                                                 layer_sizes=[50],
                                                 batch_size=512,
                                                 activate_decay=True,
                                                 initial_lr=0.1,
                                                 max_grad_norm=5.0,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 buffer_s=-1,
                                                 initial_pulls=2,
                                                 reset_lr=True,
                                                 lr_decay_rate=0.5,
                                                 training_freq=10,
                                                 training_freq_network=tfn,
                                                 training_epochs=tfe,
                                                 a0=6,
                                                 b0=6,
                                                 lambda_prior=1,
                                                 mem=MEMSIZE,
                                                 mu_prior_flag=0,
                                                 sigma_prior_flag=0,
                                                               )
  hparams_nlinear_finite_memory_no_sig_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=tfn,
                                                training_epochs=tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=MEMSIZE,
                                                mu_prior_flag=1,
                                                sigma_prior_flag=0,
                                                              )

  hparams_nlinear_finite_memory2_no_sig_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                 context_dim=context_dim,
                                                 init_scale=0.3,
                                                 activation=tf.nn.relu,
                                                 layer_sizes=[50],
                                                 batch_size=512,
                                                 activate_decay=True,
                                                 initial_lr=0.1,
                                                 max_grad_norm=5.0,
                                                 show_training=False,
                                                 freq_summary=1000,
                                                 buffer_s=-1,
                                                 initial_pulls=2,
                                                 reset_lr=True,
                                                 lr_decay_rate=0.5,
                                                 training_freq=10,
                                                 training_freq_network=tfn,
                                                 training_epochs=tfe,
                                                 a0=6,
                                                 b0=6,
                                                 lambda_prior=1,
                                                 mem=MEMSIZE,
                                                 mu_prior_flag=1,
                                                 sigma_prior_flag=0,
                                                               )


  hparams_pnoise = tf.contrib.training.HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               init_scale=0.3,
                                               activation=tf.nn.relu,
                                               layer_sizes=[50],
                                               batch_size=512,
                                               activate_decay=True,
                                               initial_lr=0.1,
                                               max_grad_norm=5.0,
                                               show_training=False,
                                               freq_summary=1000,
                                               buffer_s=-1,
                                               initial_pulls=2,
                                               optimizer='RMS',
                                               reset_lr=True,
                                               lr_decay_rate=0.5,
                                               training_freq=50,
                                               training_epochs=100,
                                               noise_std=0.05,
                                               eps=0.1,
                                               d_samples=300,
                                              )

  hparams_alpha_div = tf.contrib.training.HParams(num_actions=num_actions,
                                                  context_dim=context_dim,
                                                  init_scale=0.3,
                                                  activation=tf.nn.relu,
                                                  layer_sizes=[50],
                                                  batch_size=512,
                                                  activate_decay=True,
                                                  initial_lr=0.1,
                                                  max_grad_norm=5.0,
                                                  show_training=False,
                                                  freq_summary=1000,
                                                  buffer_s=-1,
                                                  initial_pulls=2,
                                                  optimizer='RMS',
                                                  use_sigma_exp_transform=True,
                                                  cleared_times_trained=10,
                                                  initial_training_steps=100,
                                                  noise_sigma=0.1,
                                                  reset_lr=False,
                                                  training_freq=50,
                                                  training_epochs=100,
                                                  alpha=1.0,
                                                  k=20,
                                                  prior_variance=0.1)

  hparams_gp = tf.contrib.training.HParams(num_actions=num_actions,
                                           num_outputs=num_actions,
                                           context_dim=context_dim,
                                           reset_lr=False,
                                           learn_embeddings=True,
                                           max_num_points=1000,
                                           show_training=False,
                                           freq_summary=1000,
                                           batch_size=512,
                                           keep_fixed_after_max_obs=True,
                                           training_freq=50,
                                           initial_pulls=2,
                                           training_epochs=100,
                                           lr=0.01,
                                           buffer_s=-1,
                                           initial_lr=0.001,
                                           lr_decay_rate=0.0,
                                           optimizer='RMS',
                                           task_latent_dim=5,
                                           activate_decay=False)


  hparams_nlinear_online_no_sig_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=num_actions * 16,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=1,
                                                training_epochs=1,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=num_actions*100,
                                                mu_prior_flag=1,
                                                sigma_prior_flag=0,
                                                pgd_steps=1,
                                                pgd_freq = 1,
                                                pgd_batch_size=20,
                                                verbose=False)

  hparams_nlinear_online_no_prior = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size=num_actions * 16,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=1,
                                                training_epochs=1,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=num_actions*100,
                                                mu_prior_flag=0,
                                                sigma_prior_flag=0,
                                                pgd_freq = 1,
                                                pgd_steps=1,
                                                pgd_batch_size=20,
                                                verbose=False)

  hparams_nlinear_online = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=l_sizes,
                                                batch_size= num_actions * 16,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=1,
                                                training_freq_network=1, #tfn,
                                                training_epochs=1,#tfe,
                                                a0=6,
                                                b0=6,
                                                lambda_prior=1,
                                                mem=num_actions*100,
                                                mu_prior_flag=1,
                                                sigma_prior_flag=1,
                                                pgd_freq = 1,
                                                pgd_steps=1,
                                                pgd_batch_size=20,
                                                verbose=False)


  Nruns=10
  n_algs=5
  d=50
  res = np.zeros((n_algs,Nruns,num_contexts))
  totalreward=[0 for i in range(n_algs)]
  rewards = [[] for i in range(n_algs) ]
  for i_run in range(Nruns):
      # algos = [
      #     NeuralLinearEpsilonGreedy('epsilon-greedy', hparams_epsilon),
      #     #LinearFullPosteriorSampling('LinFullPost', hparams_linear),
      #     NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
      #     NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory', hparams_nlinear_finite_memory),
      #     #NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory_noSigP',
      #     #                                         hparams_nlinear_finite_memory_no_sig_prior),
      #     #NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory_noP', hparams_nlinear_finite_memory_no_prior)
      #     ]
      algos = [
          # UniformSampling('Uniform Sampling', hparams),
          # UniformSampling('Uniform Sampling 2', hparams),
          # FixedPolicySampling('fixed1', [0.75, 0.25], hparams),
          # FixedPolicySampling('fixed2', [0.25, 0.75], hparams),
          # PosteriorBNNSampling('RMS', hparams_rms, 'RMSProp'),
          # PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
          # PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
          NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
          # NeuralLinearPosteriorSampling('NeuralLinear2', hparams_nlinear2),
          NeuralLinearPosteriorSamplingOnline('NeuralLinearOnline', hparams_nlinear_online),
          NeuralLinearPosteriorSamplingOnline('NeuralLinearOnline_noSigP', hparams_nlinear_online_no_sig_prior),
          NeuralLinearPosteriorSamplingOnline('NeuralLinearOnline_noP', hparams_nlinear_online_no_prior),

          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory', hparams_nlinear_finite_memory),
          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory2', hparams_nlinear_finite_memory2),
          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory_noP',
          #                                           hparams_nlinear_finite_memory_no_prior),
          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory2_noP', hparams_nlinear_finite_memory2_no_prior),
          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory_noSigP',
          #                                           hparams_nlinear_finite_memory_no_sig_prior),
          # NeuralLinearPosteriorSamplingFiniteMemory('NeuralLinearFiniteMemory2_noSigP', hparams_nlinear_finite_memory2_no_sig_prior),
          LinearFullPosteriorSampling('LinFullPost', hparams_linear),
          # BootstrappedBNNSampling('BootRMS', hparams_rms),
          # ParameterNoiseSampling('ParamNoise', hparams_pnoise),
          # PosteriorBNNSampling('BBAlphaDiv', hparams_alpha_div, 'AlphaDiv'),
          # PosteriorBNNSampling('MultitaskGP', hparams_gp, 'GP'),
      ]
      results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
      h_actions, h_rewards = results
      for j, a in enumerate(algos):
          print(np.sum(h_rewards[:, j]))
          totalreward[j] += ((np.sum(h_rewards[:, j])) / Nruns)
          rewards[j].append((np.sum(h_rewards[:, j])))
      actions = [[] for i in range(len(h_actions[0]))]
      for aa in h_actions:
          for i, a in enumerate(aa):
              actions[i].append(a)
      for i_alg in range(len(algos)):
          res[i_alg, i_run, :] = 1 * ((actions[i_alg] != opt_actions))
      if i_run < (Nruns - 1):
          algos = None


  print(totalreward)
  for rr in rewards:
      print(np.mean(rr))
      print(np.std(rr))
# show_predictions2(dataset, algos)
  means = np.mean(res, axis=1)
  stds = np.std(res, axis=1)

  # for i_alg in range(len(algos)):
  #     # smooth(res[i_alg, :] / Nruns, d)[int(d / 2):int(num_contexts - d / 2)]
  #     plt.subplot(510 + i_alg + 1)
  #     # plt.stem(smooth(res[i_alg,:]/Nruns,d)[int(d/2):int(num_contexts-d/2)])
  #     y_ = smooth(means[i_alg, :], d)[int(d / 2):int(num_contexts - d / 2)]
  #     s_ = smooth(stds[i_alg, :], d)[int(d / 2):int(num_contexts - d / 2)]
  #     x_ax = [x for x in range(num_contexts - d)]
  #     plt.plot(x_ax, y_)
  #     plt.fill_between(x_ax, y_ - s_, y_ + s_, alpha=0.2)
  #     plt.xticks(np.arange(0, num_contexts - d/2, step=tfn))
  #     plt.ylim(top=1)
  #     plt.title(algos[i_alg].name)
  #     plt.ylabel('% of mistakes')
  #     plt.xlabel('Number of steps')
  colors = ['g','r','b','c','m']
  for i_alg in range(len(algos)):
      # smooth(res[i_alg, :] / Nruns, d)[int(d / 2):int(num_contexts - d / 2)]
      # plt.plot(510 + i_alg + 1)
      # plt.stem(smooth(res[i_alg,:]/Nruns,d)[int(d/2):int(num_contexts-d/2)])
      y_ = smooth(means[i_alg, :], d)[int(d / 2):int(num_contexts - d / 2)]
      s_ = smooth(stds[i_alg, :], d)[int(d / 2):int(num_contexts - d / 2)] * 1.6 / 7.
      x_ax = [x for x in range(num_contexts - d)]
      plt.plot(x_ax, y_,label=algos[i_alg].name,color=colors[i_alg])
      plt.fill_between(x_ax, y_ - s_, y_ + s_,color=colors[i_alg], alpha=0.2)
      plt.xticks(np.arange(0, num_contexts - d/2, step=tfn))
      plt.ylim(top=1)
      plt.ylabel('% of mistakes')
      plt.xlabel('Number of steps')
  plt.legend()
  plt.tight_layout()

  plt.show()



if __name__ == '__main__':
  app.run(main)
