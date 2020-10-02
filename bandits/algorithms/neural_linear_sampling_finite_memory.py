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

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset_finite_memory import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel,TextCNN
import cvxpy as cvx
import math
from scipy.special import gamma
import tensorflow as tf

class NeuralLinearPosteriorSamplingFiniteMemory(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams,textflag ='no', optimizer='RMS'):

    self.name = name
    self.hparams = hparams
    self.latent_dim = self.hparams.layer_sizes[-1]
    self.intercept = False
    if self.intercept:
      self.param_dim=1+self.latent_dim
    else:
      self.param_dim = self.latent_dim
    self.EPSILON = 0.00001
    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior
    self.before=[]
    self.after=[]

    self.mu = [
        np.zeros(self.param_dim)
        for _ in range(self.hparams.num_actions)
    ]
    self.f = [
      np.zeros(self.param_dim)
      for _ in range(self.hparams.num_actions)
    ]
    self.yy = [0 for _ in range(self.hparams.num_actions)]
    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.param_dim)
                for _ in range(self.hparams.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.param_dim)
        for _ in range(self.hparams.num_actions)
    ]
    self.mu_prior_flag = self.hparams.mu_prior_flag
    self.sigma_prior_flag = self.hparams.sigma_prior_flag

    self.precision_prior=self.precision[:]
    self.mu_prior = np.zeros((self.param_dim,self.hparams.num_actions))
    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.hparams.num_actions)]
    self.b = [self._b0 for _ in range(self.hparams.num_actions)]

    # Regression and NN Update Frequency
    self.update_freq_lr = hparams.training_freq
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.optimizer_n = optimizer

    self.num_epochs = hparams.training_epochs
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=False,buffer_s=hparams.mem)
    self.latent_h = ContextualDataset(self.latent_dim,
                                      hparams.num_actions,
                                      intercept=self.intercept,buffer_s=hparams.mem)
    if textflag=='yes':
      self.bnn = TextCNN('adam', self.hparams.num_actions,self.hparams.batch_size, '{}-bnn'.format(name))
    else:
      self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.hparams.num_actions)
    ]

    try:
      beta_s = [
          np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
          for i in range(self.hparams.num_actions)
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite

      d = self.latent_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for i in range(self.hparams.num_actions)
      ]

    # Compute last-layer representation for the current context
    with self.bnn.graph.as_default():
      c = context.reshape((1, self.hparams.context_dim))
      z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})
      if self.intercept:
        z_context = np.append(z_context, 1.0).reshape((1, self.latent_dim + 1))
    # Apply Thompson Sampling to last-layer representation
    vals = [
        np.dot(beta_s[i], z_context.T) for i in range(self.hparams.num_actions)
    ]
    return np.argmax(vals)

  def calc_precision_prior(self,contexts):
    precisions_return = []
    n,m = contexts.shape
    prior = (self.EPSILON) * np.eye(self.param_dim)

    if self.cov is not None:
      for action,precision in enumerate(self.cov):
        ind = np.array([i for i in range(n) if self.data_h.actions[i] == action])
        if len(ind)>0:

          """compute confidence scores for old data"""
          d = []
          for c in self.latent_h.contexts[ind, :]:
            d.append(np.dot(np.dot(c,precision),c.T))

          """compute new data correlations"""
          phi = []
          for c in contexts[ind, :]:
            phi.append(np.outer(c,c))

          X = cvx.Variable((m, m), PSD=True)
          # Form objective.
          obj = cvx.Minimize(sum([(cvx.trace(X@phi[i]) - d[i])**2 for i in range(len(d))]))
          prob = cvx.Problem(obj)
          prob.solve()
          if X.value is None:
            precisions_return.append(np.linalg.inv(prior))
            self.cov[action] = prior

          else:
            precisions_return.append(np.linalg.inv(X.value+prior))
            self.cov[action] = X.value+prior
        else:
          precisions_return.append(np.linalg.inv(prior))
          self.cov[action] = prior

    return (precisions_return)

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.data_h.add(context, action, reward)
    c = context.reshape((1, self.hparams.context_dim))
    z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})
    self.latent_h.add(z_context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:

      if self.hparams.reset_lr:
        self.bnn.assign_lr()
      self.bnn.train(self.data_h, self.num_epochs)

      # Update the latent representation of every datapoint collected so far

      new_z = self.bnn.sess.run(self.bnn.nn,
                                feed_dict={self.bnn.x: self.data_h.contexts})

      i_contexts = None
      for context in new_z:
        c = np.array(context[:])
        if self.intercept:
          c = np.append(c, 1.0).reshape((1, self.latent_dim + 1))
        if i_contexts is None:
          i_contexts = c
        else:
          i_contexts = np.vstack((i_contexts, c))



      # Update the confidence prior using feature uncertainty matching

      #self.before.append(self.calc_model_evidence())
      if self.sigma_prior_flag==1:
        self.precision_prior = self.calc_precision_prior(contexts=i_contexts)
      # Update the mean prior using the weights of the NN
      if self.mu_prior_flag == 1:
        weights_p, bias_p = self.bnn.get_mu_prior()
        self.mu_prior[:self.latent_dim] = weights_p
        self.mu_prior[-1] = bias_p
      #self.after.append(self.calc_model_evidence())
      #print(self.before)
      #print(self.after)

      self.latent_h.replace_data(contexts=new_z)
      # Update the Bayesian Linear Regression

      for action_v in range(self.hparams.num_actions):
        # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
        z, y = self.latent_h.get_data(action_v)

        # The algorithm could be improved with sequential formulas (cheaper)
        self.precision[action_v] = (np.dot(z.T, z) + self.precision_prior[action_v])
        self.f[action_v] = np.dot(z.T, y)

    else:
      if self.intercept:
        z_context = np.append(z_context, 1.0).reshape((1, self.latent_dim + 1))
      self.precision[action] += np.dot(z_context.T, z_context)
      self.cov[action] = np.linalg.inv(self.precision[action])
      self.f[action] += (z_context.T * reward)[:, 0]


    # Calc mean and precision using bayesian linear regression
    self.mu[action] = np.dot(self.cov[action], (self.f[action]+np.dot(self.precision_prior[action],self.mu_prior[:,action])))

    # Inverse Gamma posterior update
    self.yy[action] += reward ** 2

    self.a[action] += 0.5
    b_upd = 0.5 * self.yy[action]
    b_upd += 0.5 * np.dot(self.mu_prior[:,action].T, np.dot(self.precision_prior[action], self.mu_prior[:,action]))
    b_upd -= 0.5 * np.dot(self.mu[action].T, np.dot(self.precision[action], self.mu[action]))
    self.b[action] = self.b0 + b_upd


  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
  def calc_model_evidence(self):
    vval = 0
    for action in range(self.hparams.num_actions):
      sigma0 = self.precision_prior[action]
      mu_0 = self.mu_prior[:, action]
      z, y = self.latent_h.get_data(action)
      n = z.shape[0]
      s = np.dot(z.T, z)
      s_n = (sigma0 + s)
      cov_a = np.linalg.inv(s_n)
      mu_a = np.dot(cov_a, (np.dot(z.T, y) + np.dot(sigma0, mu_0)))

      a_post = (self.a0 + n/2.0)
      b_upd = 0.5 * np.dot(y.T, y)
      b_upd += 0.5 * np.dot(mu_0.T, np.dot(sigma0, mu_0))
      b_upd -= 0.5 * np.dot(mu_a.T, np.dot(s_n, mu_a))
      b_post = self.b0 + b_upd
      val = np.float128(1)
      val/= ((np.float128(2.0) * math.pi) ** (n/2.0))
      val*= (gamma(a_post)/gamma(self.a0))
      val*= np.sqrt(np.linalg.det(sigma0)/np.linalg.det(s_n))
      val*= ((self.hparams.b0**self.hparams.a0)/(b_post**a_post))
      vval+=val
    vval/=self.hparams.num_actions
    return vval