"""TODO(tomzahavy): DO NOT SUBMIT without one-line documentation for display_results.

TODO(tomzahavy): DO NOT SUBMIT without a detailed description of display_results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys,os
import tensorflow as tf
from matplotlib.lines import Line2D

import numpy as np
from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.algorithms.neural_linear_sampling_finite_memory import NeuralLinearPosteriorSamplingFiniteMemory
def display_final_results(algos,res, Name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed.'.format(
    Name))
  print('---------------------------------------------------')
  colors = ['red', 'red','blue','blue', 'blue','blue','blue','blue', 'red']
  performance_triples = []
  means=[]
  stds=[]
  for j, a in enumerate(algos):
      performance_triples.append((a, np.mean(res[j]),np.std(res[j])))
      if a=='Uniform Sampling':
          continue
      means.append(np.mean(res[j]))
      stds.append(np.std(res[j]))
  performance_pairs = sorted(performance_triples,
                             key=lambda elt: elt[1],
                             reverse=True)
  x_pos = np.arange(len(means))

  for i, (name, mean_reward,std_reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10} +- {:10}.'.format(i, name, mean_reward,std_reward))

  fig, ax = plt.subplots()
  ax.bar(x_pos, means, yerr=stds, align='center', color=colors,alpha=0.5, ecolor='black', capsize=10)
  ax.set_ylabel('Mean Reward +- STD')
  ax.set_xticks(x_pos)
  plt.ylim(int(np.min(means)-3*stds[np.argmin(means)]),int(np.max(means)+3*stds[np.argmax(means)]))

  ax.set_xticklabels(algos[1:])
  ax.set_title(Name)
  ax.yaxis.grid(True)
  plt.xticks(fontsize=5)
  # Save the figure and show

  plt.tight_layout()
  plt.savefig(Name+'.pdf')

  print('---------------------------------------------------')
  print('Frequency of optimal actions (action, frequency):')
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def display_final_results2(algos,res, Name):
  """Displays summary statistics of the performance of each algorithm."""
  algos = ['Uniform Sampling', 'NeuralLinear', 'NeuralLinear',
           'NeuralLinear \n Both priors', 'NeuralLinear \n Both priors', 'NeuralLinear \n No Prior',
           'NeuralLinear \n No Prior', 'NeuralLinear \n Mean Prior',
           'NeuralLinear  \n Mean Prior'
      , 'LinFullPost']
  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed.'.format(
    Name))
  print('---------------------------------------------------')
  colors = ['red','blue','blue', 'blue','red']
  performance_triples = []
  means=[]
  stds=[]
  names=[]
  for j, a in enumerate(algos):
      performance_triples.append((a, np.mean(res[j]),np.std(res[j])))
      if a=='Uniform Sampling' or (j%2==1 and (j<len(algos)-2)):
          continue
      means.append(np.mean(res[j]))
      stds.append(np.std(res[j]))
      names.append(a)
  performance_pairs = sorted(performance_triples,
                             key=lambda elt: elt[1],
                             reverse=True)
  x_pos = np.arange(len(means))

  for i, (name, mean_reward,std_reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10} +- {:10}.'.format(i, name, mean_reward,std_reward))

  fig, ax = plt.subplots()
  ax.bar(x_pos, means, yerr=stds, align='center', color=colors,alpha=0.5, ecolor='black', capsize=10)
  ax.set_ylabel('Mean Reward +- STD')
  ax.set_xticks(x_pos)
  plt.ylim(int(np.min(means)-3*stds[np.argmin(means)]),int(np.max(means)+3*stds[np.argmax(means)]))

  ax.set_xticklabels(names)
  ax.set_title(Name)
  ax.yaxis.grid(True)
  custom_lines = [Line2D([0], [0], color='red', lw=4),
                  Line2D([0], [0], color='blue', lw=4)]


  ax.legend(custom_lines,('Full Memory', 'Bounded Memort'))
  plt.xticks(fontsize=10)
  # Save the figure and show

  plt.tight_layout()
  plt.savefig(Name+'.pdf')

  print('---------------------------------------------------')
  print('Frequency of optimal actions (action, frequency):')
  print('---------------------------------------------------')
  print('---------------------------------------------------')

def main():


  algos = ['Uniform Sampling', 'NeuralLinear', 'NeuralLinear2',
           'NeuralLinear \n FiniteMemory', 'NeuralLinear \n FiniteMemory2','NeuralLinear \n FiniteMemory \n No Prior',
           'NeuralLinear \n FiniteMemory2 \n No Prior','NeuralLinear \n FiniteMemory \n Mean Prior','NeuralLinear \n FiniteMemory2 \n Mean Prior'
                                                ,'LinFullPost']

  Datatypes = ['Linear', 'Sparse Linear', 'Mushroom', 'Financial', 'Jester',
                   'Statlog', 'Adult', 'Covertype', 'Census', 'Wheel']
  Datatypes = ['Mushroom', 'Financial', 'Jester',
               'Statlog', 'Adult', 'Covertype', 'Census']
  nactions = [8,7,2,8,8,7,20,7,9,5]
  nactions = [2,8,8,7,20,7,9]

  outdir  = "./13_12"
  results = []
  Nruns=50
  tfn=400
  for i in xrange(350):
      iter = i%Nruns
      if iter == 0:
        FinalResults = [[]for i in xrange(len(algos))]
      data_type = Datatypes[int(i/Nruns)]
      file_name_out = os.path.join(outdir, data_type+'_index_'+str(iter)+'_train_freq_' + str(tfn) + '_mem_size_' + str(nactions[int(i/Nruns)
                                                                                                                        ]*100) +'_13_12')
      if tf.gfile.Exists(file_name_out):
        with tf.gfile.Open(file_name_out, "r") as f_out:
          res = f_out.readline()
        b = res.split('[')
        c = []
        for bb in b:
          if bb is not '':
           c.append(float(bb.split(']')[0]))
        for j in xrange(len(algos)):
          FinalResults[j].append(c[j])
      if iter == 49 and len(FinalResults[0])>0:
        print(len(FinalResults[0]))
        display_final_results2(algos,FinalResults, data_type)


if __name__ == '__main__':
    main()
