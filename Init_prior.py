#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:07:49 2022
Adding known priors for a better initialization !
@author: surya
TODO:
    Start with a prior = Image
    Train network until the output is in sync with the prior
    Use this as the start for FEM based optimization
    Also, check what the output of network is for different seeds
    
"""
#%%
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import logging

from neural_structural_optimization import topo_api
from neural_structural_optimization import models
from neural_structural_optimization import train
from neural_structural_optimization import problems

import xarray
import os
import time
import numpy as np
#%%
data_dir =  '/home/surya/Desktop/PhD_work/data/'

exp_num = 'test'
max_iterations = 150
conv_criteria = True
problem = problems.michell_centered_both(32, 64, 0.12)
#problem = problems.causeway_bridge(128,128,0.1,deck_level = 0.5)

prob_name = "michell_centered_both(32, 64, 0.12)"
opt = 'LBFGS'

seed_val = 1
#%%
# Information about the prior
# Encoded as an image - Material at the point of forces
#1. Image of uniform density!

#%%
# Layer activations
layer_ind = 1  # Index of required layer
args = topo_api.specified_task(problem)
model = models.CNNModel_c2dt(args =args, seed =1)
print("The layers are ", model.core_model.layers)
extractor = tf.keras.Model(inputs = model.core_model.inputs,
                           outputs =[model.core_model.layers[layer_ind].output] )
                            #cnn.get_layer(theNameYouWant).outputs
#model.core_model.layers[3].get_config() -- to get the activation of each layer
#%%

