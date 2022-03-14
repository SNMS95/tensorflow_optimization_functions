#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:25:53 2022
Neurally reparameterized optimization of functions
@author: surya
"""
#%%
#Imports
# Add the directory of the files for easy import
import sys
sys.path.insert(1,"/home/surya/Desktop/PhD_work/neural-structural-optimization/PHD_1/")

import os
import obj_models as models # For using tensorflow models
import obj_funcs as fns # for using various objective functions
import obj_utilities as util # Other utilities for optimization

import xarray
import logging
import time
import numpy as np
import tensorflow as tf
#%%
#User choices
data_dir =  '/media/surya/surya_2/phd1/data/' # to save the experiment details
seeds_model = np.arange(1,16) # Seeds for initiualizing the model
max_iterations = 300
optimizer = 'lbfgs' # or 'gd'/ 'adam' /...
opt_args = {'m':10} #extra keyword arguments for the optimizer
model_args ={'depth_scale' : 3, 'width_scale':200,'latent_trainable':True}

dim = 2 # The dimension of the function
seed_off = 101 #Seed is for setting the offset only
fn = fns.Levy(seed = seed_off, dim=dim) # Choose a function 
#Adjust offset to zero using fn.o = 0.0 if needed
#%%
if optimizer == 'lbfgs':
    opt_fn = util.train_lbfgs
else:
    opt_fn = util.train_tf_optimizer
args = fn.__dict__ # Extract function details;to be passed to the model
exp_name = fn.name + "_"+str(fn.seed) # Name of the experiment
# Log experiment details
logging.basicConfig(filename='extra_data_opt.log', level=logging.DEBUG)
logging.info("Experiment No: %s" %(exp_num))
logging.info("Problem: %s" %(prob_name))
logging.info("Max_iterations: %s" %(str(max_iterations)))
logging.info("Seed: %s" %(str(seed)))
logging.info("Penality: %s" %(str(args['penal'])))
logging.info('Optimizer: %s' %(opt))
#redirection to appropriate folder
os.chdir(data_dir)
if not os.path.exists(str(exp_name)):
    os.makedirs(str(exp_name))
os.chdir( data_dir+str(exp_name))
#%%
s_loss = []
time_taken = []
indices = {}
loss_dim = []
s_alpha = {}
for j,seed in enumerate(seeds_model):
    os.chdir(data_dir+str(exp_name))    
    tf.keras.backend.clear_session() # Clear GPU of all models     
    folder1 = str(exp_name) +'_model_' + str(seed) 
    if not os.path.exists(folder1):
        os.makedirs(folder1)        
    path = os.getcwd() + '/' + folder1
    os.chdir(path)
    
    #create a model
    model = models.FCNN_simple(seed = seed, args = args,**model_args) # Done
    #model = models.PixelModel(seed = 1, args = args)       
         
    #train
    if os.path.exists("/home/surya/Desktop/n_iterations_lbfgs.txt"):
        os.remove("/home/surya/Desktop/n_iterations_lbfgs.txt")
    start_time = time.time()
    ds, ind = opt_fn(model, fn, max_iterations, **opt_args)
    dur = (time.time() - start_time)/60 
    #Delete temp file created
    os.remove("/home/surya/Desktop/n_iterations_lbfgs.txt")   
    
    indices[str(seed)] = ind
    ds.to_netcdf(folder1 +'.nc')
    print("Seed value = " + str(seed))
    print('Loss -----', ds.loss.values[-1])
    
    len_loss = len(ds.loss.values)
    if len_loss < max_iterations+1:
        lastval = np.min(ds.loss.values)#the minimum value
        n = max_iterations+1 -len_loss
        s_loss.append(np.concatenate([ds.loss.values,np.full(n,lastval)]))
    elif len_loss > max_iterations+1:
        s_loss.append(ds.loss.values[:max_iterations+1])                    
    else:
        s_loss.append(ds.loss.values)

    loss_dim.append((len_loss,len(s_loss[j])))  
    
    print('Time for seed: ',dur)
    time_taken.append((seed,dur))
    if seed == seeds_model[-1]:
        pass
    else:
        del model
  
#%%
#save relevant experiment details as a dictionary
#indices, times
os.chdir(data_dir+str(exp_num))
exp_details = dict()
exp_details['indices'] = indices #List(indices1,indices2,...)
exp_details['seed_time'] = time_taken#(seed,time)
exp_details['loss_dim'] = loss_dim#(length of loss output from algorithm,
                                    #Final length of the loss )
import pickle
pickle.dump(exp_details,open(str(exp_num)+'_expt_details','wb'))

steps = np.arange(max_iterations+1)
fds = xarray.Dataset({
    'loss':(('seed','step'),s_loss),
},coords = {'seed' :seed_val,'step':steps} )

fds.to_netcdf(str(exp_num)+'.nc')
#%%
data =  ds.output.isel(step =slice(0,None)).values
fn.plot_contour(data =data, zoom= False, px = 50)

#%%
import numpy as np
x = np.array([fn.offset(x) for x in data])
fn.plot3d(data =x)
#%%
model(None)
#%%