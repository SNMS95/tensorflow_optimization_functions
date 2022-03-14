#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:39:34 2022

@author: surya
"""
import autograd.numpy as np
import tensorflow as tf

layers = tf.keras.layers

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
######################### Main Model Class
class Model(tf.keras.Model):

  def __init__(self, seed=None, args=None):
    super().__init__()
    set_random_seed(seed)
    self.seed = seed
    self.env = args

######################### Pixel Model Class
class PixelModel(Model):
  """
    The class for performing optimization in the input space of the functions.
    The initial parameters are chosen uniformly from [0,1] so as to ensure
        similarity across all functions
    TODO: May need to add the functionality to output denormalized bounds
  """
  def __init__(self, seed=None, args = None):
    super().__init__(seed)
    z_init = tf.random.uniform((1,args['dim']), minval = 0, maxval = 1.0)
    self.z = tf.Variable(z_init, trainable=True, dtype = tf.float32)#S:ADDED 

  def call(self, inputs=None):
    return self.z


def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net


def UpSampling2D(factor):
  return layers.UpSampling2D((factor, factor), interpolation='bilinear')
#changed interpolation to 'nearest'

def Conv2D(filters, kernel_size, **kwargs):
  return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)


class AddOffset(layers.Layer):

  def __init__(self, scale=1):
    super().__init__()
    self.scale = scale

  def build(self, input_shape):
    self.bias = self.add_weight(
        shape=input_shape, initializer='zeros', trainable=True, name='bias')

  def call(self, inputs):
    return inputs + self.scale * self.bias

######################### Simple FC architecture
class FCNN_simple(Model):
# Sigmoid for last layer, non trainable latent input with size =1!!
  def __init__(
      self,
      seed=0,
      args=None,
      latent_size=1,
      depth_scale = 2, # NUmber of layers
      width_scale = 1, # To scale the width of the hidden layers
      latent_scale=1.0, # Random normal with std_dev =  scale
      latent_trainable = False,
      kernel_init = tf.keras.initializers.GlorotUniform,
      activation=tf.keras.activations.tanh,
  ):
    super().__init__(seed, args)

    n_output = args['dim']    

    net = inputs = layers.Input((latent_size,), batch_size=1)
    num_neurons = n_output*width_scale   

    for _ in range(depth_scale):
      net = layers.Dense(num_neurons, kernel_initializer=kernel_init, activation =activation)(net)

    net = layers.Dense(n_output, kernel_initializer=kernel_init ,activation =
                       tf.keras.activations.sigmoid)(net)
    outputs = layers.Reshape([n_output])(net)
    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z', 
                        trainable= latent_trainable)

  def call(self, inputs=None):
    return self.core_model(self.z)
