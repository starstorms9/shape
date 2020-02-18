#%% Imports
import numpy as np
import os
import subprocess
from sys import getsizeof
import skimage.measure as sm
import time
import json
import pandas as pd
import random

import cvae_dev as cv
import binvox_rw as bv
import utils as ut
import model_helper as ml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF

import tensorflow as tf

random.seed(488)

#%% Voxelize data with command line
model_save_filepath = '/home/starstorms/Insight/shape/models'
vox_in_dir = '/home/starstorms/Insight/ShapeNet/all'
# cat_prefixes = ['04256520','02958343']
cat_prefixes = ['02958343']

vox_size = 64
latent_dim = 10
num_models_load = 1300
batch_size = 128

#%% Load models into numpy array and create TF dataset
train_dataset = ut.loadData(vox_size, num_models_load, vox_in_dir, cat_prefixes, batch_size)

#%% Make model and print info    
model = ut.makeModel(vox_size, latent_dim, learning_rate=1e-3)

samples = list(train_dataset.unbatch().batch(1).skip(10).take(50))

#%% Train
sample_index = 1
ut.showReconstruct(model, samples, sample_index, title=ml.total_epochs, show_reconstruct=False)
losses = ml.trainModel(model, train_dataset, 100, samples, sample_index=sample_index, display_interval=3)
plt.plot(losses)

#%%
model.saveWeights(model_save_filepath, '_Epoch{:04d}'.format(ml.total_epochs))
model.loadWeights(model_save_filepath + '/p2xlarge', '_Epoch{:04d}'.format(510))

#%% Show reconstructed models
for i in range(len(samples)) : ut.showReconstruct(model, samples, index=i, show_original=True, show_reconstruct=True)

#%% Generate new designs
num_examples_to_generate = 5
vect_shape = tuple([num_examples_to_generate] + list(model.generative_net.input_shape[1:]))
random_vector_for_generation = tf.random.normal(shape=vect_shape)
gen_vect = random_vector_for_generation

v = model.sample(gen_vect)
v = v.numpy()[:,:,:,:,0]

for i, sample in enumerate(v) :
    ut.plotVox(sample, step=1, threshold=0.5, title='Index {}'.format(i))

#%% See some reconstructed models
anchors = list(train_dataset.unbatch().batch(1).shuffle(1000).take(10))
anchor_vects = [model.encode(anchors[i], reparam=True).numpy()[0,:] for i in range(len(anchors))]
anchor_vects = np.stack(anchor_vects)

v = model.sample(anchor_vects)
v = v.numpy()[:,:,:,:,0]
for i, sample in enumerate(v) :
    ut.plotVox(sample, step=1, threshold=0.5, title='Index {}'.format(i))

#%% Interpolate between 2 set reconstructions
index1, index2 = 4, 8
interp_vects = ut.interp(anchor_vects[index1], anchor_vects[index2], divs = 20)

v = model.sample(interp_vects)
v = v.numpy()[:,:,:,:,0]
for i, sample in enumerate(v) :    
    ut.plotVox(sample, step=1, threshold=0.5, limits=[60, 30, 30], show_axes=False )
    
#%% Interpolate from 1 fixed vect to random direction
index = 3
vect2 = anchor_vects[index] + (np.random.rand(anchor_vects[0].shape[0]) - 0.5) * 10
interp_vects = ut.interp(vect2, anchor_vects[index], divs = 10)

v = model.sample(interp_vects)
v = v.numpy()[:,:,:,:,0]
for i, sample in enumerate(v) :    
    ut.plotVox(sample, step=1, threshold=0.5, limits=[60, 30, 30], show_axes=False, )

#%% Create gif from images in folder with bash and ImageMagick (replace XX with max number of images or just set high and error)
# !convert -delay 10 -loop 0 *_{0..XX}.png car2truck.gif

#%%
prefix = 'sofa'
in_fp =  '/home/starstorms/Insight/ShapeNet/sofas_in'
out_fp = '/home/starstorms/Insight/ShapeNet/sofas'
file_name_vox = 'model_normalized.solid.binvox'
file_name_obj = 'model_normalized.obj'
file_name = file_name_vox
file_ext = '.' + str.split(file_name, sep='.')[-1]

files_to_move = [ '{}/{}/models/{}'.format(in_fp, folder, file_name) for folder in os.listdir(in_fp) ]
move_to = [ '{}/{}{}{}'.format(out_fp, prefix, i+1, file_ext) for i in range(len(files_to_move)) ]

for i in range(len(files_to_move)) :
    os.rename(files_to_move[i], move_to[i])
    
  

