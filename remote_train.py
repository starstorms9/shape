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
# import easy_tf_log as etl

import tensorflow as tf

random.seed(488)

#%% Set up parameters for this run
model_save_filepath = '/data/sn/all/models'
ml.model_save_filepath = model_save_filepath
vox_in_dir = '/data/sn/all/solid.binvox'
# cat_prefixes = ['04256520','02958343']
cat_prefixes = ['02958343']

load_weights = bool(int(input('Load weights? (0 for no, 1 for yes): '))==1)
if (load_weights) : total_epochs = input('Load weights? (0 for no, 1 for yes): ')
else :              total_epochs = 0

num_models_load = int(input('Number of models to load: '))
batch_size = int(input('Batch size: '))

vox_size = 64
latent_dim = 10
sample_index = 1

ut.remote = True
ut.plot_out_dir = '/data/sn/all/plots'

#%% Set up training
print("Loading data...")
train_dataset = ml.loadData(vox_size, num_models_load, model_save_filepath, vox_in_dir, cat_prefixes, batch_size)
print("Making model...")
model = ml.makeModel(vox_size, latent_dim, learning_rate=1e-3)

if (load_weights) :
    model.loadWeights(model_save_filepath + '', '_Epoch{:04d}'.format(total_epochs))

loss = tf.keras.metrics.Mean()
samples = list(train_dataset.unbatch().batch(1).take(50))

#%% Train model
print("Ready to start training model...")
ut.showReconstruct(model, samples, sample_index, title='0', show_reconstruct=False)

for i in range(20) :
    epochs = int(input('\n\nEpochs to train for (0 to exit): '))
    if (epochs == 0) : break
    ml.trainModel(model, train_dataset, epochs, samples = samples, display_interval=3, save_interval=5,)

print('\n\nAll done!')

#%%
model.saveWeights(model_save_filepath, '_latest'.format(total_epochs))
