%load_ext autoreload
%autoreload 2
# Sync scripts
!sync_scripts.sh
# Sync runs
!sync_runs.sh 
# Delete emptry dirs in runs/
!find /home/starstorms/Insight/shape/runs/  -empty -type d -delete
!find /data/sn/all/runs -empty -type d -delete

import os
os.chdir('/data/sn/all/scripts/')

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
import inspect
import pickle
import tqdm

import gan as gan
import utils as ut
import logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF

import tensorflow as tf

#%% Setup
random.seed(488)

'''
0  Table   4379243   8436
1  Chair   3001627   6778
2  Lamp    3636649   2318
3  Faucet  3325088   744
4  Clock   3046257   651 
5  Bottle  2876657   498
6  Vase    3593526 or specifically 4522168   485
7  Laptop  3642806   460 
8  Knife   3624134   424 
'''

cf_cat_prefixes = ['04379243','03001627','03636649'] #,'03325088','03046257','02876657','03593526','03642806','03624134']
cf_vox_size = 32
cf_latent_dim = 100
cf_num_models_load = 12000
cf_batch_size = 256
cf_gen_lr = 0.0001
cf_dis_lr = 0.0004

#%%
lg = logger.logger() #root_dir = '/home/starstorms/Insight/shape/runs/0131-2024')
lg.writeConfig(locals(), [gan])

#%%
# vox_in_dir = '/home/starstorms/Insight/ShapeNet/all/'
train_dataset = ut.loadData(cf_vox_size, cf_num_models_load, lg.vox_in_dir, cf_cat_prefixes, cf_batch_size)

#%% Make gen model and show initial
model = gan.GAN(cf_latent_dim, cf_vox_size, cf_gen_lr, cf_dis_lr)
model.printMSums()
model.printIO()
lg.setupCP(encoder=model.dis_model, generator=model.gen_model, gen_opt=model.gen_opt, dis_opt=model.dis_opt)
model.setLR(cf_gen_lr, cf_dis_lr)

#%%
def train(dataset, epochs):
  print('\n\nStarting training...\n\n')
  gen_loss_metric = tf.keras.metrics.Mean()
  dis_loss_metric = tf.keras.metrics.Mean()
  
  for epoch in range(1, epochs):
    start = time.time()

    for vox_batch, labels in dataset:
        noise = tf.random.normal([cf_batch_size, cf_latent_dim])
        gen_loss, dis_loss = model.train_step(vox_batch, noise)
        gen_loss_metric(gen_loss)
        dis_loss_metric(dis_loss)
    
    gen_loss, dis_loss = gen_loss_metric.result(), dis_loss_metric.result()
    lg.logMetric(gen_loss, 'Gen Loss')
    lg.logMetric(dis_loss, 'Dis Loss')
    
    if (epoch % 5 == 0) :
        model.showVoxImages(lg.total_epochs, test_input=seed)
        
    if (epoch % 10 == 0) :
        disc_acc_fake, disc_acc_real = model.getDiscAcc(50, train_dataset, print_out=True)
        lg.logMetric(disc_acc_fake, 'Disc Acc Fake')
        lg.logMetric(disc_acc_real, 'Disc Acc Real')

    if (epoch % 15 == 0):
        lg.cpSave()
    
    print ('Epoch {} took {:.2f} sec. Gen Loss: {:.3f} Disc Loss: {:.5f}'.format(lg.total_epochs + 1, time.time()-start, gen_loss, dis_loss))
        
    if (ut.checkStopSignal()) :
        print('Stop signal received.')
        break
    
    lg.incrementEpoch()

#%% Define Training loop params
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, cf_latent_dim])

#%% Train model
train(train_dataset, 3000)

#%%
model.showVoxImages(0, num_examples=10)
model.getDiscAcc(200, train_dataset, print_out=True)
