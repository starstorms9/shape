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
from shutil import copyfile
import subprocess
from sys import getsizeof, stdout
import skimage.measure as sm
from scipy import spatial
import time
import json
import pandas as pd
import random
import inspect

import pickle
from tqdm import tqdm

import cvae as cv
import utils as ut
import logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns

import tensorflow as tf
remote = os.path.isdir('/data/sn/all')

#%% Setup
'''
0   04379243   Table    8436
1   03001627   Chair    6778
2   03636649   Lamp     2318
3   03325088   Faucet   744
4   03046257   Clock    651
5   02876657   Bottle   498
6   03593526   Vase     485
7   03642806   Laptop   460
8   02818832   Bed      233
9   03797390   Mug      214
10  02880940   Bowl     186
'''
random.seed(488)
tf.random.set_seed(488)
cf_cat_prefixes = ut.cf_cat_prefixes = ['04379243','03001627','03636649','03325088','03046257','02876657','03593526','03642806','02818832','03797390','02880940'] # ['04379243','03001627','03636649','03642806']  #,'03636649','03325088','03046257','02876657','03593526','03642806']
cf_num_classes = len(cf_cat_prefixes)
cf_vox_size = 64
cf_latent_dim = 128
cf_max_loads_per_cat = 10000 if remote else 50
cf_batch_size = 40
cf_learning_rate = 4e-4
cf_limits=[cf_vox_size, cf_vox_size, cf_vox_size]
ut.readMeta('/data/sn/all/meta/dfmeta.csv') if remote else ut.readMeta()

#%% Make model and print info
model = cv.CVAE(cf_latent_dim, cf_vox_size, cf_learning_rate, training=True)
model.setLR(cf_learning_rate)
model.printMSums()
model.printIO()

#%% Setup logger info
run_name = '0209-0306'
root_dir_local = '/home/starstorms/Insight/shape/runs/' + run_name
root_dir_remote = '/data/sn/all/runs/' + run_name
lg = logger.logger(root_dir = root_dir_remote if remote else root_dir_local)
# lg = logger.logger()
lg.setupCP(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer)
lg.restoreCP() #'/home/starstorms/Insight/shape/runs/0204-2220/models/ckpt-71')

#%% Load all data
# vox_in_dir = '/home/starstorms/Insight/ShapeNet/all/'
all_voxs, all_mids = ut.loadData(cf_vox_size, cf_max_loads_per_cat, lg.vox_in_dir, cf_cat_prefixes)

#%% Save base train data to file
save_dir = '/data/sn/all/data' if remote else '/home/starstorms/Insight/ShapeNet/data'
# np.save(os.path.join(save_dir, 'some_voxs.npy'), all_voxs, allow_pickle=True)
# np.save(os.path.join(save_dir, 'some_mids.npy'), all_mids, allow_pickle=True)

#%% Load base train data from file
all_voxs = np.load(os.path.join(save_dir, 'all_voxs.npy'), allow_pickle=True)
all_mids = np.load(os.path.join(save_dir, 'all_mids.npy'), allow_pickle=True)

#%% Setup datasets
voxs_stacked = np.stack(all_voxs, axis=0)
train_dataset = tf.data.Dataset.from_tensor_slices((voxs_stacked, all_mids))
for test_samples, test_labels in train_dataset.batch(50).take(1) : pass
test_samples = tf.cast(test_samples, dtype=tf.float32)

train_dataset, test_dataset = ut.splitData(train_dataset, 0.1)
train_dataset = train_dataset.batch(cf_batch_size, drop_remainder=True)
test_dataset = test_dataset.batch(cf_batch_size, drop_remainder=False)

total_train_batchs = 0
for _ in train_dataset : total_train_batchs = total_train_batchs + 1

#%% Show initial models
sample_index = 16
ut.plotVox(test_samples[sample_index], title='Original', threshold=0.5, limits=cf_limits, save_fig=False)
if (lg.total_epochs > 10) : ut.plotVox(model.reconstruct(test_samples[sample_index][None,...], training=False), limits=cf_limits, title='Recon')

#%% Training methods
def getTestSetLoss(dataset, batches=0) :
    test_losses = []
    for test_x, test_label in (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
        test_x = tf.cast(test_x, dtype=tf.float32)
        test_loss_batch = model.compute_loss(test_x)
        test_losses.append(test_loss_batch)
    return np.mean(test_losses)

def trainModel(epochs, display_interval=-1, save_interval=10, test_interval=10) :
    print('\n\nStarting training...\n')   
    model.training=True
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        losses = []
        batch_index = 1
        for train_x, label in train_dataset :
            train_x = tf.cast(train_x, dtype=tf.float32)
            loss_batch = model.trainStep(train_x)
            losses.append(loss_batch)
            stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
            stdout.flush()
            batch_index = batch_index + 1
        elbo = np.mean(losses)
        
        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            ut.showReconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=False, save_fig=lg.remote, limits=cf_limits)
        
        if epoch % test_interval == 0:
            test_loss = getTestSetLoss(test_dataset, 2)
            print('   TEST LOSS: {:.1f}    for epoch: {}'.format(test_loss, lg.total_epochs))
            lg.logMetric(test_loss, 'test loss')
            
        if epoch % save_interval == 0:
            lg.cpSave()
            
        if (ut.checkStopSignal()) : break        

        print('Epoch: {}   Train loss: {:.1f}   Epoch Time: {:.2f}'.format(lg.total_epochs, float(elbo), float(time.time() - start_time)))
        lg.logMetric(elbo, 'train loss')
        lg.incrementEpoch()
    return

#%% Train
lg.writeConfig(locals(), [cv.CVAE, cv.CVAE.__init__])
lg.updatePlotDir()
trainModel(5000, display_interval=2, save_interval=5, test_interval=2)







#%% Show reconstructed models
model.training = False
for i in range(30) : ut.showReconstruct(model, test_samples, index=i, title='Index: {}'.format(i), show_original=True, show_reconstruct=True, limits=cf_limits)

#%% Generate new designs
num_examples_to_generate = 5
vect_shape = tuple([num_examples_to_generate] + list(model.gen_model.input_shape[1:]))
random_vector_for_generation = tf.random.normal(shape=vect_shape)
gen_vect = random_vector_for_generation

v = model.sample(gen_vect)
v = v.numpy()[:,:,:,:,0]

for i, sample in enumerate(v) :
    ut.plotVox(sample, step=1, threshold=0.5, title='Index {}'.format(i), limits=cf_limits)

#%% See some reconstructed models    
'''
0   04379243   Table    8436  :  1   03001627   Chair    6778  :  2   03636649   Lamp     2318
3   03325088   Faucet   744   :  4   03046257   Clock    651   :  5   02876657   Bottle   498
6   03593526   Vase     485   :  7   03642806   Laptop   460   :  8   02818832   Bed      233
9   03797390   Mug      214   :  10  02880940   Bowl     186
'''
model.training = False
num_to_get = 10
cat_label_index = -2
anchors, labels = [],[]
for anchor, label in train_dataset.unbatch().shuffle(100000).take(num_to_get*50) :
    catid = -1
    try: catid = cf_cat_prefixes.index('0{}'.format(ut.getMidCat(label.numpy().decode())))
    except : print('not found\n ', label.numpy().decode())
    if (catid == cat_label_index or cat_label_index==-2) :
        anchor = tf.cast(anchor, dtype=tf.float32)
        anchors.append(anchor)
        labels.append(label)
    if (len(anchors) >= num_to_get) :
        break
    
anchor_vects = [model.encode(anchors[i][None,:,:,:], reparam=True) for i in range(len(anchors))]
v = [model.sample(anchor_vects[i]).numpy()[0,...,0] for i in range (len(anchors))]

for i, sample in enumerate(v) :
    print('Index: {}   Mid: {}'.format(i, labels[i].numpy().decode()))
    ut.plotVox(anchors[i].numpy()[...,0], step=1, threshold=0.5, title='Index {} Original'.format(i), limits=cf_limits)
    ut.plotVox(v[i], step=2, threshold=0.5, title='Index {} Reconstruct'.format(i), limits=cf_limits) 
    
print([mid.numpy().decode() for mid in labels])

#%% Interpolate between 2 set reconstructions
index1, index2 = 3,5
mids_string = ' {} , {} '.format(labels[index1].numpy().decode(), labels[index2].numpy().decode())
print(mids_string)
interp_vects = ut.interp(anchor_vects[index1].numpy(), anchor_vects[index2].numpy(), divs = 10)

v = model.sample(interp_vects)
v = v.numpy()[:,:,:,:,0]
for i, sample in enumerate(v) :    
    ut.plotVox(sample, step=1, threshold=0.5, limits=cf_limits, show_axes=False )

#%% Interpolate from 1 fixed vect to random direction
index = 7
vect2 = anchor_vects[index] + (np.random.rand(anchor_vects[0].shape[0]) - 0.5) * 10
interp_vects = ut.interp(vect2, anchor_vects[index], divs = 10)

v = model.sample(interp_vects)
v = v.numpy()[:,:,:,:,0]
for i, sample in enumerate(v) :    
    ut.plotVox(sample, step=1, threshold=0.5, limits=cf_limits, show_axes=False, )

#%% Go on a journey through shapetime
model.training=False
good_ones = []
journey_length = 40
start_index = 1510
vects_sample = 100
max_dist = 8
interp_points = 8
plot_step = 2

journey_vecs = []
visited_indices = [start_index]
journey_mids = []

mids = list(shape2vec.keys())
vecs = np.array([shape2vec[m] for m in mids])
vec_tree = spatial.KDTree(vecs)
start_vect = shape2vec[mids[start_index]]
journey_mids.append(mids[start_index])

for i in range(journey_length) :
    n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist)
    if len(shape2vec) in close_ids :
        n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist*3)    
    close_ids = list(close_ids)  #[:1000]
    
    for index in sorted(close_ids, reverse=True):
        if index in visited_indices:
            close_ids.remove(index)
    
    next_index = random.choice(close_ids)
    next_vect = vecs[next_index]
    visited_indices.append(next_index)
    interp_vects = ut.interp(next_vect, start_vect, divs = interp_points)
    journey_vecs.extend(interp_vects)
    start_vect = next_vect
    journey_mids.append(mids[next_index])
    
journey_voxs = np.zeros(( len(journey_vecs), cf_vox_size, cf_vox_size, cf_vox_size))
for i, vect in enumerate(journey_vecs) :
    journey_voxs[i,...] = model.decode(vect[None,...], apply_sigmoid=True)[0,...,0]

for i, vox in enumerate(journey_voxs) :
    ut.plotVox(vox, step=plot_step, limits = cf_limits, title='', show_axes=False)    
    
#%% Sample for some starting points
# i = start_index
for i in np.random.randint(0, len(shape2vec), size=100) :
    vox = shapemodel.decode(shape2vec[mids[i]][None,...], apply_sigmoid=True)[0,...,0]    
    ut.plotVox(vox, step=2, limits = cf_limits, title=i)
    
#%% Go on a directed trajectory
model.training=False
good_ones = [1781, 557, 574, 12418, 9771, 4819, 13419, 13836, 5810, 14922, 9105]
journey_length = 10
start_index = 8213
end_index = 4002
vects_sample = 200
interp_points = 8
percent_step_size = 0.0
plot_step = 2

journey_vecs = []
visited_indices = [start_index]

mids = list(shape2vec.keys())
vecs = np.array([shape2vec[m] for m in mids])
vec_tree = spatial.KDTree(vecs)
start_vect = shape2vec[mids[start_index]]
end_vect = shape2vec[mids[end_index]]

for i in range(journey_length) :
    step_vec = (1-percent_step_size)*start_vect + (percent_step_size)*end_vect
    percent_step_size = (i+1) / journey_length
    
    n_dists, close_ids = vec_tree.query(step_vec, k = vects_sample)
    close_ids = list(close_ids)  #[:1000]
    
    for index in close_ids:
        if index in visited_indices:
            close_ids.remove(index)
    
    next_index = random.choice(close_ids)
    next_vect = vecs[next_index]
    visited_indices.append(next_index)
    interp_vects = ut.interp(next_vect, start_vect, divs = interp_points)
    journey_vecs.extend(interp_vects)
    start_vect = next_vect
    
journey_voxs = np.zeros(( len(journey_vecs), cf_vox_size, cf_vox_size, cf_vox_size))
for i, vect in enumerate(journey_vecs) :
    journey_voxs[i,...] = model.decode(vect[None,...], apply_sigmoid=True)[0,...,0]

for i, vox in enumerate(journey_voxs) :
    ut.plotVox(vox, step=plot_step, limits = cf_limits, title='', show_axes=False)  
    

   
    
    
    
    
    
    
    
    
    
    
#%% Run model on data to get latent vects and loss not using train dataset
shape2loss = {}
shape2vec = {}
for sample, label in tqdm(zip(all_voxs, all_mids), unit_scale=True, desc="Saving shape 2 vec: ", unit=" encodes", total=len(all_voxs))  :
    sample = tf.cast(sample, dtype=tf.float32)
    shape2vec[label] = model.encode(sample[None,...], reparam=True).numpy()[0]
    shape2loss[label] = model.compute_loss(sample[None,...]).numpy()    
    
rdir = lg.root_dir
f = open(os.path.join(rdir,"shape2vec.pkl"),"wb")
pickle.dump(shape2vec,f)
f.close()

f = open(os.path.join(rdir,"shape2loss.pkl"),"wb")
pickle.dump(shape2loss,f)
f.close()

#%% Read text2vec pickle
infile = open(os.path.join(rdir,"shape2vec.pkl"),'rb')
shape2vec = pickle.load(infile)
infile.close()

#%% Load pickle file
vect = new_dict['17c5c22c9ab97788db67d56f11b1bed6'].reshape((1, 100))
gen_model = model.decode(vect).numpy()[0,...,0]
ut.plotVox(gen_model)














