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
import binvox_rw as bv
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

#%% Create gifs
prefix = 'sofa'
in_fp =  '/home/starstorms/Insight/ShapeNet/sofas_in'
out_fp = '/home/starstorms/Insight/ShapeNet/sofas'
file_name_vox = 'model_normalized.solid.binvox'
file_name_obj = 'model_normalized.obj'
file_name = file_name_vox
file_ext = '.' + str.split(file_name, sep='.')[-1]

files_to_move = [ '{}/{}/models/{}'.format(in_fp, folder, file_name) for folder in os.listdir(in_fp) ]
move_to = [ '{}/{}{}{{}'.format(out_fp, prefix, i+1, file_ext) for i in range(len(files_to_move)) ]

for i in range(len(files_to_move)) :
    os.rename(files_to_move[i], move_to[i])

#%% Look at the difference between original and grown voxel models
for sample, _ in train_dataset.unbatch().shuffle(10000).take(3) :
    vox = sample[...,0].numpy()
    sparsity = ut.getSparsity(vox)
    ut.plotVox(vox, limits=cf_limits, title='Original\n Sparse: {:.2f}'.format(100*sparsity))
    if (sparsity > .00) : continue    
    grown = ut.growVox(vox, amount=0.5)
    sparsity_grown = ut.getSparsity(grown)
    ut.plotVox(grown, limits=cf_limits, title='Grown\n Sparse: {:.2f}'.format(100*sparsity_grown))
     

#%%
import shutil
import tqdm
rdir = '/home/starstorms/Insight/ShapeNet/partnetmeta/renders/data_v0'
outdir = '/home/starstorms/Insight/ShapeNet/partnetmeta/pics'

counter = 0
for root, dirs, files in os.walk(rdir):
    for file in files:
        if file == '0.png' :
            fullpath = os.path.join(root, file)
            annoid = fullpath.split('/')[-3]
            name = str(annoid)
            try : name = dfall[dfall.annoid==str(annoid)].mid.values[0]
            except : pass
            newpath = os.path.join(outdir, name+'.png')            
            
            # print('Ano: {}\nNew: {}\n'.format(annoid, newpath))
            shutil.copyfile(fullpath, newpath)
            counter = counter + 1
            if (counter % 1000 == 0) : print(counter)
    
count = 0        
fps = os.listdir(outdir)
for file in fps :
    if len(file) > 20 :
        count = count + 1
print('{} / {}'.format(count, len(fps)))