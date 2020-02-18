#%% Imports
import numpy as np
import os
import subprocess
from sys import getsizeof
import skimage.measure as sm
import time
import datetime
import json
import pandas as pd
import random
import subprocess
import inspect
import csv

import cvae_dev as cv
import binvox_rw as bv
import utils as ut

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF
import easy_tf_log as etl

import tensorflow as tf

#%%
def trainModel(model, train_dataset, epochs, samples, logger, sample_index=0, display_interval=-1, save_interval=10) :
    print('\n\nStarting training...\n\n')
    loss = tf.keras.metrics.Mean()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, _ in train_dataset:
            model.compute_apply_gradients(train_x)
        end_time = time.time()
            
        if epoch % 1 == 0:
            for test_x, _ in train_dataset :
                loss(model.compute_loss(test_x))
            elbo = loss.result()
            logger.logMetric(elbo)
            print('Epoch: {}, Test set ELBO: {:.2f}, time elapsed for current epoch {:.2f}'.format(logger.total_epochs, float(elbo), float(end_time - start_time)))
        
        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            ut.showReconstruct(model, samples, title=logger.total_epochs, index=sample_index, show_original=False, save_fig=True)
        
        if epoch % save_interval == 0:
            logger.cpSave()
            
        if (ut.checkStopSignal()) :
            break
    return
   
        
    
        
        
        