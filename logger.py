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
class logger():
    def __init__(self, run_name='', root_dir=None, trainMode = False, txtMode = False):
        self.remote = os.path.isdir('/data/sn/all') 
        self.training = trainMode or self.remote
        self.total_epochs = 1
        self.run_name = run_name
        
        local_root_dir = '/home/starstorms/Insight/shape/runs/'
        local_vox_in_dir = '/home/starstorms/Insight/ShapeNet/all/'
        remote_root_dir = '/data/sn/all/runs/'
        remote_vox_in_dir = '/data/sn/all/all/'

        self.new_run = (root_dir == None)
        if (self.new_run) :
            self.root_dir = remote_root_dir if self.remote else local_root_dir
            self.root_dir = self.root_dir + ut.addTimeStamp()
        else :
            self.root_dir = root_dir
        
        if txtMode : self.root_dir = self.root_dir.replace('runs', 'txtruns')
        
        self.vox_in_dir = remote_vox_in_dir if self.remote else local_vox_in_dir
        self.plot_out_dir = self.root_dir + '/plots'
        self.model_saves = self.root_dir + '/models'
        self.tblogs = self.root_dir + '/logs'
        self.updatePlotDir()
        
    def checkMakeDirs(self) :
        if os.path.isdir(self.root_dir) : return
        if (self.training) :
            os.mkdir(self.root_dir)
            os.mkdir(ut.plot_out_dir)
            os.mkdir(self.model_saves)
            os.mkdir(self.tblogs)
            etl.set_dir(self.tblogs)        
        
    def reset(self):
        self.total_epochs = 1
        
    def logMetric(self, metric, name):
        self.checkMakeDirs()
        etl.tflog(name, metric, step=self.total_epochs)
        
    def incrementEpoch(self) :
        self.total_epochs = self.total_epochs + 1        
    
    def setupCP(self, generator, encoder, opt) :
        if encoder == None :
            self.checkpoint = tf.train.Checkpoint(optimizer = opt, generator = generator)
        else : 
            self.checkpoint = tf.train.Checkpoint(optimizer = opt, generator = generator, encoder = encoder)
        self.cpmanager = tf.train.CheckpointManager(self.checkpoint, directory=self.model_saves, max_to_keep= (3 if (self.remote) else 999))

    def cpSave(self):
        self.cpmanager.save()
        
    def restoreCP(self, path=None):
        if not os.path.isdir(self.root_dir) :
            print('No folder found, not restored.')
            return
        if (path==None) :
            status = self.checkpoint.restore(self.cpmanager.latest_checkpoint)
            print('Latest model chkp path is : {}'.format(self.cpmanager.latest_checkpoint))
            return status
        else :
            status = self.checkpoint.restore(path)
            print('Latest model chkp path is : {}'.format(status))
            return status
    
    def writeConfig(self, variables, code):
        self.checkMakeDirs()
        if (not self.training) :
            print('Cannot, in read only mode.')
            return
        
        if (len(code) > 0) :
            code_file = open(os.path.join(self.root_dir, "code_file.txt"), "w")
            for source in code :
                code_file.write(repr(source) + '\n\n')
                code_file.write(inspect.getsource(source) + '\n\n')
            code_file.close()
        
        filtered_vars = {key:value for (key,value) in variables.items() if (key.startswith('cf_'))}
        w = csv.writer(open(os.path.join(self.root_dir, "configs.csv"), "w"))
        for key, val in filtered_vars.items():
            w.writerow([key, val])      
    
    def updatePlotDir(self):
        ut.plot_out_dir = self.plot_out_dir    