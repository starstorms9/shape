'''
This class is used to automatically create and organize training info.

For each run, it creates a new folder structure like this example:
    
0131-0411/
├── code_file.txt
├── configs.csv
├── logs
│   └── events.out.tfevents.1580472882.ip-172-31-26-36
├── models
│   ├── checkpoint
│   ├── ckpt-1.data-00000-of-00002
│   ├── ckpt-1.data-00001-of-00002
│   ├── ckpt-1.index
│   ├── ckpt-2.data-00000-of-00002
│   ├── ckpt-2.data-00001-of-00002
│   └── ckpt-2.index
└── plots
    ├── Original.png
    ├── Reconstruct 10.png
    └── Reconstruct 15.png
    
The top level folder is named DDDD-TTTT where DDDD is the date with month followed by date and TTT is the military time the folder was created.
The code_file.txt contains copies of the actual code from any input functions / classes.
The configs.csv takes in any local variables with the prefix cf_ and puts them into a csv.
The logs file is a tensorboard log file created with easy_tf2_log.py
The models folder contains all of the saved models from training.
The plots folder contains any output plots during training such as shape reconstructions.    
'''

#%% Imports
import os
import inspect
import csv

import utils as ut
import easy_tf2_log as etl
import tensorflow as tf
import configs as cf

#%% Logger class
class logger():
    def __init__(self, run_name='', root_dir=None, trainMode = False, txtMode = False):
        self.remote = cf.REMOTE
        self.training = trainMode or self.remote
        self.total_epochs = 1
        self.run_name = run_name

        self.new_run = (root_dir == None)
        if (self.new_run) :
            self.root_dir = cf.SHAPE_RUN_DIR
            self.root_dir = self.root_dir + ut.addTimeStamp()
        else :
            self.root_dir = root_dir
        
        if txtMode : self.root_dir = self.root_dir.replace('runs', 'txtruns')
        
        self.vox_in_dir = cf.VOXEL_FILEPATH
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
        