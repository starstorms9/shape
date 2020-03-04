'''
This configuration file stores common variables for use across the program.
It uses a simple file exists check to determine whether or not it is running remotely or locally
and changes the filepaths accordingly for convenience.
'''

#%% Imports
import os

#%% Properties
# This folder path should only exist on the remote machine
REMOTE = os.path.isdir('/data/sn/all')

if REMOTE :     # Running on EC2 instance or similar
    META_DATA_CSV = '/data/sn/all/meta/dfmeta.csv'
    VOXEL_FILEPATH = '/data/sn/all/all/'
    SHAPE_RUN_DIR = '/data/sn/all/runs/'
    TXT_RUN_DIR = '/data/sn/all/txtruns/'
    DATA_DIR = '/data/sn/all/data/'

else :          # Running locally 
    META_DATA_CSV = '/home/starstorms/Insight/shape/data/dfmeta.csv'
    '''
    This is the directory file that stores all of the metadata information gathered and analyzed by the program to generate descriptions.
    It is made by the program and shouldn't require additional action but is a useful resource to inspect manually    
    '''    
    
    VOXEL_FILEPATH = '/home/starstorms/Insight/ShapeNet/all'
    '''
    This is the location of the data from the ShapeNetCore database downloaded from here: https://www.shapenet.org/
        (note that to actually download the data you need an approved account)
    Note that while the folder structure from the zip file is required, only the model_normalized.solid.binvox files are necessary to 
        actually be present in the models folder. All of the other data in the ShapeNetCore zip file is not used so it doesn't need to be extracted.
    
    Example file tree setup:
        
    VOXEL_FILEPATH
    ├── 02946921                                    (synsetid, aka the object category label)
    │   └── 100c5aee62f1c9b9f54f8416555967          (model hash id, the label for the specific object)
    │       └── models                              (the directory with the models)
    │           └── model_normalized.solid.binvox   (the solid voxelized model, 128 voxel cube size)
    │   ...     
    │
    ├── 02747177
    │   └── 1b7d468a27208ee3dad910e221d16b18
    │       └── models
    │           └── model_normalized.solid.binvox 
        ... 
    '''
    
    SHAPE_RUN_DIR = '/home/starstorms/Insight/shape/runs'
    TXT_RUN_DIR = '/home/starstorms/Insight/shape/txtruns/'
    '''
    These are the run log and model checkpoint folders. This folder structure is generated and managed by the logger.py class.

    Example run directory tree structure:
        
    RUN_DIR
    ├── 0217-0434
    │   ├── code_file.txt
    │   ├── configs.csv
    │   ├── logs
    │   │   └── events.out.tfevents.1581942983.ip-172-31-21-198
    │   ├── models
    │   │   ├── checkpoint
    │   │   ├── ckpt-161.data-00000-of-00002
    │   │   ├── ckpt-161.data-00001-of-00002
    │   │   ├── ckpt-161.index
    │   │   └── epoch_161.h5
    │   └── plots
    ├── 0217-0437
    │   ├── code_file.txt
    │   ├── configs.csv
    │   ├── logs
    │   │   └── events.out.tfevents.1581943124.ip-172-31-24-21
    │   ├── models
    │   │   ├── checkpoint
    │   │   ├── ckpt-258.data-00000-of-00002
    │   │   ├── ckpt-258.data-00001-of-00002
    │   │   ├── ckpt-258.index
    │   │   └── epoch_258.h5
    │   └── plots
        ...
    '''    
    
    DATA_DIR = '/home/starstorms/Insight/shape/data'
    '''
    This folder is used to cache various computation and memory intensive generated files like the randomized descriptions of objects.
    '''   
    
    RENDERS_DIR = '/home/starstorms/Insight/ShapeNet/renders'
    '''
    This folder is used to store rendered images of the models for quick and easy viewing and for use in the streamlit app.
    Primarily used when inspecting the quality of generated descriptions.
    '''
        
    PARTNET_META_STATS_DIR = '/home/starstorms/Insight/ShapeNet/partnetmeta/stats/'
    '''
    This folder contains all of the metadata that is used to generate the descriptions.
    It is extracted from the PartNet data that can be downloaded from the ShapeNet website (approved account required).
    While the full dataset is extremely large when extracted, only the .json files are required to actually be extracted 
        into the same folder structure to save space.
        
    Specifically, only the meta.json and result_after_merging.json files are necessary.
    '''
    
    
    
