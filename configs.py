"""
This configuration file stores common variables for use across the program.
"""

#%% Imports
import os

#%% Properties
REMOTE = os.path.isdir('/data/sn/all')

if REMOTE :
    VOXEL_FILEPATH_LOCAL = 'REMOTE'
    SHAPE_RUN_DIR = '/data/sn/all/runs/'
    TXT_RUN_DIR = '/data/sn/all/txtruns/'
    DATA_DIR = '/data/sn/all/data/'

else :
    VOXEL_FILEPATH_LOCAL = ''
    META_DATA_CSV = '/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv'
    TAXONOMY_JSON = '/home/starstorms/Insight/ShapeNet/meta/taxonomy.json'
    SHAPE_RUN_DIR = '/home/starstorms/Insight/shape/runs'
    TXT_RUN_DIR = '/home/starstorms/Insight/shape/txtruns/'
    DATA_DIR = '/home/starstorms/Insight/shape/data'