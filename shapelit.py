#%% Imports
import numpy as np
# import pandas as pd
import streamlit as st
import skimage.measure as sm
import plotly.figure_factory as FF
# import plotly.express as px
# from plotly.offline import plot
# import plotly.graph_objects as go
from plotly import tools
from plotly.subplots import make_subplots

import cvae as cv
from utils import plotlySurf

# import tensorflow as tf

#%% Voxelize data with command line
model_save_filepath = '/home/starstorms/Insight/shape/models'
vox_in_dir = '/home/starstorms/Insight/ShapeNet/cars'
vox_file_prefix = 'car'

vox_input_dim = 64
vox_size = 64
latent_dim = 10
num_models_load = 10

#%%
def loadBV(filename, coords = False, reduction_factor=1) :
    with open(filename, 'rb') as f:
        model = read_as_3d_array(f, coords, reduction_factor)
    return model

def createMesh(vox, step=1, threshold = 0.5) :    
    vox = np.pad(vox, step)
    verts, faces, normals, values = sm.marching_cubes_lewiner(vox, 0.5, step_size=step)
    return verts, faces

def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1), plot_it=True, title='Mesh') :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title=title, aspectratio=aspect)
    return fig

class Voxels(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).
    dims, translate and scale are the model metadata.
    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.
    scale and translate relate the voxels to the original model coordinates.
    To translate voxel coordinates i, j, k to original coordinates x, y, z:
    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]
    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write(self, fp)

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp, fix_coords=True, reduction_factor=1):
    """ Read binary binvox format as array.
    Returns the model with accompanying metadata.
    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).
    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    
    if (reduction_factor > 1) :
        data = data[0::reduction_factor, 0::reduction_factor, 0::reduction_factor]
        
    return Voxels(data, dims, translate, scale, axis_order)
    
#%% Load models into numpy array and create TF dataset
@st.cache
def loadData() :
    voxs = np.zeros((num_models_load, vox_size, vox_size, vox_size, 1))
    for i in range(num_models_load) :
        voxs[i,:,:,:,0] = loadBV('{}/{}{}.binvox'.format(vox_in_dir, vox_file_prefix, i+1), reduction_factor= int(vox_input_dim/vox_size)).data
        
    voxs = np.expand_dims( np.float32(voxs), axis=1)
    # train_dataset = tf.data.Dataset.from_tensor_slices(voxs).batch(1, drop_remainder=True)
    return list(voxs)

samples = loadData()

#%% Make model and print info
@st.cache(allow_output_mutation=True)
def makeModel() :
    model = cv.CVAE(latent_dim, vox_size)
    model.setLR(5e-3)
    model.loadWeights(model_save_filepath + '', '_Epoch{:04d}'.format(1010))
    return model

model = makeModel()

def getAnchorVect(index=1) :
    return model.encode(samples[index], reparam=True).numpy()[0,:]

#%% Setup streamlit app
model_index = int(st.sidebar.number_input('Index of model to anchor on: ', value=0, min_value=0, max_value=len(samples)-1, step=1))
show_original = st.checkbox('Show Original Model', value=0)

anchor = getAnchorVect(model_index)
mods = np.zeros((anchor.shape))
mods = [st.sidebar.slider('Value {}: '.format(i), min_value= -3.0, max_value=3.0, step=0.1, value=0.0) for i in range(latent_dim)]
anchor = anchor + np.stack(mods)

v = model.sample( np.expand_dims(anchor, axis=0) )
v = v.numpy()[0,:,:,:,0]
vertsR, facesR = createMesh(v)
vertsO, facesO = createMesh(samples[model_index][0,:,:,:,0])

#%%
plot_dataR = plotlySurf(vertsR, facesR)
plot_dataO = plotlySurf(vertsO, facesO)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]])
for data in plot_dataO : fig.add_trace( data, row=1, col=1)
for data in plot_dataR : fig.add_trace( data, row=1, col=2)
fig.update_layout( title_text='Reconstructed vs Original Model', height=800, width=1600)
st.write(fig)

#%%
# fig = showMesh(vertsR, facesR, plot_it=False, title='Reconstructed: {}{}'.format(vox_file_prefix, model_index), aspect=dict(x=2, y=1, z=1))
# st.write(fig)

# if show_original :
#     verts, faces = createMesh(samples[model_index][0,:,:,:,0])
#     fig = showMesh(verts, faces, plot_it=False, title='Original: {}{}'.format(vox_file_prefix, model_index), aspect=dict(x=2, y=1, z=1))
#     st.write(fig)
