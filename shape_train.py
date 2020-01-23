#%% Imports
import numpy as np
import time

import cvae as cv
import binvox_rw as bv

import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.figure_factory as FF
from tqdm import tqdm

import easy_tf_log as etl
import tensorflow as tf

#%% Setup file paths
model_save_filepath = '/home/starstorms/Insight/shape/models'
vox_in_dir = '/home/starstorms/Insight/ShapeNet/voxs'
vox_file_prefix = 'car'

remote = bool(int(input('\n\nRemote? (0 for no, 1 for yes): '))==1)
if (remote) :
    model_save_filepath = '/home/ubuntu/shape/models'
    vox_in_dir = '/home/ubuntu/data/cars'

load_weights = bool(int(input('Load weights? (0 for no, 1 for yes): '))==1)

vox_input_dim = 64
vox_size = 64
latent_dim = int(input('Latent dim: '))
num_models_load = 1300

#%% Helper functions
def loadBV(filename, coords = False, scale=1, reduction_factor=1) :
    with open(filename, 'rb') as f:
        model = bv.read_as_3d_array(f, coords, scale, reduction_factor)
    return model

def makeOBJ(name, verts, faces) :
    F = open(name,'w+')    
    faces = faces + 1
    for vert in verts : F.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
    for face in faces : F.write("f {} {} {}\n".format(face[0], face[1], face[2]))    
    F.close()
        
def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1)) :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title="Mesh", aspectratio=aspect)
    plot(fig)
  
def prepVox(voxin, pad=1):
    vox = voxin
    vox = np.pad(vox, pad)
    return vox

def minmax(arr) :
    print('Min: {} \nMax: {}'.format(np.min(arr), np.max(arr)))
    
def interp(vec1, vec2, divs=5, include_ends=True) :
    out = []
    amounts = np.array(range(divs+1))/divs if include_ends else np.array(range(1, divs))/divs
    for amt in amounts :
        interpolated_vect = vec1 * amt + vec2 * (1-amt)
        out.append( interpolated_vect )
    return np.stack(out)

#%% Load models and put into arrays
print("Loading info...")
voxs = np.zeros((num_models_load, vox_size, vox_size, vox_size, 1))

for i in tqdm(range(num_models_load)) :
    voxs[i,:,:,:,0] = loadBV('{}/{}{}.binvox'.format(vox_in_dir, vox_file_prefix, i+1), scale=1, reduction_factor= int(vox_input_dim/vox_size)).data
    
voxs = np.float32(voxs)

SHUFFLE_BUF = 1000
BATCH_SIZE = 32
print("Putting data into TF dataset...")
train_dataset = tf.data.Dataset.from_tensor_slices(voxs).shuffle(SHUFFLE_BUF).batch(BATCH_SIZE, drop_remainder=True)
  
#%% Make model and print info
print("Making model...")
model = cv.CVAE(latent_dim, vox_size)
model.setLR(6e-4)
model.printMSums()
loss = tf.keras.metrics.Mean()
total_epochs = 0

if (load_weights) :
    try:
        model.loadWeights(model_save_filepath, '_latest')
        print('\nWeights loaded successfully!\n')
    except Exception as e:
        print('\n\nWeights not found, using new model.\n\n')

#%% Train
def trainModel(epochs, sample_index=0) :
    print('\n\nStarting training...\n\n')
    global total_epochs
    losses = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            model.compute_apply_gradients(train_x)
        end_time = time.time()
    
        total_epochs = total_epochs + 1
        
        if epoch % 1 == 0:
            for test_x in train_dataset:
                loss(model.compute_loss(test_x))
            elbo = -loss.result()
            losses.append(elbo)
            print('Epoch: {}, Test set ELBO: {:.2f}, time elapsed for current epoch {:.2f}'.format(total_epochs, float(-elbo), float(end_time - start_time)))
        
        if epoch % 100 == 0:
            model.saveWeights(model_save_filepath, '_Epoch{:04d}'.format(total_epochs))
    
    model.saveWeights(model_save_filepath, '_latest')
    print("\n\nDone, weights saved to:\n{}".format(model_save_filepath))
    return losses

#%% 
print("Ready to start training model...")
for i in range(20) :
    epochs = int(input('\n\nEpochs to train for (0 to exit): '))
    if (epochs == 0) : break
    trainModel(epochs)

print('\n\nAll done!')