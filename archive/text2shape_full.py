#%% Imports
import numpy as np
import os
# os.chdir('/home/ubuntu/sn/scripts')
import subprocess
from sys import getsizeof
import skimage.measure as sm
from scipy import spatial
import time
import json
import pandas as pd
import random
import inspect
import pickle
import re
import tqdm

import cvae_dev as cv
import binvox_rw as bv
import utils as ut
import model_helper as ml
import text as tx
import logger
import augment as ag

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
np.set_printoptions(precision=3, suppress=True)
remote = os.path.isdir('/data/sn/all')

#%%
cf_max_length = 130
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'

#%% Process data
rdir_local = '/home/starstorms/Insight/shape/runs/0206-1623'
rdir_remote = '/data/sn/all/runs/0206-1623'
infile = open(os.path.join(rdir_remote if remote else rdir_local,"shape2vec_72.pkl"),'rb')
shape2vec = pickle.load(infile)
infile.close()

latent_dim = shape2vec[list(shape2vec.keys())[0]].shape[0]
meta_fp_local = '/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv'
meta_fp_remote = '/data/sn/all/meta/dfmeta.csv'
dfmeta = pd.read_csv(meta_fp_remote if remote else meta_fp_local)
dfdesc = dfmeta[dfmeta.desc.notnull()]

#%% Prepare text for model
augdescs_all = ag.augmentMutli(dfdesc.mid.values)
vocab = set()
max_words = 0
for desc in augdescs_all :    
    split = desc.split()
    vocab.update(split)
    max_words = max(max_words, len(split))    
vocab = sorted(vocab)
    
# shape = (len(dfdesc), 1, cf_max_length, latent_dim)

#%%
dall = []
for mid in tqdm.tqdm(dfdesc.mid, total=len(dfdesc.mid)) :
    if (not mid in shape2vec.keys()) : continue
    desc = dfdesc[dfdesc.mid==mid].desc.to_numpy()[0]
    desc = desc.replace(".", " . ")
    ptvect = np.zeros((cf_max_length)) 
    svect = shape2vec[mid]
    dall.append([mid, desc, ptvect, svect])
dallnp = np.array(dall)

#%% Tokenizer setup    
tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n', oov_token = cf_oov_tok) # all without .
tokenizer.fit_on_texts(augdescs_all)
vocab_size = len(tokenizer.word_index.keys())

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
#%% Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#%%
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
def decode_article(text):
    if (type(text) == list) : return [' '.join([reverse_word_index.get(i, '?') for i in txt]) for txt in text]
    else : return ' '.join([reverse_word_index.get(i, '?') for i in text])

def padEnc(text) :
    if type(text) == str :
        return pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    else :
        return pad_sequences(tokenizer.texts_to_sequences(text), maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)

#%%
seqs = [item for item in dallnp[:,1]]
ptv = tokenizer.texts_to_sequences(seqs)

seqs = [item for item in dallnp[:,1]]
ptv = padEnc(seqs)
for i in range(len(dallnp)) :
    dallnp[i][2] = ptv[i]

#%% Setup variables for model and training
mids = list(dallnp[:,0])
descs = list(dallnp[:,1])
enc_descs = list(dallnp[:,2])
label_vect = list(dallnp[:,3])

#%%
def getAug(mids_tensor) :
    ptv = padEnc(ag.augmentMutli(mids_tensor.numpy()))
    return ptv

cf_descs_per_shape = 4
aug_enc = np.zeros((len(mids), cf_descs_per_shape, cf_max_length))  # len(mids)
for i, mid in tqdm.tqdm(enumerate(mids), total=len(mids)) :
    augs = padEnc(ag.augmentDescMany(mid, cf_descs_per_shape))
    augs = np.stack(augs)
    aug_enc[i,:,:] = augs
 
#%%
# np.save('aug_encs.npy', aug_enc, allow_pickle=True)
# np.save('dallnp.npy', dallnp, allow_pickle=True)

#%%
aug_enc = np.load('aug_encs.npy')
dallnp = np.load('dallnp.npy', allow_pickle=True)

#%%
cf_batch_size = 2
cf_learning_rate = 1e-3
cf_latent_dim = 100
cf_embed_dim = 100
train_ds = tf.data.Dataset.from_tensor_slices((aug_enc, label_vect)).shuffle(10000).batch(cf_batch_size, drop_remainder=True)
for sample_desc, sample_labs in train_ds.take(1) : pass

#%% Make model
modeltxt = tx.Text(vocab_size=len(tokenizer.word_index)+1+4, embed_dim=cf_embed_dim, batch_size=cf_batch_size, latent_dim=cf_latent_dim, learning_rate=cf_learning_rate)
modeltxt.setLR(cf_learning_rate)
modeltxt.printMSums()
modeltxt.printIO()
loss_mean = tf.keras.metrics.Mean()

#%% Setup logger info
run_name = 'txtruns/0207-0831'
root_dir_local = '/home/starstorms/Insight/shape/' + run_name
root_dir_remote = '/data/sn/all/runs/' + run_name
lg = logger.logger(root_dir = root_dir_remote if remote else root_dir_local)
# lg = logger.logger()
lg.setupCP(encoder=None, generator=modeltxt.text_model, opt=modeltxt.optimizer)
lg.restoreCP() #'/home/starstorms/Insight/shape/runs/0204-2220/models/ckpt-71')

#%% Train model manually
def trainModel(num_epochs, display_interval=-1, save_interval=10) :
    print('\nStarting training...')
    modeltxt.training=True
    for epoch in range(1, num_epochs):
        start_time = time.time()
        loss_mean.reset_states()
        for train_x, train_y in train_ds : 
            desc_index = random.randint(0,cf_descs_per_shape-1)
            # st = time.time()
            trn_x = train_x[:,desc_index,:]
            # st2 = time.time()
            loss_mean(modeltxt.trainStep(trn_x, train_y))
            # print('Augmenting took: {:.3f}  Training took: {:.3f}'.format(st2-st, time.time()-st2))  
            
        loss_epoch = loss_mean.result().numpy()
        modeltxt.text_model.reset_states()
         
        if (epoch % display_interval == 0) :
            pass
            
        if (epoch % save_interval == 0) :
            lg.cpSave()
            
        if (ut.checkStopSignal()) :
            print('Stop signal recieved...')
            break
    
        print('Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}'.format(lg.total_epochs, float(loss_epoch), float(time.time() - start_time)))
        lg.logMetric(loss_epoch, 'train loss')
        lg.incrementEpoch()
        
#%% Train the model
lg.writeConfig(locals(), [tx])
trainModel(20000, save_interval=20)

#%% Show an example vector comparison
for train_x, train_y in train_ds.shuffle(1000) : pass
trn_x = train_x[:,0,:]

#%%
def getVect(text) :
    pet = padEnc(text)
    pt = np.repeat(pet, 1024, axis=0)
    pred = modeltxt.text_model(pt).numpy()
    return pred[0]

#%%
modeltxt.training = False
index = 1
pred = modeltxt.text_model.predict(trn_x[0])
print('\nPredicted vector: \n', pred, '\n')
print('Label vector: \n', label_vect[index])
l = modeltxt.compute_loss(label_vect[index], pred[index]).numpy()
print('\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}'.format(index, l, np.sum(pred), np.sum(label_vect[index])))

#%%
txt_vec_tree = spatial.KDTree(label_vect)
n_dists, close_ids = txt_vec_tree.query(label_vect[0], k = 5)













#%%
for i in range(10) :
    text = input('Text description: ')
    if (text=='') : break
    n_dists, close_ids = txt_vec_tree.query(modeltxt.sample(padEnc([text,text])), k = 2)
    print('\n', mids[close_ids[0][0]], '\n', n_dists[0][0], '\n')
    
#%% Get global max an min vals in the label vectors
max_val, min_val = 1, -1
for _, train_y in train_ds: 
    max_val = max(max_val, np.max(train_y))
    min_val = min(min_val, np.min(train_y))
print(min_val, max_val)

#%%   
def getVox(text) :
    enc_text = padEnc(text)
    pred_sv = modeltxt.model(np.array(enc_text))
    vox = modeltxt.sample(pred_sv).numpy()[0,...,0]
    return vox, pred_sv

def descRecon(index, title='') :
    enc_text = enc_descs[index]
    pred = modeltxt.model(enc_text[None,...])
    print('\nPredicted vector: \n', pred, '\n')
    print('Label vector: \n', label_vect[index])
    l = modeltxt.compute_loss(label_vect[index], pred).numpy()[0]
    mid = mids[index]
    print('\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}\n  Mid : {}\n'.format(index, l, np.sum(pred), np.sum(label_vect[index]), mid))
    
    vox_gen = model.sample(pred).numpy()[0,...,0]
    vox_lab = model.sample(label_vect[index][None,...]  ).numpy()[0,...,0]
    
    ut.plotVox(vox_lab, title='Org {}'.format(title), limits=cf_limits)
    ut.plotVox(vox_gen, title='Rec {}'.format(title), limits=cf_limits)

#%%
for i in range(250, 260) : 
    descRecon(i, title=i)
    
#%% Generate shapes from descriptions
desc_test = text = 'lamp that is a table or floor lamp that is made of a lamp unit, a lamp body and a lamp base. the lamp body is made of a lamp pole. the lamp base which is the lamp holistic base is made of a lamp base part. the object is long in length and regular in height. it is regular in width. it is square in shape. '
enc_text_test = padEnc(desc_test)
pred = modeltxt.model(enc_text_test)
vox_gen = model.sample(pred).numpy()[0,...,0]
ut.plotVox(vox_gen, limits=cf_limits)