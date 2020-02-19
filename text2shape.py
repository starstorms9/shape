#%% Imports
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import random as rn
import time
import spacy
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cvae as cv
import utils as ut
import logger
import textspacy as ts

np.set_printoptions(precision=3, suppress=True)
remote = os.path.isdir('/data/sn/all')

#%% Load text data
cf_vox_size = 64
cf_latent_dim = 128
cf_max_length = 50
cf_test_ratio = .15
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'
cf_limits=[cf_vox_size, cf_vox_size, cf_vox_size]
dfmeta = ut.readMeta('/data/sn/all/meta/dfmeta.csv') if remote else ut.readMeta()

rdir_local = '/home/starstorms/Insight/shape/runs/0209-0306'
rdir_remote = '/data/sn/all/runs/0209-0306'
infile = open(os.path.join(rdir_remote if remote else rdir_local,"shape2vec.pkl"),'rb')
shape2vec = pickle.load(infile)
infile.close()

#%% Setup spacy
def get_embeddings(vocab):
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors

nlp = spacy.load('en_core_web_md', entity=False)
embeddings = get_embeddings(nlp.vocab)
elayer = tf.keras.layers.Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], input_length=cf_max_length, trainable=False)
               
#%% Encoding methods
def padEnc(text) :
    texts = text if type(text) == list else [text]
    ranks = [[nlp.vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

def decode(arr) :
    output = [ '{}'.format(vocab[index]) for index in arr if index in vocab.keys()]
    return ' '.join(output).replace('_','')

#%% Get a list of all mids and descs and latent vectors
save_template = '/home/starstorms/Insight/shape/data/{}.npy'
if remote : save_template = '/data/sn/all/data/{}.npy'

alldnp = np.load(save_template.format('alldnp'))
all_mids = list(alldnp[:,0])
all_descs = list(alldnp[:,1])
all_pdescs = list(padEnc(all_descs))

#%% Filter out any data that is not present in shape2vec and stack it into np arrays
not_found_count = 0
mids, descs, vects, padenc = [], [], [], []
for mid, desc, pdesc in tqdm(zip(all_mids, all_descs, all_pdescs), total=len(all_mids)) :
    if (mid in shape2vec.keys()) :
        mids.append(mid)
        descs.append(desc)
        vects.append(shape2vec[mid])
        padenc.append(pdesc)
    else :
        not_found_count += 1
        
mnp, dnp, vnp, pnp = np.stack(mids), np.stack(descs), np.stack(vects), np.stack(padenc)

#%%
# np.save(save_template.format('mnp'), mnp) , np.save(save_template.format('dnp'), dnp) , np.save(save_template.format('vnp'), vnp) , np.save(save_template.format('pnp'), pnp)
# mnp, dnp, vnp, pnp = np.load(save_template.format('mnp')), np.load(save_template.format('dnp')), np.load(save_template.format('vnp')), np.load(save_template.format('pnp'))

#%% Average and remove duplicates
# def avg_dups(labels, values, label_values):
#     folded, indices, counts = np.unique(labels, return_inverse=True, return_counts=True)
#     output = np.zeros((folded.shape[0], values.shape[1]))
#     np.add.at(output, indices, values)
#     output /= counts[:, None]
    
#     label_output = np.zeros((folded.shape[0], label_values.shape[1]))
#     label_output[indices] = label_values
    
#     return label_output, output

# lnp, onp = avg_dups(dnp, vnp, pnp)

#%% Make datasets
num_samples = len(pnp)
val_samples = int(cf_test_ratio * num_samples)
train_samples = num_samples - val_samples
dataset = tf.data.Dataset.from_tensor_slices((pnp, vnp)).shuffle(1000000)
train_ds = dataset.take(train_samples).batch(1024)
val_ds = dataset.skip(train_samples).take(val_samples).batch(1024)

for train_x, train_y in train_ds : pass
for val_x, val_y in val_ds : pass

#%% Make text model
txtmodel = ts.TextSpacy(128, learning_rate=6e-4, max_length=cf_max_length, training=True) #, embeddings=embeddings)
txtmodel.printMSums()
txtmodel.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
loss_mean = tf.keras.metrics.Mean()

#%% Setup logger info
run_name = 'txtruns/0217-0437'
root_dir_local = '/home/starstorms/Insight/shape/' + run_name
root_dir_remote = '/data/sn/all/' + run_name
lg = logger.logger(root_dir = root_dir_remote if remote else root_dir_local)
# lg = logger.logger(trainMode=remote, txtMode=True)
lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restoreCP()

#%% Train model manually
def trainModel(num_epochs, display_interval=-1, save_interval=10, validate_interval=5) :
    print('\nStarting training...')
    txtmodel.training=True
    for epoch in range(1, num_epochs):
        start_time = time.time()
        loss_mean.reset_states()
        for train_x, train_y in train_ds : 
            loss_mean(txtmodel.trainStep(train_x, train_y))
        loss_epoch = loss_mean.result().numpy()
        
        if (epoch % validate_interval == 0) :
            loss_mean.reset_states()
            for validation_x, validation_y in val_ds :
                pred_y = txtmodel.model(validation_x)
                loss_mean(txtmodel.compute_loss(pred_y, validation_y))
            val_loss_epoch = loss_mean.result().numpy()
            lg.logMetric(val_loss_epoch, 'val loss')
            print('TEST LOSS: {:.3f}'.format(val_loss_epoch))
            
        if (epoch % save_interval == 0) :
            lg.cpSave()
            # txtmodel.saveMyModel(lg.model_saves, lg.total_epochs)
            
        if (ut.checkStopSignal()) :
            print('Stop signal recieved...')
            break
    
        print('Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}'.format(lg.total_epochs, float(loss_epoch), float(time.time() - start_time)))
        lg.logMetric(loss_epoch, 'train loss')
        lg.incrementEpoch()
        
#%% Train the model
lg.writeConfig(locals(), [ts])
trainModel(20000, save_interval=20, validate_interval=5)

#%%
index = 1
for tx, tl in train_ds.take(1) : pass
txtmodel.model.training = False
pred = txtmodel.sample(tx)
print('\nPredicted vector: \n', pred[index], '\n')
print('Label vector: \n', tl[index])
l = txtmodel.compute_loss(tl[index], pred[index]).numpy()
signs_eq = sum(np.sign(pred[index]) == np.sign(tl[index])) / pred[index].shape[0]
print('\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}  Same Sign: {:.1f}%'.format(index, l, np.sum(pred[index]), np.sum(tl[index]), 100*signs_eq))

#%%
txtmodel.sample(pnp[0:1])
vnp[0:1]

#%% Show nearest neighbor descriptions
index = 1
n_dists, close_ids = txt_vec_tree.query(label_vects[index], k = 3)
nearest_descs = [ dfmeta[dfmeta.mid == mids[cid]].desc.values[0] for cid in close_ids]
for d in nearest_descs : print(d, '\n')


#%%
shapemodel = cv.CVAE(cf_latent_dim, cf_vox_size) #, training=False)
shapemodel.printMSums()
shapemodel.printIO()

run_name = '0209-0306'
root_dir_local = '/home/starstorms/Insight/shape/runs/' + run_name
root_dir_remote = '/data/sn/all/runs/' + run_name
lg = logger.logger(root_dir = root_dir_local if not remote else root_dir_remote)
lg.setupCP(encoder=shapemodel.enc_model, generator=shapemodel.gen_model, opt=shapemodel.optimizer)
lg.restoreCP()

def getVox(text) :
    ptv = padEnc(text)
    preds = txtmodel.sample(ptv)
    vox = shapemodel.sample(preds).numpy()[0,...,0]
    return vox

#%% Test text2shape model
keyword = 'bowl'
# ex_descs = []
for i in range(10) :
    desc = dnp[np.random.randint(0,len(dnp))]
    while not keyword in desc :
        desc = dnp[np.random.randint(0,len(dnp))]
    ex_descs.append(desc)
    print(desc)
    
for i in range(20) :
    text = input('Text description: ')
    vox = getVox(text)
    ut.plotVox(vox, limits=cf_limits)

#%% See averaged out reconstructions
for i in range(100):
    index = rn.randint(0, len(onp))
    rand_vect = onp[index][None,...]
    desc = decode(lnp[index])
    vox = shapemodel.sample(rand_vect).numpy()[0,...,0]
    ut.plotVox(vox, limits=cf_limits, title='')
    # print(desc)
    # temp = input('Enter to continue...')


#%% Run on single line of text
text = ' ceiling lamp that is very skinny and very tall.  it has one head .  it has a base.  it has one chain .'
tensor = tf.constant(text)
tbatch = tensor[None,...]
preds = txtmodel.model(tbatch)