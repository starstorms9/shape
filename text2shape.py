'''
This file is intended for three purposes:
    1. Training the text model
    2. Loading in and quickly testing the overall text2shape model
    3. Generating sample description datasets for use elsewhere
'''

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
import seaborn as sns

import cvae as cv
import utils as ut
import logger
import textspacy as ts
import configs as cf

np.set_printoptions(precision=3, suppress=True)

#%% Load text data
cf_vox_size = 64
cf_latent_dim = 128
cf_max_length = 50
cf_test_ratio = .15
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'
cf_limits=[cf_vox_size, cf_vox_size, cf_vox_size]
dfmeta = ut.readMeta()

shp_run_id = '0209-0306'
infile = open(os.path.join(cf.SHAPE_RUN_DIR, shp_run_id, "shape2vec.pkl"),'rb')
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

#%% Get a list of all mids and descs and latent vectors
save_template = cf.DATA_DIR + '/{}.npy'
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

#%% Save / load the generated arrays to avoid having to regenerate them
# np.save(save_template.format('mnp'), mnp) , np.save(save_template.format('dnp'), dnp) , np.save(save_template.format('vnp'), vnp) , np.save(save_template.format('pnp'), pnp)
mnp, dnp, vnp, pnp = np.load(save_template.format('mnp')), np.load(save_template.format('dnp')), np.load(save_template.format('vnp')), np.load(save_template.format('pnp'))

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
txtmodel = ts.TextSpacy(128, learning_rate=6e-4, max_length=cf_max_length, training=True)
txtmodel.printMSums()
txtmodel.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
loss_mean = tf.keras.metrics.Mean()

#%% Setup logger info
train_from_scratch = False
if train_from_scratch :
    lg = logger.logger(trainMode=cf.REMOTE, txtMode=True)
else :    
    txt_run_id = '0217-0434'
    root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
    lg = logger.logger(root_dir)

lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restoreCP()

#%% Method for training the model manually
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
            
        if (ut.checkStopSignal()) :
            print('Stop signal recieved...')
            break
    
        print('Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}'.format(lg.total_epochs, float(loss_epoch), float(time.time() - start_time)))
        lg.logMetric(loss_epoch, 'train loss')
        lg.incrementEpoch()
        
#%% Train the model
lg.writeConfig(locals(), [ts])
trainModel(20000, save_interval=20, validate_interval=5)

#%% Compare predicted and labeled vectors, useful sanity check on trained model
index = 1
for tx, tl in train_ds.take(1) : pass
txtmodel.model.training = False
pred = txtmodel.sample(tx)
print('\nPredicted vector: \n', pred[index], '\n')
print('Label vector: \n', tl[index])
l = txtmodel.compute_loss(tl[index], pred[index]).numpy()
signs_eq = sum(np.sign(pred[index]) == np.sign(tl[index])) / pred[index].shape[0]
print('\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}  Same Sign: {:.1f}%'.format(index, l, np.sum(pred[index]), np.sum(tl[index]), 100*signs_eq))

#%% Get test set loss
for tx, tl in train_ds.shuffle(100000).take(1000) : pass
txtmodel.model.training = False
pred = txtmodel.sample(tx)
losses = np.mean(txtmodel.compute_loss(pred, tl))
print(losses)

#%% Load shape model
shapemodel = cv.CVAE(cf_latent_dim, cf_vox_size)
shapemodel.printMSums()
shapemodel.printIO()

shp_run_id = '0209-0306'
root_dir = os.path.join(cf.SHAPE_RUN_DIR, shp_run_id)
lg = logger.logger(root_dir)
lg.setupCP(encoder=shapemodel.enc_model, generator=shapemodel.gen_model, opt=shapemodel.optimizer)
lg.restoreCP()

# Method for going from text to voxels
def getVox(text) :
    ptv = padEnc(text)
    preds = txtmodel.sample(ptv)
    vox = shapemodel.sample(preds).numpy()[0,...,0]
    return vox

#%% Test text2shape model
for i in range(20) :
    text = input('Text description: ')
    vox = getVox(text)
    ut.plotVox(vox, limits=cf_limits)

#%% Run on single line of text
text = 'ceiling lamp that is very skinny and very tall. it has one head. it has a base. it has one chain.'
tensor = tf.constant(text)
tbatch = tensor[None,...]
preds = txtmodel.model(tbatch)

#%% Generate a balanced set of sample descriptions to show on streamlit app
ex_descs = []
for keyword in ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl'] :
    for i in range(50) :
        desc = dnp[np.random.randint(0,len(dnp))]
        while not keyword.lower() in desc :
            desc = dnp[np.random.randint(0,len(dnp))]
        ex_descs.append(desc)
        print(desc)
np.save(os.path.join(cf.DATA_DIR, 'exdnp.npy'), np.array(ex_descs))

#%% Generate a large set of sample descriptions to inform nearby descriptions on app
mid2desc = {}
for mid in tqdm(shape2vec.keys()) :
    indices = np.where(mnp==mid)[0]
    if len(indices) > 0 :
        desc = dnp[rn.sample(list(indices), 1)][0]
        mid2desc[mid] = desc

file = open(os.path.join(cf.DATA_DIR, 'mid2desc.pkl'), 'wb')
pickle.dump(mid2desc, file)
file.close()