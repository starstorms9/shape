import numpy as np
import pandas as pd
import streamlit as st
import plotly
import re
import time
import skimage.measure as sm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as FF
import altair as alt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

import cvae as cv
import utils as ut
import logger
import textspacy as ts

#%% Global setup variables
cf_vox_size = 64
cf_latent_dim = 128
cf_max_length = 50
cf_test_ratio = .15
cf_trunc_type = 'post'
cf_padding_type = 'post'
cf_oov_tok = '<OOV>'
cf_limits=[cf_vox_size, cf_vox_size, cf_vox_size]

#%% Setup sub methods
def setWideModeHack():
    max_width_str = f"max-width: 2000px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True)

@st.cache
def loadTSNEData2D(axes='2d') :
    return pd.read_csv('data/df_sl_2d.csv')

@st.cache
def getTSNE2DData() :
    df_tsne = loadTSNEData2D()
    df_tsne = df_tsne[df_tsne.columns.drop(list(df_tsne.filter(regex='Unnamed:')))]
    named_cols = ['Model ID', 'tsne1', 'tsne2', 'Category ID', 'Category', 'Anno ID',
                  'Sub Categories', 'Description', 'Details', 'Length', 'Height', 'Width', 'Squareness',
                  'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Loss', 'Log Loss']
    df_tsne.columns = named_cols
    return df_tsne    

@st.cache(allow_output_mutation=True)
def makeShapeModel() :
    model_in_dir = '/home/starstorms/Insight/shape/shape.git/models/autoencoder'
    shapemodel = cv.CVAE(128, 64, training=False)
    shapemodel.loadMyModel(model_in_dir, 195)
    return shapemodel

@st.cache(allow_output_mutation=True)
def makeTextModel() :
    model_in_dir = '/home/starstorms/Insight/shape/shape.git/models/textencoder'
    textmodel = ts.TextSpacy(cf_latent_dim, max_length=cf_max_length, training=False)
    textmodel.loadMyModel(model_in_dir, 5669)
    return textmodel

@st.cache(allow_output_mutation=True)
def getSpacy() :
    nlp = spacy.load('en_core_web_md', entity=False)
    return nlp.vocab

def padEnc(text, vocab) :
    texts = text if type(text) == list else [text]
    ranks = [[vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

def getVox(text, shapemodel, textmodel, nlp) :
    ptv = padEnc(text, nlp)
    preds = textmodel.sample(ptv)
    vox = shapemodel.sample(preds).numpy()[0,...,0]
    return vox

def conditionTextInput(text) :
    return text

def addThumbnailSelections(df_tsne) :
    pic_in_dir='/home/starstorms/Insight/ShapeNet/renders'
    if not os.path.isdir(pic_in_dir) : return
    
    annoid_input = st.sidebar.text_input('Anno ID to view')
    if len(annoid_input) > 1 :
        annosinspect = [annoid.strip().replace("'", "").replace("[", "").replace("]", "") for annoid in re.split(',',annoid_input) if len(annoid) > 1]
        modelid = df_tsne[df_tsne['Anno ID'] == int(annosinspect[0])]['Model ID'].values[0]
        
        pic_in_dir='/home/starstorms/Insight/ShapeNet/renders'
        fullpath = os.path.join(pic_in_dir, modelid+'.png')
        img = mpimg.imread(fullpath)
        st.sidebar.image(img)

def createMesh(vox, step=1, threshold = 0.5) :    
    vox = np.pad(vox, step)
    verts, faces, normals, values = sm.marching_cubes_lewiner(vox, 0.5, step_size=step)
    return verts, faces

def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1), plot_it=True, title='') :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title=title, aspectratio=aspect)
    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[0,cf_vox_size+1],),
                     yaxis = dict(nticks=4, range=[0,cf_vox_size+1],),
                     zaxis = dict(nticks=4, range=[0,cf_vox_size+1],),),
    width=900, height=700,
    margin=dict(r=20, l=10, b=10, t=10))
    return fig

#%% Setup main methods
def text2Shape() :
    st.write('Text 2 shape')
    loading_text = st.empty()
    loading_text.text('Getting Spacy Embeddings...')
    vocab = getSpacy()
    loading_text.text('Making text encoder model...')
    shapemodel = makeShapeModel()
    loading_text.text('Making shape generator model...')
    textmodel = makeTextModel()
    loading_text.text('Models done being made!')
    
    description = st.text_input('Shape Description:', value='a regular chair. it has four legs.')
    vox = getVox(description, shapemodel, textmodel, vocab)
    
    verts, faces = createMesh(vox, step=1)
    fig = showMesh(verts, faces)
    st.write(fig)

def vectExplore() :
    st.write('Vector exploration')
    
    df_tsne = getTSNE2DData()    
    bright=["#023EFF","#FF7C00","#1AC938","#E8000B","#8B2BE2","#9F4800","#F14CC1","#A3A3A3","#000099","#00D7FF","#222A2A"]    
    color_option = st.sidebar.selectbox("Color Data", ['Category', 'Length', 'Height', 'Width', 'Squareness', 'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Log Loss'])
    size = st.sidebar.number_input('Plot Dot Size', value=6.0, min_value=0.1, max_value=30.0, step=1.0)
    
    padding = 1.05
    xmin, xmax = df_tsne.tsne1.min(), df_tsne.tsne1.max()
    ymin, ymax = df_tsne.tsne2.min(), df_tsne.tsne2.max()
    xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]
    
    config={'scrollZoom': True, 'modeBarButtonsToRemove' : ['lasso2d','zoom2d']}
    fig = px.scatter(data_frame=df_tsne.dropna(), range_x = xlims, range_y = ylims,
                      hover_name='Category',
                      hover_data=['Sub Categories', 'Anno ID'],
                      size='Width',
                      x='tsne1', y='tsne2', color=color_option, color_discrete_sequence=bright, width=1400, height=900)
    fig.update_traces(marker= dict(size=size, opacity=0.7, line=dict(width=0.1))) # size=2.3
    
    midslist = list(df_tsne['Model ID'])
    mids_input = st.sidebar.text_area('Model IDs (comma separated)')
    midsinspect = [mid.strip().replace("'", "").replace("[", "").replace("]", "") for mid in re.split(',',mids_input) if len(mid) > 20]
    some_found = False
    for mid in midsinspect : 
        found = (mid in midslist)
        some_found = some_found or found
        if not found : st.sidebar.text('{} \n Not Found'.format(mid))
    
    if (some_found) :
        midpoints = [[df_tsne.tsne1[midslist.index(mid)], df_tsne.tsne2[midslist.index(mid)]] for mid in midsinspect if (mid in midslist)]
        dfline = pd.DataFrame( midpoints, columns=['x','y'])
        fig.add_scatter(name='Between Models', text=dfline.index.values.tolist(), mode='lines+markers', x=dfline.x, y=dfline.y, line=dict(width=5), marker=dict(size=10, opacity=1.0, line=dict(width=5)) )
    
    st.write(fig)    
    addThumbnailSelections(df_tsne)

def shapetime() :
    st.write('Shapetime journeys')
    
def manual() :
    st.write('This is the manual')

#%% Main selector system
setWideModeHack()
modeOptions = ['Text to Shape', 'Latent Vect Exploration', 'Shapetime Journey', 'Manual']
mode = st.sidebar.selectbox("Select Mode:", modeOptions)

tabMethods = [text2Shape, vectExplore, shapetime, manual]
tabMethods[modeOptions.index(mode)]()