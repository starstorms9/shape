'''
This is the streamlit app for my Insight AI 2020A.SV project.
All file references are relative to the this file in the github project so that it works with streamlit for teams (SL4T).

The app can be launched locally by using a terminal to navigate to the directory of this .py file and running this command:
    streamlit run streamlit_app.py
'''

import sys
import streamlit as st
header = st.title('')
header.header("Importing libraries...")

import numpy as np
import pandas as pd

server_up = True
try :
    import plotly
    import re
    import time
    import pickle
    from scipy import spatial
    import skimage.measure as sm
    
    import matplotlib.pyplot as pltmodel
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    import plotly.express as px
    import plotly.figure_factory as FF
    import os
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import spacy
    import random as rn
    
    import cvae as cv
    import textspacy as ts
except :
    header.header("Server is currently overloaded, please try again later!")
    server_up = False

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
    return pd.read_csv(os.path.join(os.getcwd(),'data/df_sl_2d.csv'))

@st.cache
def getTSNE2DData() :
    df_tsne = loadTSNEData2D()
    df_tsne = df_tsne[df_tsne.columns.drop(list(df_tsne.filter(regex='Unnamed:')))]
    named_cols = ['Model ID', 'tsne1', 'tsne2', 'Category ID', 'Category', 'Anno ID',
                  'Sub Categories', 'Description', 'Details', 'Length', 'Height', 'Width', 'Squareness',
                  'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Loss', 'Log Loss']
    df_tsne.columns = named_cols
    return df_tsne    

@st.cache
def loadExampleDescriptions() :
    example_descriptions = np.load(os.path.join(os.getcwd(), 'data/exdnp.npy'))
    return list(example_descriptions)

@st.cache
def loadShape2Vec() :
    infile = open(os.path.join(os.getcwd(), 'data/shape2vec.pkl'),'rb')
    shape2vec = pickle.load(infile)
    infile.close()
    mids = list(shape2vec.keys())
    vecs = np.array([shape2vec[m] for m in mids])
    vec_tree = spatial.KDTree(vecs)
    return shape2vec, mids, vecs, vec_tree

@st.cache
def loadMid2Desc() :
    infile = open(os.path.join(os.getcwd(), 'data/mid2desc.pkl'),'rb')
    mid2desc = pickle.load(infile)
    infile.close()
    return mid2desc

@st.cache(allow_output_mutation=True)
def makeShapeModel() :
    model_in_dir = os.path.join(os.getcwd(), 'models/autoencoder')
    shapemodel = cv.CVAE(128, 64, training=False)
    shapemodel.loadMyModel(model_in_dir, 195)
    return shapemodel

@st.cache(allow_output_mutation=True)
def makeTextModel() :
    model_in_dir = os.path.join(os.getcwd(),'models/textencoder')
    textmodel = ts.TextSpacy(cf_latent_dim, max_length=cf_max_length, training=False)
    textmodel.loadMyModel(model_in_dir, 10569)
    return textmodel

@st.cache(allow_output_mutation=True)
def getSpacy() :
    nlp = spacy.load("en_core_web_md", parser=False, tagger=False, entity=False)
    return nlp.vocab

def interp(vec1, vec2, divs=5, include_ends=True) :
    out = []
    amounts = np.array(range(divs+1))/divs if include_ends else np.array(range(1, divs))/divs
    for amt in amounts :
        interpolated_vect = vec1 * amt + vec2 * (1-amt)
        out.append( interpolated_vect )
    return np.stack(out)

def padEnc(text, vocab) :
    texts = text if type(text) == list else [text]
    ranks = [[vocab[t].rank for t in sent.replace('.', ' . ').split(' ') if len(t)>0] for sent in texts]
    padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded

def getVox(text, shapemodel, textmodel, nlp) :
    ptv = padEnc(text, nlp)
    preds = textmodel.sample(ptv)
    vox = shapemodel.sample(preds).numpy()[0,...,0]
    return vox, preds

def conditionTextInput(text) :
    replacements = {
    ',' : '',
    '  ' : ' ',
    'its' : 'it is',
    'an' : 'a',
    'doesnt' : 'does not',
    '1' : 'one',
    '2' : 'two',
    '3' : 'three',
    '4' : 'four',
    '5' : 'five',
    '6' : 'six',
    '7' : 'seven',
    '8' : 'eight',
    '9' : 'nine',
    }
    
    desc = text.lower().strip()
    for fix in replacements.keys() :
        desc = desc.replace(fix, replacements[fix])
    return desc

def addThumbnailSelections(df_tsne) :   
    annoid_input = st.sidebar.text_input('Anno IDs to view (comma separated, from plot):')
    sidebar_renders = st.sidebar.checkbox('Show renders in sidebar?')
    if len(annoid_input) > 1 :
        annosinspect = [annoid.strip().replace("'", "").replace("[", "").replace("]", "") for annoid in re.split(',',annoid_input) if len(annoid) > 1]
                
        pic_in_dir = 'https://starstorms-shape.s3-us-west-2.amazonaws.com/renders/'
        mid2desc = loadMid2Desc()
                
        for i, aid in enumerate(annosinspect) :
            try :
                mid = df_tsne[df_tsne['Anno ID'] == int(aid)]['Model ID'].values[0]
                fullpath = os.path.join(pic_in_dir, mid+'.png')
                img = mpimg.imread(fullpath)
                
                if (sidebar_renders) :
                    st.sidebar.text('Annod ID: {}'.format(annosinspect[i]))
                    desc_empty = st.sidebar.empty()
                    st.sidebar.image(img, use_column_width = True)
                    desc_empty.text(mid2desc[mid])
                else :
                    st.text('Annod ID: {}'.format(annosinspect[i]))
                    desc_empty = st.empty()
                    st.image(img, use_column_width = False)
                    desc_empty.text(mid2desc[mid])                
            except :
                if (sidebar_renders) :
                    st.sidebar.text('Could not find {}'.format(annosinspect[i]))
                else :
                    st.text('Could not find {}'.format(annosinspect[i]))

def addMIDLines(df_tsne, fig) :
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
    return

def getStartVects() :
    sindices = {
        'Table'  : [7764, 6216, 3076, 2930, 715, 3165],
        'Chair'  : [9479, 13872, 12775, 9203, 9682, 9062, 8801, 8134, 12722, 7906, 10496, 11358, 13475, 9348, 13785, 11697],
        'Lamp'   : [15111, 15007, 14634, 14646, 15314, 14485],
        'Faucet' : [15540, 15684, 15535, 15738, 15412],
        'Clock'  : [16124, 16034, 16153],
        'Bottle' : [16690, 16736, 16689],
        'Vase'   : [17463, 17484, 17324, 17224, 17453],
        'Laptop' : [17780, 17707, 17722],
        'Bed'    : [18217, 18161],
        'Mug'    : [18309, 18368, 18448],
        'Bowl'   : [18501, 17287, 18545, 18479, 18498]}
    return sindices

def createMesh(vox, step=1, threshold = 0.5) :    
    vox = np.pad(vox, step)
    verts, faces, normals, values = sm.marching_cubes_lewiner(vox, 0.5, step_size=step)
    return verts, faces

def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1), plot_it=True, title='') :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title=title, aspectratio=aspect)
    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=1, range=[0,cf_vox_size+1],backgroundcolor='white', gridcolor='white'),
        yaxis = dict(nticks=1, range=[0,cf_vox_size+1],backgroundcolor='white', gridcolor='white'),
        zaxis = dict(nticks=1, range=[0,cf_vox_size+1],backgroundcolor='white', gridcolor='white'),),
    width=900, height=700,
    margin=dict(r=20, l=10, b=10, t=10))
    return fig

def plotVox(voxin, step=1, title='', tsnedata=None) :       
    try :
        verts, faces = createMesh(voxin, step)
    except :
        st.write('Failed creating mesh for voxels.')
        return
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.set_xlim(0, cf_vox_size)
    ax.set_ylim(0, cf_vox_size)
    ax.set_zlim(0, cf_vox_size)
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
    
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

#%% Setup main methods
def text2Shape() :
    setWideModeHack()
    header.title('Text 2 shape')
    loading_text = st.empty()
    loading_text.text('Getting Spacy Embeddings...')
    vocab = getSpacy()
    loading_text.text('Making text encoder model...')
    shapemodel = makeShapeModel()
    loading_text.text('Making shape generator model...')
    textmodel = makeTextModel()
    loading_text.text('Models done being made!')
    
    description = st.text_input('Enter Shape Description:', value='a regular chair with four legs.')
    description = conditionTextInput(description)
    vox, encoding = getVox(description, shapemodel, textmodel, vocab)
    
    verts, faces = createMesh(vox, step=1)
    fig = showMesh(verts, faces)
    st.write(fig)
    
    st.header('Similar descriptions:')
    mid2desc = loadMid2Desc()
    shape2vec, mids, vecs, vec_tree = loadShape2Vec()
    _, close_ids = vec_tree.query(encoding, k = 5)
    close_ids = list(close_ids[0])
    for i, index in enumerate(close_ids) :
        try :
            mid = mids[int(index)]
            st.write('{}. {}'.format(i+1, mid2desc[mid]))
        except :
            continue

def vectExplore() :
    setWideModeHack()
    header.title('Vector exploration')
    
    df_tsne = getTSNE2DData()    
    bright=["#023EFF","#FF7C00","#1AC938","#E8000B","#8B2BE2","#9F4800","#F14CC1","#A3A3A3","#000099","#00D7FF","#222A2A"]    
    color_option = st.sidebar.selectbox("Color Data", ['Category', 'Length', 'Height', 'Width', 'Squareness', 'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Log Loss'])
    size = st.sidebar.number_input('Plot Dot Size', value=6.0, min_value=0.1, max_value=30.0, step=1.0)
    
    padding = 1.05
    xmin, xmax = df_tsne.tsne1.min(), df_tsne.tsne1.max()
    ymin, ymax = df_tsne.tsne2.min(), df_tsne.tsne2.max()
    xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]
    
    fig = px.scatter(data_frame=df_tsne.dropna(), range_x = xlims, range_y = ylims,
                      hover_name='Category',
                      hover_data=['Sub Categories', 'Anno ID'],
                      size='Width',
                      x='tsne1', y='tsne2', color=color_option, color_discrete_sequence=bright, width=1400, height=900)
    fig.update_traces(marker= dict(size=size, opacity=0.7, line=dict(width=0.1))) # size=2.3
    
    # addMIDLines(df_tsne, fig)
    st.write(fig)    
    addThumbnailSelections(df_tsne)

def shapetime() :
    setWideModeHack()
    header.title('Shape Interpolations')
    subheader = st.subheader('Loading data...')
    cat_options = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']        
    starting_cat = st.sidebar.selectbox('Choose starting shape:', cat_options)
    start_indices = getStartVects()
    start_index = rn.choice(start_indices[starting_cat])    
    df_tsne = getTSNE2DData()
    
    vects_sample = st.sidebar.number_input('Variety of samples (higher --> diverse)', value=200, min_value=10, max_value=1000, step=5)
    max_dist = 8
    interp_points = 6
    plot_step = 2
    
    shapemodel = makeShapeModel()
    shape2vec, mids, vecs, vec_tree = loadShape2Vec()
    empty = st.empty()
    
    start_vect = shape2vec[mids[start_index]]
    visited_indices = [start_index]
    for i in range(3) :
        journey_vecs = []
        journey_mids = []
        journey_mids.append(mids[start_index])
        subheader.subheader('Generating models... please wait...')
        
        for i in range(5) :
            n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist)
            if len(shape2vec) in close_ids :
                n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist*3)    
            close_ids = list(close_ids)
            
            visited_indices = visited_indices[:50]            
            for index in sorted(close_ids, reverse=True):
                if index in visited_indices:
                    close_ids.remove(index)
            
            next_index = rn.choice(close_ids)
            next_vect = vecs[next_index]
            visited_indices.append(next_index)
            interp_vects = interp(next_vect, start_vect, divs = interp_points)
            journey_vecs.extend(interp_vects)
            start_vect = next_vect
            journey_mids.append(mids[next_index])
            
        journey_voxs = np.zeros(( len(journey_vecs), cf_vox_size, cf_vox_size, cf_vox_size))
        for i, vect in enumerate(journey_vecs) :
            journey_voxs[i,...] = shapemodel.decode(vect[None,...], apply_sigmoid=True)[0,...,0]
        
        subheader.subheader('Showing models... (may have to scroll down)')
        for i, vox in enumerate(journey_voxs) :
            data = plotVox(vox, step=plot_step, tsnedata=df_tsne)
            empty.image(data)
    subheader.subheader('All done!')
        
def manual() :
    example_descriptions = loadExampleDescriptions()
    header.title('Streamlit App Manual')
    st.write(
            """
            This is my streamlit app for my Insight AI.SV.2020A project.
            
            See slides related to the development of this app [here](https://drive.google.com/open?id=1u9Iq2nyES0Rx55F-Nn02x4phViBae2Ps4181UONW_DI) 
            and the github repo with code [here](https://github.com/starstorms9/shape).
            
            **Below is an example of what can be generated. Input was 'a regular chair with four legs.'**
            """ )
     
    chair_verts = pickle.load( open( os.path.join(os.getcwd(), 'media/chair_verts.p'), "rb" ))
    chair_faces = pickle.load( open( os.path.join(os.getcwd(), 'media/chair_faces.p'), "rb" ))
    fig = showMesh(chair_verts, chair_faces)
    st.write(fig)
    
    st.write(
            """
            ## Available Tabs:            
            - ### Text to shape generator
            - ### Latent vector exploration
            - ### Shape interpolation
            
            ## Text to Shape Generator
            This tab allows you to input a description and the generator will make a model based on that description.
            The 3D plotly viewer generally works much faster in Firefox compared to chrome so use that if chrome is being slow.
            
            The bottom of this tab shows similar descriptions to the input description. Use these samples to see new designs and 
            learn how the model interprets the text.
            
            #### Models were trained on these object classes _(number of train examples)_:
            - Table    (8436)
            - Chair    (6778)
            - Lamp     (2318)
            - Faucet   (744)
            - Clock    (651)
            - Bottle   (498)
            - Vase     (485)
            - Laptop   (460)
            - Bed      (233)
            - Mug      (214)
            - Bowl     (186)
            """)
            
    if st.button('-->Click here to get some random example descriptions<--') :
        descs = rn.sample(example_descriptions, 5)
        for d in descs : st.write(d)
        
    st.write(
            """    
            ## Latent Vector Exploration
            This tab shows the plot of the shape embedding vectors reduced from the full model dimensionality of 128 dimensions
            down to 2 so they can be viewed easily. The method for dimensionality reduction was TSNE.
            
            #### In the exploration tab, there are several sidebar options:
            - **Color data**
                - This selector box sets what determines the color of the dots. (the class selections are particularly interesting!)
            - Plot dot size
                - This sets the dot size. Helpful when zooming in on a region.
            - Model IDs             
                - This allows for putting in multiple model IDs to see how they're connected on the graph.
            - **Anno IDs to view**
                - From the hover text on the TSNE plot points you can see the 'Anno ID' (annotation ID) and enter it into this box to see a render of the object and 1 of its generated descriptions.
                - Multiple IDs can be entered and separated by commas.
                - The renders can be viewed in the sidebar or in the main area below the TSNE graph.
            
            Additionally, using the plotly interface you can **double click** on a category in the legend to show only that
            category of dots. Or **click once** to toggle showing that category. You can also zoom in on specific regions to
            see local clustering in which case it may be useful to increase the plot dot size.
            
            The shape embeddings are very well clustered according to differt shape classes but also to sub categories
            inside those classes. By playing with the color data, it can be seen that the clusters are also organized very strongly
            by specific attributes about the object such as is it's overall width, length, or height.

            ### TSNE map showing different colors for the different shape classes:            
                """)
          
    tsne_pic = os.path.join(os.getcwd(), 'media/tsne_small.png')
    img = mpimg.imread(tsne_pic)    
    st.image(img, use_column_width=True)
           
    st.write(
            """    
            ## Shape Interpolation
            This tab is just for fun and is intended to show how well the model can interpolate between various 
            object models. Note that this runs the model many times and as such can be quite slow online. You may need to hit 'stop' 
            and then 'rerun' from th menu in the upper right corner to make it behave properly.
            
            To generate these plots, the algorithm finds the nearest K shape embedding vectors
            (K set by the variety parameter in the sidebar) and randomly picks one of them.
            Then it interpolates between the current vector and the random new vector
            and at every interpolated point it generates a new model from the interpolated latent space vector.
            Then it repeats indefinitely finding new vectors as it goes.
            
            #### In this tab there are 2 sidebar options:
            - Starting shape
                - This sets the starting category for the algorithm but it will likely wander off into other categories
                after a bit
            - Variety parameter
                - This determines the diversity of the models by setting how many local vectors to choose from.
        
            ### Example pre-rendered gif below:
                """)
            
    cat_options = ['Couches','Chairs','Faucets','Beds','Bottles','Bowls','Lamps','Mugs','RoundTables','SwivelChairs','Tables']
    gif_urls = ['https://github.com/starstorms9/shape/blob/master/media/{}.gif?raw=true'.format(cat.lower()) for cat in cat_options]
    selected_cat = st.selectbox('Select a category to see shape interpolations', cat_options, index=0)
    gif_url = gif_urls[cat_options.index(selected_cat)]
    st.image(gif_url, use_column_width=True)
    
    
#%% Main selector system
if server_up :
    modeOptions = ['Manual', 'Text to Shape', 'Latent Vect Exploration', 'Shape Interpolation']
    st.sidebar.header('Select Mode:')
    mode = st.sidebar.radio("", modeOptions, index=0)
    
    tabMethods = [manual, text2Shape, vectExplore, shapetime]
    tabMethods[modeOptions.index(mode)]()