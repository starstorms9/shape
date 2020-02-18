import numpy as np
import pandas as pd
import streamlit as st
import plotly
import re
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os

def setWideModeHack():
    max_width_str = f"max-width: 2000px;"
    st.markdown( f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """, unsafe_allow_html=True)
setWideModeHack()

# @st.cache
def loadData2D(axes='2d') :
    return pd.read_csv('df_sl_2d.csv')
# @st.cache
def loadData3D(axes='2d') :
    return pd.read_csv('df_sl_3d.csv')

# df_subset = pd.read_csv('/home/starstorms/Insight/shape/streamlit/df_sl_2d.csv')
df_subset = loadData2D()
df_subset = df_subset[df_subset.columns.drop(list(df_subset.filter(regex='Unnamed:')))]
named_cols = ['Model ID', 'tsne1', 'tsne2', 'Category ID', 'Category', 'Anno ID',
              'Sub Categories', 'Description', 'Details', 'Length', 'Height', 'Width', 'Squareness',
              'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Loss', 'Log Loss']
df_subset.columns = named_cols

bright=["#023EFF","#FF7C00","#1AC938","#E8000B","#8B2BE2","#9F4800","#F14CC1","#A3A3A3","#000099","#00D7FF","#222A2A"]
cats = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']

color_option = st.sidebar.selectbox("Color Data", ['Category', 'Length', 'Height', 'Width', 'Squareness', 'Class Length', 'Class Height', 'Class Width', 'Class Square', 'Log Loss'])
color_palette = bright #plotly.express.colors.qualitative.G10
size = st.sidebar.number_input('Size', value=6.0, min_value=0.1, max_value=30.0, step=1.0)

padding = 1.05
xmin, xmax = df_subset.tsne1.min(), df_subset.tsne1.max()
ymin, ymax = df_subset.tsne2.min(), df_subset.tsne2.max()
xlims, ylims = [xmin*padding, xmax*padding], [ymin*padding, ymax*padding]

config={'scrollZoom': True, 'modeBarButtonsToRemove' : ['lasso2d','zoom2d']}
fig = px.scatter(data_frame=df_subset.dropna(), range_x = xlims, range_y = ylims,
                  hover_name='Category',
                  hover_data=['Sub Categories', 'Anno ID'],
                  size='Width',
                  x='tsne1', y='tsne2', color=color_option, color_discrete_sequence=color_palette, width=1400, height=900)
fig.update_traces(marker= dict(size=size, opacity=0.7, line=dict(width=0.1))) # size=2.3

midslist = list(df_subset['Model ID'])
mids_input = st.sidebar.text_area('Model IDs (comma separated)')
midsinspect = [mid.strip().replace("'", "").replace("[", "").replace("]", "") for mid in re.split(',',mids_input) if len(mid) > 20]
some_found = False
for mid in midsinspect : 
    found = (mid in midslist)
    some_found = some_found or found
    if not found : st.sidebar.text('{} \n Not Found'.format(mid))

if (some_found) :
    midpoints = [[df_subset.tsne1[midslist.index(mid)], df_subset.tsne2[midslist.index(mid)]] for mid in midsinspect if (mid in midslist)]
    dfline = pd.DataFrame( midpoints, columns=['x','y'])
    fig.add_scatter(name='Between Models', text=dfline.index.values.tolist(), mode='lines+markers', x=dfline.x, y=dfline.y, line=dict(width=5), marker=dict(size=10, opacity=1.0, line=dict(width=5)) )

st.write(fig)

annoid_input = st.sidebar.text_area('Anno IDs (comma separated)')
annosinspect = [annoid.strip().replace("'", "").replace("[", "").replace("]", "") for annoid in re.split(',',annoid_input) if len(annoid) > 1]

# st.write(annosinspect[0])
modelid = df_subset[df_subset['Anno ID'] == int(annosinspect[0])]['Model ID'].values[0]
# st.write(modelid)

pic_in_dir='/home/starstorms/Insight/ShapeNet/renders'
fullpath = os.path.join(pic_in_dir, modelid+'.png')
img = mpimg.imread(fullpath)
fig = plt.imshow(img)
st.image(img)


anids = [21786, 8479]
modelid1 = df_subset[df_subset['Anno ID'] == anids[0]]['Model ID'].values[0]
modelid2 = df_subset[df_subset['Anno ID'] == anids[1]]['Model ID'].values[0]
img1 = mpimg.imread(os.path.join(pic_in_dir, modelid1+'.png'))
img2 = mpimg.imread(os.path.join(pic_in_dir, modelid2+'.png'))
imgs = [img1, img2]

st.write('Before the empty')
empty = st.empty()
st.write('After the empty')

empty.text('Inside the empty')
time.sleep(2)

empty.image(img[0])

for i in range(5) : 
    time.sleep(1.0)
    empty.image(imgs[i % 2])
    