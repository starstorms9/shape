'''
This file is used to generate the TSNE maps from the shape2vec file generated in vae.py

It visualizes the TSNE maps in a variety of ways including 2D and 3D maps with plotly and seaborn.
It also outputs the vectors in a pandas dataframe file that can be read in by the streamlit app for interactive visualization.
'''

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns

import utils as ut
import configs as cf

#%% Read text2vec pickle
shape_run_id = '0209-0306'
run_root_dir = os.path.join(cf.SHAPE_RUN_DIR, shape_run_id)

shape2vec = ut.loadPickle(os.path.join(run_root_dir,"shape2vec.pkl"))
shape2loss = ut.loadPickle(os.path.join(run_root_dir,"shape2loss.pkl"))    

bright = ["#023EFF","#FF7C00","#1AC938","#E8000B","#8B2BE2","#9F4800","#F14CC1","#A3A3A3","#000099","#00D7FF","#222A2A"]

#%% Run TSNE on the latent vectors
latent_dim = shape2vec.get('52255064fb4396f1b129901f80d24b7b').shape[0]
latent_vects = np.zeros((len(shape2vec), latent_dim))
for i, key in enumerate(shape2vec.keys()):
    latent_vects[i,:] = shape2vec[key]

perp, lr = 40, 200
tsne = TSNE(n_components=2, n_iter=1100, verbose=3, perplexity=perp, learning_rate=lr)
lvects = tsne.fit_transform(latent_vects)
plt.scatter(lvects[:,0], lvects[:,1], s=0.99, marker='.')

#%% Put TSNE data into frames and format
dfmeta = ut.readMeta()
df_subset = pd.DataFrame(list(shape2vec.keys()), columns=['mid'])
df_subset['tsne1'] = lvects[:,0]
df_subset['tsne2'] = lvects[:,1]
if (lvects.shape[1])==3 : df_subset['tsne3'] = lvects[:,2]

df_subset = pd.merge(df_subset, dfmeta, how='left', on=['mid', 'mid'])
dfloss = pd.DataFrame.from_dict(shape2loss, orient='index', columns=['loss'])
dfloss['logloss'] = np.log(dfloss.loss)
dfloss.index.name = 'mid'
df_subset = pd.merge(df_subset, dfloss, how='left', on=['mid', 'mid'])

# Just to rescale them for easy viewing
df_cols_to_increase = ['dx', 'dy', 'dz', 'dsq']
for col in df_cols_to_increase :
    df_subset[col] = (df_subset[col] - df_subset[col].min() + .1) * .05

#%% Plot 2D tsne with seaborn
plt.rcParams['figure.figsize'] = (12, 8)
sns.scatterplot(data=df_subset, x='tsne1', y='tsne2', hue='cattext', s=10, linewidth=0, palette='bright')
plt.axis('off')

#%% Plotly 2D tsne
fig = px.scatter(data_frame=df_subset.dropna(), hover_data=['subcats', 'mid'], size='dz',
                 x='tsne1', y='tsne2', color='cattext')
fig.update_traces(marker= dict(size=6, opacity=1.0, line=dict(width=0.0))) # size=2.3
plot(fig, filename = 'tsne_plot.html', auto_open=False, config={'scrollZoom': True, 'modeBarButtonsToRemove' : ['lasso2d','zoom2d']})

#%% Plotly 3D tsne
fig = px.scatter_3d(data_frame=df_subset.dropna(), hover_data=['subcats', 'mid'], size='dy',
                    x='tsne1', y='tsne2', z='tsne3', color_discrete_sequence=bright, color='cattext')
fig.update_traces(marker= dict(size=2.3, opacity=1.0, line=dict(width=0))) # size=2.3
plot(fig, filename = 'tsne_plot.html', auto_open=False)

#%% Save tsne vectors
df_sl_cols_keep = ['mid', 'tsne1', 'tsne2', 'cat', 'dx', 'dy', 'dz', 'cattext', 'dsq', 'cx', 'cy', 'cz', 'csq', 'subcats']
df_subset[df_sl_cols_keep].to_csv( os.path.join(cf.DATA_DIR, 'df_sl.csv') )