#%% Imports
import numpy as np
import os
import subprocess
from sys import getsizeof
import skimage.measure as sm
import time
import json
import pandas as pd
import random

import cvae as cv
import binvox_rw as bv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF
import matplotlib.cm as cm
# from pandas_ods_reader import read_ods

import tensorflow as tf

#%% Basic Functions
def minmax(arr) :
    print('Min: {} \nMax: {}'.format(np.min(arr), np.max(arr)))
    
def interp(vec1, vec2, divs=5, include_ends=True) :
    out = []
    amounts = np.array(range(divs+1))/divs if include_ends else np.array(range(1, divs))/divs
    for amt in amounts :
        interpolated_vect = vec1 * amt + vec2 * (1-amt)
        out.append( interpolated_vect )
    return np.stack(out)

def superSample(list_to_sample, samples) :
    pop = len(list_to_sample)
    if samples > pop :
        result = random.sample(list_to_sample, pop)
        for i in range(samples - pop) : result.append(random.choice(list_to_sample))
        return result
    else:
        return random.sample(list_to_sample, samples)

#%% ShapeNet
def getMeta(reg_fp = "/home/starstorms/Insight/ShapeNet/meta/meta.ods", sheet_name = 'meta'):    
    return read_ods(reg_fp, sheet_name)

def getJSON(json_fp) :
    json = pd.read_json(json_fp)
    return json

def getTax(tax_fn = "/home/starstorms/Insight/ShapeNet/meta/taxonomy.json") :
    tax = pd.read_json(tax_fn)
    tax['numc'] = tax.apply (lambda row: len(row.children), axis=1)
    return tax

def ranameVoxs(vox_in_dir, prefix) :
    for i, file in enumerate(os.listdir(vox_in_dir)) :
        fullpath = os.path.join(vox_in_dir, file)
        newpath = os.path.join(vox_in_dir, '{}_{:05d}.binvox'.format(prefix, i))
        print(fullpath, '\n', newpath, '\n')
        os.rename(fullpath, newpath)
    
def getMixedFPs(vox_in_dir, num_models_load, cat_prefixes) :
    vox_fps = os.listdir(vox_in_dir)
    cat_vox_fps = [ [item for item in vox_fps if item.startswith(cat)] for cat in cat_prefixes ]
    cat_vox_fps = [superSample(cat, int(num_models_load / len(cat_prefixes))) for cat in cat_vox_fps]
    
    vox_fps = []
    for cat_list in cat_vox_fps : vox_fps.extend(cat_list)
    return vox_fps

#%% 3D Model Functions
def loadBV(filename, coords = False, reduction_factor=1) :
    with open(filename, 'rb') as f:
        model = bv.read_as_3d_array(f, coords, reduction_factor)
    return model

def loadBVVariable(filename, coords = False, target_vox_size=64) :
    with open(filename, 'rb') as f:
        model = bv.read_as_3d_array_variable(f, coords, target_vox_size)
    return model

def makeOBJ(name, verts, faces) :
    F = open(name,'w+')    
    faces = faces + 1
    for vert in verts : F.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
    for face in faces : F.write("f {} {} {}\n".format(face[0], face[1], face[2]))    
    F.close()

def saveOBJs(model, voxs, out_folder, suffix) :
    voxs = voxs.numpy()[:,:,:,:,0]
    voxs[voxs >= 0.5] = 1.
    voxs[voxs <  0.5] = 0.
    
    for i, vox in enumerate(voxs) :
        vox = np.pad(vox, 1)
        verts, faces, normals, values = sm.marching_cubes_lewiner(vox, 0, step_size=1)
        makeOBJ('{}/{}{}.obj'.format(out_folder, suffix, i+1), verts, faces)

#%% Display Functions
def plotMesh(verts, faces) :
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
    plt.show()
    
def plotVox(voxin, step=1, title='', threshold = 0.5, stats=False, limits=None, show_axes=True) :
    vox = np.squeeze(voxin)
    
    if (stats) :
        plt.subplot(1, 2, 1)
        plt.hist(voxin.flatten(), bins=10)
    
    vflat = vox.flatten()
    if (threshold == None) : threshold = (np.min(vflat) + np.max(vflat)) / 2
    
    if (np.any(vox) == False) :
        print('No voxels for: {}'.format(title))
        vflat = voxin.flatten()
        plt.hist(vflat, bins=10)
        plt.suptitle(title)
        return

    if (stats) :
        plt.subplot(1, 2, 2)
        plt.hist(vox.flatten(), bins=10)
        plt.show()
        
    verts, faces = createMesh(vox, step, threshold)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if (not show_axes) :
        ax.axis('off')
    
    if (limits != None) :
        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
        ax.set_zlim(0, limits[2])
        
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
    plt.suptitle(title)
    plt.show()

def createMesh(vox, step=1, threshold = 0.5) :   
    vox = np.pad(vox, step)
    verts, faces, normals, values = sm.marching_cubes_lewiner(vox, 0.5, step_size=step)
    return verts, faces
  
def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1), plot_it=True) :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title="Mesh", aspectratio=aspect)
    if (plot_it) : plot(fig)
    return fig
    
def showReconstruct(model, item, index = 0, show_original=True, show_reconstruct=True) :
    xvox = item[index].numpy()[0,:,:,:,0]
    xprobs = model.reconstruct(item)
    predictions = xprobs.numpy()[0,:,:,:,0]
    
    if (np.max(predictions) < 0.5) :
        print('No voxels')
        return
    
    if (show_original) : plotVox(xvox, step=1, title='Original', threshold=0.5, stats=False)  
    if (show_reconstruct) : plotVox(predictions, step=1, title='Reconstruct', threshold=0.5, stats=False)
    
def startStreamlit(filepath) :
    os.subprocess.call('streamlit run {}'.format(filepath), shell=True)
    os.subprocess.call('firefox new-tab http://localhost:8501/')
    
def exportBinvoxes(in_dir, out_dir, obj_prefix, vox_size) :
    # Remove any .binvox files in directory
    subprocess.call('rm {}/*.binvox'.format(in_dir), shell=True)
    # Create binvox files     In bash it's this:    for f in objs/{obj_prefix}*.obj; do file=${f%%.*}; ./binvox ${file}.obj -pb -d {vox_size};  done;
    subprocess.call('for f in {}/{}*.obj; do file=${{f%%.*}}; ./binvox ${{file}}.obj -pb -d {};  done;'.format(in_dir, obj_prefix, vox_size), shell=True)
    # Rename files with voxel size and other info
    # subprocess.call('mv {}/*.binvox {}'.format(in_dir, out_dir), shell=True)
    # Move binvox files to output dir
    subprocess.call('mv {}/*.binvox {}'.format(in_dir, out_dir), shell=True)
    
#%% For plotting meshes side by side
def map_z2color(zval, colormap, vmin, vmax):
    #map the normalized value zval to a corresponding color in the colormap

    if vmin>vmax:
        raise ValueError('incorrect relation between vmin and vmax')
    t=(zval-vmin)/float((vmax-vmin))#normalize val
    R, G, B, alpha=colormap(t)
    return 'rgb('+'{:d}'.format(int(R*255+0.5))+','+'{:d}'.format(int(G*255+0.5))+\
           ','+'{:d}'.format(int(B*255+0.5))+')'

def tri_indices(simplices):
    #simplices is a numpy array defining the simplices of the triangularization
    #returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))

def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
    #x, y, z are lists of coordinates of the triangle vertices 
    #simplices are the simplices that define the triangularization;
    #simplices  is a numpy array of shape (no_triangles, 3)
    #insert here the  type check for input data

    points3D=np.vstack((x,y,z)).T
    tri_vertices= list(map(lambda index: points3D[index], simplices))# vertices of the surface triangles     
    zmean=[np.mean(tri[:,2]) for tri in tri_vertices ]# mean values of z-coordinates of triangle vertices
    min_zmean=np.min(zmean)
    max_zmean=np.max(zmean)
    facecolor=[map_z2color(zz,  colormap, min_zmean, max_zmean) for zz in zmean]
    I,J,K=tri_indices(simplices)

    triangles=go.Mesh3d(x=x,
                     y=y,
                     z=z,
                     facecolor=facecolor,
                     i=I,
                     j=J,
                     k=K,
                     name=''
                    )

    if plot_edges is None:# the triangle sides are not plotted 
        return [triangles]
    else:
        #define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        #None separates data corresponding to two consecutive triangles        
        Xe = []
        Ye = []
        Ze = []
        for T in tri_vertices:
            Xe.extend( [T[k%3][0] for k in range(4)] + [ None] )
            Ye.extend( [T[k%3][1] for k in range(4)] + [ None] )
            Ze.extend( [T[k%3][2] for k in range(4)] + [ None] )

        #define the lines to be plotted
        lines=go.Scatter3d(x=Xe,
                        y=Ye,
                        z=Ze,
                        mode='lines',
                        line=dict(color= 'rgb(70,70,70)', width=0.5)) 
        return [triangles, lines]
    
def plotlySurf(verts, faces) :
    x, y, z = verts[:,0], verts[:,1], verts[:,2]
    triangles, lines = plotly_trisurf(x,y,z, faces, colormap=cm.RdBu, plot_edges=True)
    return triangles, lines