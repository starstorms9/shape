#%% Imports
import numpy as np
import os
import subprocess
# if (not os.path.isdir('/data/sn/all')) : subprocess.call('sync_scripts.sh', shell=True)

import skimage.measure as sm
import time
import json
import pandas as pd
import random
from tqdm import tqdm
from scipy import signal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF
import tensorflow as tf

#%% Global variables
dfmeta_fp = '/data/sn/all/meta/dfmeta.csv'
plot_out_dir = ''
tax = []
meta = []
kernel = 'none'

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

#%% Data methods
def addTimeStamp(path='') :
    os.environ['TZ'] = 'US/Pacific'
    time.tzset()
    return path + time.strftime("%m%d-%H%M")

def getSubDirs(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]    

def getMixedFPs(vox_in_dir, num_models_load, cat_prefixes) :
    vox_fps = os.listdir(vox_in_dir)
    cat_vox_fps = [ [ os.path.join(cat, directory, 'models/model_normalized.solid.binvox') for directory in os.listdir(os.path.join(vox_in_dir, cat))] for cat in cat_prefixes ]
    cat_vox_fps = [superSample(cat, int(num_models_load / len(cat_prefixes))) for cat in cat_vox_fps]
    
    vox_fps = []
    for cat_list in cat_vox_fps : vox_fps.extend(cat_list)
    return vox_fps

def splitData(dataset, test_split) :
    dataset_size = 0
    for _ in dataset : dataset_size = dataset_size + 1
    
    train_size = int((1-test_split) * dataset_size)
    test_size = int(test_split * dataset_size)
    
    shuffled_set = dataset.shuffle(100000)
    train_dataset = shuffled_set.take(train_size)
    test_dataset = shuffled_set.skip(train_size).take(test_size)
    return train_dataset, test_dataset

def loadData(target_vox_size, max_loads_per_cat, vox_in_dir, cat_prefixes):    
    # vox_fps = getMixedFPs(vox_in_dir, num_models_load, cat_prefixes)    
    cat_vox_fps = [ [ os.path.join(cat, directory, 'models/model_normalized.solid.binvox') for directory in os.listdir(os.path.join(vox_in_dir, cat))] for cat in cat_prefixes ]    
    voxs, mids = [], []
    too_sparse_count = 0
    grown_count = 0
    
    for i, vox_fps in enumerate(cat_vox_fps) :
        index = 0
        for _, file in tqdm(enumerate(vox_fps), unit_scale=True, desc='Loading {} cat {}/{}'.format(cat_prefixes[i], i+1, len(cat_prefixes)), total=min(max_loads_per_cat, len(vox_fps))):
            fullpath = os.path.join(vox_in_dir, file)
            try :        
                vox = readBV(fullpath, target_vox_size = target_vox_size)
                sparsity = getSparsity(vox)
                if sparsity < .01 : 
                    too_sparse_count = too_sparse_count + 1
                    continue
                if sparsity < .4 : 
                    vox = growVox(vox, amount=.1)
                    grown_count = grown_count + 1
                vox = reduceVoxels(vox, target_vox_size)
                vox = centerVox(vox)
                vox = np.expand_dims(vox, axis=-1)
                vox = np.bool8(vox)
                voxs.append(vox)
                mids.append(fullpath.split('/')[-3])
                index = index + 1
                if (index >= max_loads_per_cat) : break
            except :
                print('\nCould not load: ', fullpath)
    
    # voxs = np.stack(voxs, axis=0)
    print('Done. Discared Sparse Models: {}   Grown Models: {}'.format(too_sparse_count, grown_count))
    return voxs, mids

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

def readBV(filename, coords = False, target_vox_size=64) :
    with open(filename, 'rb') as fp:
        dims, translate, scale = read_header(fp)
        raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool)
        data = data.reshape(dims)
    return data

def getJSON(json_fp, df=False) :
    if (df) :
        json_file = pd.read_json(json_fp)
    else :
        with open(json_fp, 'r') as json_file:
            json_file = json.load(json_file)
    return json_file

def readTax(tax_fn = "/home/starstorms/Insight/ShapeNet/meta/taxonomy.json") :
    global tax
    tax = pd.read_json(tax_fn)
    tax['numc'] = tax.apply (lambda row: len(row.children), axis=1)
    return tax

def readMeta(meta_fn = "/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv") :
    global meta
    meta = pd.read_csv(meta_fn)
    return meta

def getMidCat(modelid) :
    global meta
    if (len(meta) < 2) :
        meta = readMeta()
    return meta.cat[meta.mid == modelid].to_numpy()[0]

def getCats(labels_tensor, cf_cat_prefixes) :
    output = ['0{}'.format(getMidCat(item.numpy().decode())) for item in labels_tensor]
    outcats = [cf_cat_prefixes.index(item) if item in cf_cat_prefixes else len(cf_cat_prefixes) for item in output]
    return tf.convert_to_tensor(outcats, dtype=tf.int32)

def getCatName(catid) :
    global tax
    if (len(tax) == 0) :
        tax = readTax()
    return tax.name[tax.synsetId == int(catid)].to_numpy()[0]    

def renameVoxs(vox_in_dir, prefix) :
    for i, file in enumerate(os.listdir(vox_in_dir)) :
        fullpath = os.path.join(vox_in_dir, file)
        newpath = os.path.join(vox_in_dir, '{}_{:05d}.binvox'.format(prefix, i))
        print(fullpath, '\n', newpath, '\n')
        os.rename(fullpath, newpath)

#%% 3D Model Functions
def showBinvox(modelid, vox_in_dir = '/media/starstorms/DATA/Insight/ShapeNet/all') :
    category = '0{}'.format(getMidCat(modelid))
    fullpath = '{}/{}/{}/models/model_normalized.solid.binvox'.format(vox_in_dir, category, modelid)
    subprocess.call('viewvox {}'.format(fullpath), shell=True)

def showPic(modelid, title='', pic_in_dir='/home/starstorms/Insight/ShapeNet/renders') :
    fullpath = os.path.join(pic_in_dir, modelid+'.png')
    img = mpimg.imread(fullpath)
    plt.suptitle(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def annoToMid(annoid) :
    return meta[meta.annoid== annoid].mid.values[0]

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

def reduceVoxels(vox, target_vox_size) :
    dim = vox.shape[0]
    reduction_factor = int(dim/target_vox_size)
    if (reduction_factor > 1) :
        vox = vox[0::reduction_factor, 0::reduction_factor, 0::reduction_factor]
    return vox

def getSparsity(vox) :
    vox = np.squeeze(np.array(vox))
    vox = np.where(vox>0.5, 1, 0)
    if (len(vox.shape)==3):
        return np.sum(vox) / (vox.shape[1] ** 3)
    if (len(vox.shape)==4):
        result = np.squeeze(np.apply_over_axes(np.sum, vox, [1,2,3]))
        return result / (vox.shape[1] ** 3)

def makeGaussKernel() :
    global kernel
    sigma = 1.0
    # x = np.arange(-3,4,1)
    # y = np.arange(-3,4,1)
    # z = np.arange(-3,4,1)
    x = np.arange(-6,7,1)
    y = np.arange(-6,7,1)
    z = np.arange(-6,7,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

def growVox(vox, amount=0.5) :
    global kernel
    if (kernel=='none') : makeGaussKernel()
    blurred = signal.convolve(vox, kernel, mode="same")
    blurred = np.where(blurred>amount, 1, 0)
    return blurred

def centerVox(voxin) :
    voxel_max_size = voxin.shape[0]
    vflatx = np.pad(np.max(np.max(voxin, axis=0), axis=1),1)
    vflatz = np.pad(np.max(np.max(voxin, axis=2), axis=1),1)
    
    xmin, zmin, xmax, zmax = 0,0,0,0
    for i in range(1,len(vflatx)) :
        jump = round(vflatx[i])-round(vflatx[i-1])
        if jump>0 : xmin = i
        if jump<0 : xmax = i
        
    for i in range(1, len(vflatz)) :
        jump = round(vflatz[i])-round(vflatz[i-1])
        if jump>0 : zmin = i
        if jump<0 : zmax = i
    
    xmin, xmax, zmin, zmax = xmin-1, xmax-1, zmin-1, zmax-1    
    x_off = int((voxel_max_size-(xmax-xmin))/2) - xmin
    z_off = int((voxel_max_size-(zmax-zmin))/2) - zmin
    vcent = np.roll(voxin, x_off, axis=1)
    vcent = np.roll(vcent, z_off, axis=0)
    return vcent

def getMetric(original_voxs, generated_voxs) :
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(original_voxs, generated_voxs)).numpy() * 1000

def checkStopSignal(dir_path='/data/sn/all/'):
    stop_path = os.path.join(dir_path, 'stop')
    go_path = os.path.join(dir_path, 'go')
    if (os.path.isdir(stop_path)):
        os.rename(stop_path, go_path)
        return True
    else :
        return False

#%% Display Functions
def plotMesh(verts, faces) :
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
    plt.show()
    
def plotVox(voxin, step=1, title='', threshold = 0.5, stats=False, limits=None, show_axes=True, save_fig=False, show_fig=True) :
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
        
    try :
        verts, faces = createMesh(vox, step, threshold)
    except :
        print('Failed creating mesh for voxels.')
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if (not show_axes) :
        ax.axis('off')
    
    if (limits != None) :
        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
        ax.set_zlim(0, limits[2])
        
    _ = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
    plt.suptitle(title)

    global plot_out_dir   
    if (save_fig) : _ = fig.savefig(os.path.join(plot_out_dir, title))
    if (show_fig) : _ = plt.show()
    return

def createMesh(vox, step=1, threshold = 0.5) :   
    vox = np.pad(vox, step)
    verts, faces, normals, values = sm.marching_cubes_lewiner(vox, threshold, step_size=step)
    return verts, faces
  
def showMesh(verts, faces, aspect=dict(x=1, y=1, z=1), plot_it=True) :
    fig = FF.create_trisurf(x=verts[:,0], y=verts[:,1], z=verts[:,2], simplices=faces, title="Mesh", aspectratio=aspect)
    if (plot_it) : plot(fig)
    return fig
    
def showReconstruct(model, samples, index = 0, title='', show_original=True, show_reconstruct=True, save_fig=False, limits=None) :
    predictions = model.reconstruct(samples[index][None,...], training=False)
    xvox = samples[index]
    
    if (np.max(predictions) < 0.5) :
        print('No voxels')
        return
    
    if (show_original) : plotVox(xvox, step=1, title='Original {}'.format(title), threshold=0.5, stats=False, save_fig=save_fig, limits=limits)  
    if (show_reconstruct) : plotVox(predictions, step=1, title='Reconstruct {}'.format(title), threshold=0.5, stats=False, save_fig=save_fig, limits=limits)
    
def startStreamlit(filepath) :
    subprocess.call('streamlit run {}'.format(filepath), shell=True)
    subprocess.call('firefox new-tab http://localhost:8501/')
    
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