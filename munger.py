#%% Imports
import os,sys,shutil
import json

#%%
def moveFiles(treeroot, outDir, file_types):
    for ft in file_types : os.mkdir(os.path.join(outDir, ft))
    
    registry = [] # hash, index
    i = 0
    for dir,subdirs,files in os.walk(treeroot):
        for f in files : 
            fullpath = os.path.realpath( os.path.join(dir,f) )
            splits = fullpath.split('.')
            ext = splits[-1]
            if ('solid' in splits) : ext = 'solid.binvox'
            if ('surface' in splits) : ext = 'surface.binvox'
            if (ext in file_types) :
                category = dir.split('/')[-3]       
                file_name = '{}_{:05d}.{}'.format(category, i, ext)
                newpath = os.path.join(outDir, ext, file_name)
                # os.rename(fullpath, newpath)
                shutil.copyfile(fullpath, newpath)
                # print('From: {}\nTo:   {}\n'.format(fullpath, newpath))
                
        if ('models' in str(dir)) :
            hashid = dir.split('/')[-2]
            registry.append([i, hashid, '{}_{:05d}'.format(category, i)])
            i = i + 1
            
            if (i % 1000 == 0) : print(i)
            
    with open(os.path.join(outDir, 'registry.json'), 'w') as f:
        json.dump(registry, f, ensure_ascii=False, indent=4)
        
#%%
# rootDir = '/home/starstorms/Insight/ShapeNet/testing'
# outDir =  '/home/starstorms/Insight/ShapeNet/testingout'
rootDir = '/data/sn/all/all'
outDir  = '/data/sn/all'

file_types = ['json', 'solid.binvox']
moveFiles(rootDir, outDir, file_types)

#%%
def loadRegs(file_name):
    with open(file_name) as json_file :
        data = json.load(json_file)
    return data
        
# regs = loadRegs(os.path.join(outDir, 'registry.json'))