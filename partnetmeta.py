'''
This file is used to read in the json files that contain the metadata from the PartNet meta data files and ultimately creates the first round of the dfmeta file.
It uses some complicated tree structures to capture the data and attempt to generate a description.
However, this method was mostly overwritten by the descriptor.py file which uses a different methodology to make the descriptions.
But that file still uses a large amount of the data generated here.

This file is only intended to be run locally.
'''

#%% Imports
import json
import os
import inflect
from nltk.corpus import wordnet as wn
import pandas as pd
import pandas_ods_reader as por
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils as ut
import configs as cf

#%% Setup text processing
inflect = inflect.engine()
vocab = set()

#%% Define loadset
cats_to_load = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']
catids_to_load = [4379243,3001627,3636649,3325088,3046257,2876657,3593526,3642806,2818832,3797390,2880940]

#%% Get JSON files and preprocess
def getAndPrintJSON(index, dir_fp, reverse=False, verbose=False) :
    indent = 2
    json_res_fp = os.path.join(dir_fp, str(index), 'result_after_merging.json')
    json_meta_fp = os.path.join(dir_fp, str(index), 'meta.json')
    
    meta = ut.getJSON(json_meta_fp)
    mid = meta['model_id']
    mcat = meta['model_cat']
    annoid = meta['anno_id']
    
    if (not mcat in cats_to_load) : return None
    if (verbose) : print('{}\n{}\n{}'.format(index, mid, mcat))
        
    res = ut.getJSON(json_res_fp)
    json_formatted_str = json.dumps(res, indent=indent)
    resl = json_formatted_str.splitlines()        
    
    output = []    
    index = 0
    for line in resl :
        if ('name' in line) :
            level = int(line.split(':')[0].count(' ') / (2*indent) - 1)
            name = line.split('"')[-2]
            name = name.replace('_', ' ')
            output.append({'name':name, 'level':level})
            vocab.add(name)
            index = index + 1
        
    if (output[0]['level'] > output[-1]['level']) :
        output.reverse()
        
    output.insert(0, {'mid':mid})
    output.insert(1, {'mcat':mcat})
    output.insert(2, {'annoid':annoid})

    return output
            
#%% Get all JSON data
json_dir = cf.PARTNET_META_STATS_DIR
json_fps = os.listdir(json_dir)
    
data = []
for fp in tqdm(json_fps, total=len(json_fps)) :
    shape = getAndPrintJSON(fp, json_dir, False)
    if (shape == None) : continue
    data.append(shape)
    
dfmids = pd.DataFrame([ [item[0]['mid'], item[1]['mcat'], item[2]['annoid']] for item in data], columns=['mid', 'cattext', 'annoid'])

#%% Explore vocab
def getSyns(word) :
    syns = wn.synsets(word)
    synonyms = []
    for syn in syns :
        for l in syn.lemmas() :
            synonyms.append(l.name())
    return synonyms

def getDef(word) :
    syns = wn.synsets(word)
    if len(syns) > 0 :
        return syns[0].definition()
    else :
        return 'N/A'

for word in vocab : 
    for w in word.split(' ') :
        syns = getSyns(w)
        print('{:32} {}'.format(w, set(syns[1:5])))
        
for word in vocab : 
    for w in word.split(' |') :
        print('{:32} {}'.format(w, getDef(w)))
        
for word in vocab :
    defs = wn.synsets(word)
    print(word)
    if len(defs) > 0 :
        print('{} : {}'.format(word, defs[0]))
    
#%% Node class for gathering tree info from the given part hierarchy
class Node:
    def __init__(self, name='root', level=0):
        self.name = name
        self.children = []
        self.level = level
        self.total_children = 0
        self.quantity = 1
        self.mid = ''
        self.mcat = ''
        self.desc_level = 0
        self.desc_max = 0
    def __repr__(self):
        return "{}".format(self.name)
  
    def printTree(self):
        for child in self.children:
            print('{:2d} : {} : {} : {}{}'.format(child.level, child.total_children, child.quantity, ' '*child.level*2, child.name)) 
            child.printTree()
    
    def getDetails(self, names_only=False) :
        desc = ''
        if not self.name == 'root' :
            desc = ' {},{},{},{}|'.format(self.name, self.level, len(self.children), self.quantity)
        if (names_only) : desc = '{}|'.format(self.name)
        for child in self.children :
            desc = desc + child.getDetails(names_only)
        return desc
    
    def treesEqual(self, root1, root2) :
        return (root1.getDetails() == root2.getDetails())
    
    def collapseMulti(self) :
        if (len(self.children) == 0) :
            return
            
        child_descs = [child.getDetails() for child in self.children]
        uniq_descs = set(child_descs)
        
        ids_to_remove = []
        for uniq in uniq_descs :
            found = False
            for i, entry in enumerate(child_descs) :
                if (uniq==entry and not found) :
                    # print(' {} : {} : {} '.format(entry, i, len(self.children)))
                    self.children[i].quantity = child_descs.count(uniq)
                    found = True
                elif (uniq==entry and found) :
                    # print('removed', entry, i)
                    ids_to_remove.append(i)     

        for index in sorted(ids_to_remove, reverse=True):
            del self.children[index]
        
        for child in self.children :
            child.collapseMulti()
    
    def removeZeros(self) :
        for child in reversed(self.children) :
            if (child.quantity == 0) :
                self.children.remove(child)
            child.removeZeros()
            if (child.quantity > 1) : child.name = inflect.plural(child.name)
    
    def sumUp(self) :
        total = len(self.children)
        for child in self.children :
            total = total + child.sumUp()
        self.total_children = total
        return total

    def dSmall(self) :
        names = list(filter(None, self.getDetails(names_only=True).split('|')))
        desc = ''
        if len(names) == 0 : desc = 'Nothing'
        if len(names) == 1 : desc = '{}.'.format(names[0])
        if len(names) == 2 : desc = '{} with a {}.'.format(names[0], names[1])
        if len(names) == 3 : desc = '{} with a {} made of a {}.'.format(names[0],names[1], names[2])
        if len(names) >= 4 : desc = 'Overload'
        return desc

    def dReg(self, index, outof) :
        if (len(self.children) == 0) :
            return 'OVERLOAD: {} '.format(self.name)
            # return ' {} {}.'.format( self.quantity if self.quantity>1 else 'a', self.name)
        
        if (len(self.children) == 1) :
            # multi = 'each of ' if self.quantity > 1 else ''
            multi = (self.quantity > 1)
            
            if (len(self.children[0].children) == 0) :
                if multi :
                    return 'each {} is made of {} {}. '.format( inflect.singular_noun(self.name), self.children[0].quantity if self.children[0].quantity>1 else 'a', self.children[0].name) 
                else :
                    return 'the {} is made of {} {}. '.format(self.name, self.children[0].quantity if self.children[0].quantity>1 else 'a', self.children[0].name)
            elif (len(self.children[0].children) == 1) :  # has just 1 child
                return 'the {} which is {} '.format(self.name, self.children[0].dReg(index+1, outof))
            else : # has multiple children
                return 'the {} which is {} '.format(self.name, self.children[0].dReg(index+1, outof))

        desc_subs = ''
        i = len(self.children)
        imax = i
        for child in self.children :
            singular = not inflect.singular_noun(child.name)
            multi = 'a' if singular else ' {} '.format(child.quantity)
            template = ', {} {} '
            if i==imax  : template = '{} {} '
            if i==1     : template = 'and {} {} '
            desc_subs = desc_subs + template.format(multi, child.name)
            i = i - 1
        return 'the {} has {}. '.format(self.name, desc_subs)
    
    def dRoot(self) :
        if (self.mcat == 'Vase') :
            return self.dRootVase()
        if (self.level<1 and self.total_children < 3) :
            return ' {} '.format(self.dSmall())

        if (len(self.children) > 1 or self.level >= 2) :
            desc_subs = ''
            desc_all_subs = ''
            i = len(self.children)
            imax = i
            for child in self.children :
                singular = not inflect.singular_noun(child.name)
                multi = 'a' if singular else 'a set of {}'.format(child.quantity)
                template = '{}, {} {}'
                if i==imax  : template = '{} {} {}'
                if i==1     : template = '{} and {} {}'
                desc_subs = template.format(desc_subs, multi, child.name)
                desc_all_subs = desc_all_subs + (child.dReg(1,0) if (len(child.children)>0) else '')
                i = i - 1
            return 'a {} that is made of {}. {}'.format(self.name, desc_subs, desc_all_subs)
        else :
            return '{} that is {}'.format( self.name, self.children[0].dRoot() )

    def dSmallVase(self) :
        names = list(filter(None, self.getDetails(names_only=True).split('|')))
        desc = ''
        if len(names) == 0 : desc = 'Nothing'
        if len(names) == 1 : desc = '{}.'.format(names[0])
        if len(names) == 2 : desc = '{} with a {}.'.format(names[0], names[1])
        if len(names) == 3 : desc = '{} with a {} made of a {}.'.format(names[0],names[1], names[2])
        if len(names) >= 4 : desc = 'Overload'
        return desc

    def dRegVase(self) :
        if self.name == 'containing things' :
            desc_subs = ''
            i = len(self.children)
            imax = i
            for child in self.children :
                singular = not inflect.singular_noun(child.name)
                multi = 'a' if singular else ' {} '.format(child.quantity)
                template = ', {} {} '
                if i==imax  : template = '{} {} '
                if i==1     : template = 'and {} {} '
                desc_subs = desc_subs + template.format(multi, child.name)
                i = i - 1
            return ' the {} contains {}. '.format('vase', desc_subs)
        
        if (len(self.children) == 0) :
            return 'OVERLOAD: {} '.format(self.name)
        
        if (len(self.children) == 1) :
            multi = (self.quantity > 1)
            
            if (len(self.children[0].children) == 0) :
                if multi :
                    return 'each {} is made of {} {}. '.format( inflect.singular_noun(self.name), self.children[0].quantity if self.children[0].quantity>1 else 'a', self.children[0].name) 
                else :
                    return 'the {} is made of {} {}. '.format(self.name, self.children[0].quantity if self.children[0].quantity>1 else 'a', self.children[0].name)
            elif (len(self.children[0].children) == 1) :  # has just 1 child
                return 'the {} which is {} '.format(self.name, self.children[0].dRegVase())
            else : # has multiple children
                return 'the {} which is {} '.format(self.name, self.children[0].dRegVase())

        desc_subs = ''
        i = len(self.children)
        imax = i
        for child in self.children :
            singular = not inflect.singular_noun(child.name)
            multi = 'a' if singular else ' {} '.format(child.quantity)
            template = ', {} {} '
            if i==imax  : template = '{} {} '
            if i==1     : template = 'and {} {} '
            desc_subs = desc_subs + template.format(multi, child.name)
            i = i - 1
        return 'the {} has {}. '.format(self.name, desc_subs)
    
    def dRootVase(self) :       
        new_children = self.children
        
        if (self.level<1 and (self.total_children) < 3) :
            return ' {} '.format(self.dSmallVase())

        if (len(new_children) > 1 or self.level >= 2) :
            desc_subs = ''
            desc_all_subs = ''
            i = len(new_children)
            imax = i
            for child in new_children :                    
                if child.name == 'containing things' :
                    desc_all_subs = desc_all_subs + child.dRegVase()
                    continue
                multi = 'a' if child.quantity > 0 else 'a set of {}'.format(child.quantity)
                template = '{}, {} {}'
                if i==imax  : template = '{} {} {}'
                if i==1     : template = '{} and {} {}'
                desc_subs = template.format(desc_subs, multi, child.name)
                desc_all_subs = desc_all_subs + (child.dRegVase() if (len(child.children)>0) else '')
                i = i - 1                
            return 'a {} that is made of {}. {}'.format(self.name, desc_subs, desc_all_subs) 
        else :
            return '{} that is {}'.format( self.name, new_children[0].dRootVase() )

'''
Exampple descriptions and tree structures :
    
a chair that has a chair back and a chair seat. the chair back is made of a back surface with a back 
    single surface. the chair seat is made of a seat surface with a seat single surface.
 0 : 6 : 1 : chair
 1 : 2 : 1 :   chair back
 2 : 1 : 1 :     back surface
 3 : 0 : 1 :       back single surface
 1 : 2 : 1 :   chair seat
 2 : 1 : 1 :     seat surface
 3 : 0 : 1 :       seat single surface


 0 : 12 : 1 : table
 1 : 11 : 1 :   regular table
 2 : 7 : 1 :     table base
 3 : 6 : 1 :       regular leg base
 4 : 0 : 2 :         circular stretchers
 4 : 0 : 4 :         legs
 2 : 2 : 1 :     tabletop
 3 : 1 : 1 :       tabletop surface
 4 : 0 : 1 :         board

 0 : 8 : 1 : pot
 1 : 2 : 1 :   containing things
 2 : 0 : 1 :     liquid or soil
 2 : 0 : 1 :     plant
 1 : 1 : 1 :   body
 2 : 0 : 1 :     container
 1 : 2 : 1 :   base
 2 : 1 : 1 :     foot base
 3 : 0 : 4 :       feet
 a pot that is made of a body and a base. it contains liquid or soil and a plant. the body is made of a container and the
     base is made of a food base with 4 feet.

'''

#%% Generate descriptions
phrases = {
    'combiner' : ['with', 'with', 'with', 'that has', 'made up of'],
    'starter' : ['a'],
    'multi' : ['each with', 'each one has', 'each having'],
    }

def getShapeTree(data, max_depth = 10) :
    fdata = [item for item in data[3:] if item['level'] <= max_depth]
    root = Node()
    root.mid = data[0]['mid']
    root.mcat = data[1]['mcat']
    root.annoid = data[2]['annoid']
        
    # print('{}  {}  {}'.format(len(data), root.mid, root.mcat, len(root.children)))
    
    for record in fdata:
        last = root
        for _ in range(record['level']):
              last = last.children[-1]
        last.children.append(Node(record['name'], record['level']))
           
    root.collapseMulti()
    # root.removeZeros()
    root.sumUp()
    return root

fix_replacements = [['  ', ' '],
                    [' .', '.'],
                    ['..', '.'],
                    [' ,', ',']]
def removeDescExtras(desc) :
    for rp in fix_replacements :
        desc = desc.replace(rp[0], rp[1])        
    if desc.startswith('a ') : desc = desc[2:]
    return desc

#%% Get shape descriptions based on tree hierarchies and create dataframe
# rn.shuffle(data)
cat_index = 3      # cats_to_load = [' 0 Table','1 Chair','2 Lamp','3 Faucet','4 Clock','5 Bottle','6 Vase','7 Laptop','8 Knife']
cat_data = [entry for entry in data if cats_to_load[cat_index] in entry[1]['mcat']]
shapes = [getShapeTree(shape, 4) for shape in data]

dfdesc = {}
for index, shape in enumerate(shapes) :
        desc = '{}'.format(shape.children[0].dRoot()) if not shape.mcat == 'Vase' else '{}'.format(shape.children[0].dRootVase())
        desc = removeDescExtras(desc)
        details = shape.getDetails()
        dfdesc[shape.mid] = [shape.mcat, shape.annoid, desc, details]
        # print('\nIndex: {:2d}   {}    {}\n{}'.format(index, shape.mcat, shape.mid, desc))
        # shape.prinTree()
        
dfdesc = pd.DataFrame.from_dict(dfdesc, orient='index')
dfdesc.columns = ['cattext', 'annoid', 'desc', 'details']
dfdesc.index.name = 'mid'

#%% Inspect trees
for i, shape in enumerate(shapes[:20]) :
    print('\nIndex: {}'.format(i))
    shape.printTree()

#%% Load all meta data filepaths from ShapeNetCore database
meta_dir = '/media/starstorms/DATA/Insight/ShapeNet/stats/ShapeNetCore.v2'
meta_fps = []

for dirName, subdirList, fileList in os.walk(meta_dir):
    for fname in fileList:
        fullpath = os.path.join(dirName, fname)
        meta_fps.append(fullpath)

#%% Get all metadata and put into DF (takes a long time, use precomputed below)
dfmeta = pd.DataFrame(columns = ['mid', 'cat', 'numv',
                                 'xmin', 'xmax', 'centx', 'dx',
                                 'ymin', 'ymax', 'centy', 'dy',
                                 'zmin', 'zmax', 'centz', 'dz'])
i = 0
for meta_fp in tqdm(meta_fps, total=len(meta_fps)) :
    meta_js = ut.getJSON(meta_fp)
    mcat = meta_fp.split('/')[8]
    dfmeta.loc[i] = [meta_js['id'], mcat, meta_js['numVertices'],
                     meta_js['min'][0], meta_js['max'][0], meta_js['centroid'][0], meta_js['max'][0] - meta_js['min'][0],
                     meta_js['min'][1], meta_js['max'][1], meta_js['centroid'][1], meta_js['max'][1] - meta_js['min'][1],
                     meta_js['min'][2], meta_js['max'][2], meta_js['centroid'][2], meta_js['max'][2] - meta_js['min'][2]  ]
    i = i + 1
    
#%% Write / reload the data from the previous cell
# pd.DataFrame.to_csv(dfmeta, '/home/starstorms/Insight/ShapeNet/meta/df_meta_raw.csv')
dfmeta = pd.read_csv('/home/starstorms/Insight/ShapeNet/meta/df_meta_raw.csv')

#%% Read tax and meta info
tax = ut.readTax()
tax_cats = tax[tax.synsetId.isin(catids_to_load)]
# dfmeta = ut.readMeta()
# dfmeta.drop(['desc','cattext'], axis=1, inplace=True)

#%% Fix and normalize numeric columns of interest
dffixcols = ['dx','dy','dz', 'dsq']
dfnormcols = ['dx','dy','dz','dsq', 'numv']
dfall = pd.DataFrame.merge(dfmeta, dfdesc[['cattext', 'annoid', 'desc', 'details']], how='left', on=['mid', 'mid'])
dfall = dfall.drop_duplicates(subset='mid')
dfall['dsq'] = abs(abs(dfall.dx) - abs(dfall.dz))
dfall[dffixcols] = dfall[dffixcols].div(dfall[dffixcols].sum(axis=1), axis=0)

# dfall[dfnormcols] = dfall[dfnormcols].apply(stats.zscore)
# dfstats = [dfall[dfall.cattext==cattxt][dfnormcols].describe().reset_index() for cattxt in cats_to_load]
 
#%% Create shape overall classes based on bboxs
duniq = dfall.cat.unique()
qbins = [0, .1, .3, .7, .9, 1.0]
dfclasscols = [col.replace('d','c') for col in dffixcols]
for col in dfclasscols : dfall[str(col)] = int(len(qbins)/2)
for col1, col2 in zip(dffixcols, dfclasscols) :
    for catid_uniq in duniq :
        dfall[col2].loc[dfall.cat==catid_uniq] = pd.qcut(dfall[col1].loc[dfall.cat==catid_uniq], labels=False, q=qbins, precision=0, duplicates='drop')

#%% Show distributions of various size extremes in each dimension
for catid in cats_to_load :
    dfds = dfall[dfall.cattext == catid][dfnormcols]
    
    for i, d in enumerate(dfnormcols) :
        mean = dfds[d].mean()
        stdev = dfds[d].std()
        lines = np.arange(-1, 2) * .674 * stdev + mean
        
        plt.subplot(5, 1, i+1)        
        plt.hist(dfds[d], bins = 50)
        for line in lines : plt.axvline(line, color= 'black' if line==mean else 'r', linewidth=1.5 )
    plt.suptitle(catid)    
    plt.show()

#%% Add in sub categories
dfsubcats = por.read_ods('/home/starstorms/Insight/ShapeNet/meta/meta.ods', 'reg')
subcatcols = ['mid', 'subcats']
dfsubcats = dfsubcats[subcatcols]
dfsubcats = dfsubcats[dfsubcats.mid.isin(dfall.mid)]
dfsubcats = dfsubcats.drop_duplicates()
dfall = pd.DataFrame.merge(dfall, dfsubcats, how='left', on=['mid', 'mid'])

#%% Rearrange columns and save the completed dfmeta file
dcols = ['mid','cat','cattext','annoid','subcats','desc','details','dx','dy','dz','dsq','cx','cy','cz','csq']
dfall[dcols].to_csv('/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv')

