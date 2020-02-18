#%% Imports
import numpy as np
import os
# os.chdir('/home/ubuntu/sn/scripts')
import subprocess
from sys import getsizeof
import skimage.measure as sm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from collections import Counter
import time
import json
import pandas as pd
import random
import inspect
import pickle
import re
from tqdm import tqdm
import random as rn
import nltk
import inflect
import spacy
from nltk.corpus import wordnet as wn
import math

import cvae_dev as cv
import binvox_rw as bv
import utils as ut
import model_helper as ml
import text as tx
import logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
np.set_printoptions(precision=3, suppress=True)
remote = os.path.isdir('/data/sn/all')

#%% Setup text processing
inflect = inflect.engine()

# nltk.download()
vocab = set()
corpdict = {}

cats_to_load = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']
catids_to_load = [4379243,3001627,3636649,3325088,3046257,2876657,3593526,3642806,2818832,3797390,2880940]

meta_fp_local = '/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv'
meta_fp_remote = '/data/sn/all/meta/dfmeta.csv'
dfmeta = pd.read_csv(meta_fp_remote if remote else meta_fp_local)
dfdesc = dfmeta[dfmeta.details.notnull()]
dfcaps = pd.read_csv('/home/starstorms/Insight/ShapeNet/text2shape/captions.tablechair.csv')
dfcaps.columns = ['id', 'mid', 'caption', 'cattext', 'topLevelSynsetId', 'subSynsetId']

dfcapsall = pd.merge(dfmeta, dfcaps, how='inner', on=['mid','mid'])

#%% Build corpus of frequency 
buildCorpus(dfdesc)

#%% Generate all descriptions
dfmeta['desc'] = dfmeta.apply(lambda row: dRow(row), axis=1)
dfmeta.to_csv('/home/starstorms/Insight/ShapeNet/meta/dfmeta.csv')

#%%
dfsingles = dfdesc.groupby('desc').mean()

#%% Setup spacy
def get_embeddings(vocab):
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors

nlp = spacy.load('en_core_web_md')
embeddings = get_embeddings(nlp.vocab)




#%% Playing with Spacy
vector1 = nlp("large")[0].vector
vector2 = nlp("small")[0].vector
vector3 = vector1 - vector2

vector4 = nlp("high")[0].vector
vector5 = vector3 + vector4


cvs = closestVect(nlp("high")[0].vector)
cvs = closestVect(vector5)
for w in cvs : print(w.text)


#%%tvec = nlp("Queen")[0].vector
most_similar = nlp.vocab.vectors.most_similar(tvec.reshape(1,tvec.shape[0]))

result = nlp.vocab.vectors.most_similar(vector1)
result = nlp.vocab.vectors.most_similar(vector2)
result = nlp.vocab.vectors.most_similar(vector3)

i = nlp.vocab.vectors.key2row[5247273317732208552]
nlp.vocab.vectors.data[i]


#%%
def most_similar(word):
   queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15 and w.has_vector]
   by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
   return by_similarity[:10]

def closestVect(vect):
   queries = [w for w in nlp.vocab if w.prob >= -15 and w.has_vector]
   by_similarity = sorted(queries, key=lambda w: cosine_similarity(w.vector.reshape(1,-1), vect.reshape(1,-1)), reverse=True)
   return by_similarity[:10]

model_embeds = txtmodel.layers[0].get_weights()[0]
siminlp = lambda x, y : cosine_similarity(nlp.vocab[x].vector.reshape(1,-1), nlp.vocab[y].vector.reshape(1,-1))[0][0]
simmodel = lambda x,y : cosine_similarity(model_embeds[nlp.vocab[x].rank].reshape(1,-1), model_embeds[nlp.vocab[y].rank].reshape(1,-1))[0][0]

def compareSim(x,y) :
    print('{:.3f}  {:.3f}'.format(siminlp(x,y), simmodel(x,y)))

#%%
syns = most_similar(nlp.vocab['high'])
for w in syns:
    print(w.text, w.cluster)


#%% Part of speech tagging
samples=  [
    'easy chair that is very small and very tall and very thin. it has two wide chair arms that are large. it has two legs. it has two runners.',
    'easy couch that is average size and regular height and wide.  it has two sofa style chair arms .  it has four legs .',
    'easy couch that is average size and short and wide.  it has a back frame made of one back support .  it has two sofa style chair arms .  it has four legs .',
    'easy couch that is large and regular height and wide and long.  it has two sofa style chair arms .  it has four legs .',
    'easy couch that is large and short and regular width and very long.  it has a back frame made of three back supports .  it has four legs . it has two leg connecting bars .',
    'easy couch that is very large and short and very wide and long.  it has two sofa style chair arms .  it has four feet .',
    'easy couch that is very large and very short and very wide and very long.  it has two sofa style chair arms .  it has four feet .',
    'faucet that is regular. it has no switches. it has no spouts ',
    'faucet that is regular. it has three switches and two parts. it has one spout ',
    'faucet that is tall and large. it has two switches. it has one spout ',
    'a tall cream colour chair with wooden base and backrest and metal legs',
    'folding chair that is average size and regular height and regular width and long.  it has a back frame made of two vertical bars and six horizontal bars and two connectors .  it has two  chair arms .  it has four legs . it has two leg connecting bars .',
    'It is classical chair, It has no arm rests. Its back is see-through.',]

text = nlp(samples[-5])
for token in text:
    print('{:12}  {:5}  {:6}  {}'.format(str(token), str(token.tag_), str(token.pos_), str(spacy.explain(token.tag_))))

print('\n\n')
for token in text:
    print('{:10}  {:8}  {:8}  {:8}  {:8}  {:15}  {}'.format(token.text, token.dep_, token.tag_, token.pos_, token.head.text,
              str([item for item in token.conjuncts]), [child for child in token.children]))


doc = nlp("I like apples and oranges")
apples_conjuncts = doc[2].conjuncts

#%%
doc = nlp(samples[0])
shtml = spacy.displacy.render(doc, page=True, style="dep")

#%% Aggregator method
def augmentMutli(mids) :
    mids = list(np.squeeze(np.array(mids)))
    descs = [augmentDesc(  (mid.decode() if type(mid)==bytes else mid )  ) for mid in mids]
    # descs = [augmentDesc(  (mid.decode if type(mid)==bytes else mid )  ) for mid in mids]
    return descs

def augmentDescMany(mid, count) :
    return [augmentDesc(mid) for _ in range(count)]

def augmentDesc(mid) :
    row = dfdesc[dfdesc.mid==mid]
    row = addInSubcat(row)
    row = addShapeDesc(row)
    row = deleteSentences(row, rate=.4)
    row = shuffleSentences(row)    
    row = fixPuncs(row)    
    return row.desc.values[0]

'''
0   04379243   Table    8436
1   03001627   Chair    6778
2   03636649   Lamp     2318
3   03325088   Faucet   744
4   03046257   Clock    651
5   02876657   Bottle   498
6   03593526   Vase     485
7   03642806   Laptop   460
8   02818832   Bed      233
9   03797390   Mug      214
10  02880940   Bowl     186
'''

'''
Bed:    -special cases for subcats
        -if just 'bed' check if 'hammock'
        -note number of pillows
        -note sum of 'bed units'
        -note if ladder is present and #
        -note headboard if present
        -note quite done

Lamp:   -'table lamp' or 'floor lamp' if present in subcats. Ceiling, wall, or street lamp if in level 1 pos. If ceil, check level 2 for specific category.
        -always note number of chains if present. Go to many for >4
        -can say just 'lamp'
        -note 'lamp arm curved bar' or straight
        -ignore 'lamp' or 'lamp arm' in any subpart names
        -note total number of 'lamp units'. if just 1, check if > 1 lamp heads
        -note # of lamp poles if present
        -not quite done here

Faucet: -Just faucet, spigot, or sink (rarely) for subcat.
        -Note if 'shower faucet' (make subcat) always in second position (only level 1 item)
        -Only note height and squareness in shape. Squareness as long or short
        -Always note # switches and # of spouts
        -Note any horizontal or vertical supports and # if > 1. maybe replace support with bar or other
        -Note sum of qtys of frames in level 2 as parts if > 1. could be switches, spigots, or other
            
Clock:  rare subcat. note foot(s), box, screen, . # of chains. any qtys > 1. note 'base' if present. if nothing notable, note rare thing. ignore 'alarm' in subcats, use 'alarm clock'. note any foot(s)
Bottle: rare subcat. note 'lid', 'neck', 'mouth', 'handle'(s), ''. 'closure'/'lid'
Vase:   rare subcat. note 'lid', 'base' if present.
        note what it contains (may be nothing, or 'liquid or soil' or 'plant'). note if its empty. note nots.
        shape just note tall / short and wide / skinny
Laptop: laptop/computer. Note touchpad, keyboard, and screen. Note if screen closed
Mugs:  rare subcat. Note containing things or empty. Note handle or no. Note multiple handles
Bowls: just bowl. If containing things then say it's full. Note any qty > 1 items. note if it has a 'bottom' which is a base

Category: 
Nouns: 
Descriptors: shape, specific
Contains: 




'''

#%% Inspect some objects
catid = 1
while True:
    index = rn.randint(1, len(dfdesc)-1)
    row = dfdesc.iloc[index]
    if (not row.cattext == cats_to_load[catid]) : continue    
    print('-------------------------------------------{}------------------------------------------------'.format(index))
    print('{:20}    {}'.format(row.cattext, row.mid))

    subcats = row.subcats.split(',')
    rarity = 9999
    try : 
        for cat in subcats :
            rarity = min(rarity, getFreq(cat))
            print('{:5d}  {}'.format(getFreq(cat), cat))
    except : ' {} not in corpus '.format(cat)
    # if (rarity > 200) : continue
    
    print(getShapeDesc(row)[1])
    # printDetArr(detToArr(row.details))
    printShapeClasses(row)
    print(dRow(row))
    
    ut.showPic(dfdesc.iloc[index].mid,title=index)
    try : 
        i = input('')
        if i=='s' : 
            print('showing binvox...')
            ut.showBinvox(row.mid)
    except (KeyboardInterrupt, SystemExit): break

#%% Adding in shape info
odd_words = ['finial', 'windsor', 'tete-a-tete', 'vis-a-vis', 'zigzag chair', 'barcelona chair', 'longcase clock', 'tulip chair', 'Morris chair', 'crt screen', 'hydrant', 'tap']
space_splits = ['worktable', 'loveseat']
custom = ['containing things', 'glass']
sizes = [['very skinny', 'skinny', 'regular', 'long', 'very long'], # x axis options (lengthwise)  0
         ['very short', 'short', 'regular', 'tall', 'very tall'],   # y axis options (heightwise)  1
         ['very thin',  'thin',  'regular', 'wide', 'very wide'],   # z axis options (widthwise)   2
         ['very square', 'square', 'regular', 'rectangular', 'elongated']] # square options (squarewise)  3
# flat, slim, skinny, boxy, square
templates = [' it is {0} in length. it is {1} in height. it is {2} in width. it is {3} in shape. ',
             ' it is {0} and {1}. it is {3} in shape. ',
             ' it has a {3} shape. it is {0} and {1}. ',
             ' it has a {3} shape. it is {2} and {1}. ',
             ' the shape of the object is {3}. it is {1}. ',
             ' it is {0} in length and {1} in height. it is {2} in width. it is {3} in shape. ',
             ' the object is {0} in length and {1} in height. it is {2} in width. it is {3} in shape. ',
             ' even more {0}. even more {1}. ',
             ' even more {1}. even more {2}. ',
             ' even more {2}. even more {3}. ',
             ' even more {3}. even more {0}. ',
             ' it is {0} but also {2}. it is {1}. it is {3} in shape. ',
             '']

def dRow(row) :
    if   row.cattext == 'Bowl': return dBowl(row)
    elif row.cattext == 'Mug': return dMug(row)
    elif row.cattext == 'Laptop': return dLaptop(row)
    elif row.cattext == 'Vase': return dVase(row)
    elif row.cattext == 'Bottle': return dBottle(row)
    elif row.cattext == 'Clock': return dClock(row)
    elif row.cattext == 'Faucet': return dFaucet(row)
    elif row.cattext == 'Lamp': return dLamp(row)
    elif row.cattext == 'Bed': return dBed(row)
    elif row.cattext == 'Chair': return dChair(row)
    elif row.cattext == 'Table': return dTable(row)
    else : return row.subcats

chairSizes = [['very small', 'small', 'average size', 'large', 'very large'], # size options 0
              ['very short', 'short', 'regular height', 'tall', 'very tall'],   # y axis options (heightwise)  1
              ['very thin',  'thin',  'regular width', 'wide', 'very wide'],   # x axis options (widthwise)   2
              ['', '', '', 'long', 'very long']] # length options (z)  3
def dChair(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=5, exclude='armchair')
    if ('sofa' in subcat) : subcat = rn.choice(['sofa', 'couch', 'sofa couch', 'lounge chair'])
    if ('camp chair' in subcat) : subcat = rn.choice(['rolling chair', 'chair'])
    if ('wing chair' in subcat) : subcat = rn.choice(['easy couch', 'sofa couch', 'sofa'])
    if ('chaise' in subcat) : subcat = rn.choice(['cantilever chair', 'straight chair'])
    if ('lounger' in subcat) : subcat = rn.choice(['lounger', 'recliner', 'reclining chair'])
    if ('barcelona chair' in subcat) : subcat = rn.choice(['chair', 'padded chair'])
    if ('loveseat' in subcat) : subcat = 'love seat'
    if ('tete-a-tete' in subcat) : subcat = rn.choice(['double chair', 'lounge chair'])
    if ('vis-a-vis' in subcat) : subcat = rn.choice(['double chair', 'lounge chair'])
    if len(det) > 1 :
        if det[1][0] == 'game table' : 
            subcat = 'game table'
            if len(det) > 2 :
                subcat = det[2][0]

    desc = ' {} that is '.format(subcat)
    
    # item name to look for, what to call it, note qty
    items = [ 
              ['star leg base', 'set of star legs', False],     # 0
              ['pedestal base', '', False],                     # 1
              # ['central support', '', True],                    # 2
              ['caster', 'wheel', True],                             # 3
              ['foot', '', True],                               # 4
              ['leg', '', True],
              ['bar stretcher', 'leg connecting bar', True],
              ['runner', '', True],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              ['vertical side panel', '', True],
              ['vertical front panel', '', True],
              ['back panel', '', True],
              ['bottom panel', '', True],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', False],
              # ['', '', True],
              # ['', '', True],
              # ['', '', True],
              # ['', '', False],
              # ['', '', True],
              # ['', '', True],
              # ['', '', True],
               ['chair head', 'headrest', False],
               ['footrest', '', False],
              # ['', '', True],
              # ['', '', True],
              # ['', '', True],
              # ['', '', True],
             ]
    
    arms = [ ['chair arm', '', True],
             ['arm sofa style', 'sofa style', True],
             ['arm writing table', 'writing desk', True],
             # ['arm sofa style', 'sofa style', True],
             # ['arm sofa style', 'sofa style', True],
             ]
    arms_qty = [getSumDet(det, item[0]) for item in arms]
    num_arms = arms_qty[0]
    arms_desc = ''
    if (num_arms > 0) :
        arm_style = 'regular'
        for i, arm in enumerate(arms) :
            if arms_qty[i] > 0 : arm_style = arm[1]
        arm_style = '{} {}'.format(arm_style, 'chair arm')
        arms_desc = ' it has {}'.format(multiScriptor(arm_style, num_arms))
        
    backs = [ ['back frame vertical bar', 'vertical bar', True],
              ['back surface vertical bar', 'vertical bar', True],
              ['back frame horizontal bar', 'horizontal bar', True],
              ['back surface horizontal bar', 'horizontal bar', True],
              ['back connector', 'connector', True],
               ['back support', 'back support', True],
             ]
    backs_qty = [getSumDet(det, item[0]) for item in backs]
    backs_desc = ''    
    if (sum(backs_qty) > 0) :
        back_items = []
        for i, back in enumerate(backs) :
            if backs_qty[i] > 0 :
                back_items.append(multiScriptor(back[1], backs_qty[i]))
        backs_desc = ' it has a back frame made of {}'.format('and '.join(back_items))    
    
    items_qty = [getSumDet(det, item[0]) for item in items]    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc += ' it has {}.'.format(multiScriptor(items[i][1] if not items[i][1] == '' else items[i][0] , item_qty, 9))
            else :
                item_desc += ' it has a {}. '.format(items[i][1] if not items[i][1] == '' else items[i][0])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    csize = int(round((cx + cz)/2))
    # cwide = clamp(math.sqrt(round((dx / dy) * 13)), 0, len(chairSizes[0])-1)    
    shape_desc = chairSizes[0][csize]
    shape_desc += ' and ' + chairSizes[1][cy]
    shape_desc += ' and ' + chairSizes[2][cx]
    if cz >= 3 : shape_desc += ' and ' + chairSizes[3][cz]
    
    return joinPhrases([ desc + shape_desc, backs_desc, arms_desc, item_desc])

tableSizes = [['very small', 'small', 'average size', 'large', 'very large'], # size options 0
              ['very short', 'short', 'regular height', 'tall', 'very tall'],   # y axis options (heightwise)  1
              ['very thin',  'thin',  'regular', 'wide', 'very wide'],   # z axis options (widthwise)   2
              ['square', '', 'rectangular', 'long', 'skinny']] # square options (squarewise)  3
def dTable(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=0)
    if ('writing table' in row.subcats) : subcat = rn.choice(['writing table', 'secretary desk'])
    if ('coffee table' in subcat) : subcat = rn.choice(['coffee table', 'cocktail table'])
    if ('worktable' in subcat) : subcat = 'work table'
    if len(det) > 1 :
        if det[1][0] == 'picnic table' : subcat = 'picnic table'
        if det[1][0] == 'game table' :
            subcat = 'game table'
            if len(det) > 2 :
                subcat = det[2][0]

    desc = ' {} that is '.format(subcat)
    
    # item name to look for, what to call it, note qty
    items = [ ['glass', 'tabletop surface made of glass', False],
              # ['board', 'tabletop surface made of a board', False],
              ['bar stretcher', 'connecting bar', True],
              ['runner', 'runner', True],
              ['star leg base', 'set of star legs', False],
              ['pedestal base', '', False],
              ['central support', '', True],
             # ['', '', False],
             # ['', '', False],
             # ['', '', False],
              ['vertical side panel', '', True],
              ['vertical front panel', '', True],
              ['back panel', '', True],
              ['bottom panel', '', True],
             # ['', '', False],
             # ['', '', True],
             # ['', '', True],
              ['foot', '', True],
              ['leg', '', True],
              ['caster', 'wheel', True],
              ['drawer', '', True],
              # ['drawer base', '', False],
               ['handle', 'handle', False],
               ['cabinet door', 'cabinet', True],
               ['shelf', '', True],
              # ['drawer', '', True],
              # ['', '', True],
               ['circular stretcher', '', True],
              # ['', '', True],
               ['bench', '', True],
                ['pool ball', '', False],
               # ['bench', '', True],
               # ['bench', '', True],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc += ' it has {}.'.format(multiScriptor(items[i][1] if not items[i][1] == '' else items[i][0] , item_qty, 9))
            else :
                item_desc += ' it has a {}. '.format(items[i][1] if not items[i][1] == '' else items[i][0])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = tableSizes[1][cy]    
    if (not csq == 1) : 
        shape_desc += ' and ' + tableSizes[3][csq]
    csize = int(round((cx + cz)/2))
    shape_desc += ' and ' + tableSizes[0][csize]
    return joinPhrases([ desc + shape_desc, item_desc])

bedSizes = [['very skinny', 'skinny', 'average width', 'wide', 'very wide'], # bed width (x)
            ['very short', 'short', 'average length', 'long', 'very long'],   # length size (z)
            ['very thin', 'thin', 'average thickness', 'thick', 'very thick'],   # y axis options (heightwise) for regular beds
            ['super short', 'very short', 'short', 'average height', 'tall']]   # y axis options (heightwise) for bunks
def dBed(row) :
    det = detToArr(row.details)
    subcat = 'bed'
    subcat = rarestCat(row, min_cutoff=71)   
    if 'platform' in row.subcats : subcat = 'platform bed'
    if 'headboard' in row.subcats : subcat = 'headboard bed'
    if 'hammock' in row.subcats : subcat = 'hammock'
    desc = ' {} that is '.format(subcat)
    
    # item name to look for, what to call it, note qty
    items = [['headboard', 'headboard', False],
             ['ladder', 'ladder', False],
             ['pillow', 'pillow', True],
             ['foot', 'foot', True],
             ['bed post', 'post', True],
             ['bed unit', 'bed', True],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc += ' it has {}.'.format(multiScriptor(items[i][1], item_qty, 6))
            else :
                item_desc += ' it has a {}. '.format(items[i][1])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = ''
    if ('bunk' in row.subcats) :
        shape_desc = bedSizes[3][cy]
        shape_desc += ' and ' + bedSizes[1][cz]
        shape_desc += ' and ' + bedSizes[0][cx]   
    else :
        shape_desc = bedSizes[2][cy]
        shape_desc += ' and ' + bedSizes[1][cz]
        shape_desc += ' and ' + bedSizes[0][cx]  
    
    return joinPhrases([ desc + shape_desc, item_desc])

lampSizes = [['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
             ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dLamp(row) :
    det = detToArr(row.details)
    subcat = 'lamp'
    if   ('table' in row.subcats) : subcat = 'table lamp'
    elif ('table' in row.subcats) : subcat = 'table lamp'
    elif ('floor' in row.subcats) : subcat = 'floor lamp'
    ceil_sub = ''
    if len(det) > 1 :
        if det[1][0] == 'table or floor lamp' and subcat=='lamp' : subcat = rn.choice(['table lamp', 'floor lamp'])
        if det[1][0] == 'street lamp' : subcat = 'street lamp'
        if det[1][0] == 'wall lamp' : subcat = 'wall lamp'
        if det[1][0] == 'ceiling lamp' : subcat = 'ceiling lamp'
        if len(det) > 2 :
            ceil_sub = det[2][0] if 'ceiling' in subcat else ''
            if 'ceiling' in subcat : subcat = rn.choice([subcat, ceil_sub])

    desc = ' {} that is '.format(subcat)    
    # item name to look for, what to call it, note qty
    items = [['lamp arm curved bar', 'curved bar', False],
             ['lamp arm straight bar', 'straight bar', False],
             ['lamp cover', 'cover', False],
             ['power cord', 'power cord', False],
             ['lamp body jointed', 'jointed body', False],
             ['lamp finial', 'decorative', False],
             ['lamp body solid', 'solid body', False],
             ['lamp holistic base', 'base', False],
             ['chain', 'chain', True],
             ['foot', 'foot', True],
             ['leg', 'leg', True],
             ['lamp pole', 'pole', True],
             ['lamp arm', 'arm', True],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc += ' it has {}.'.format(multiScriptor(items[i][1], item_qty, 6))
            else :
                item_desc += ' it has a {}. '.format(items[i][1])
    
    descriptors = []
    if getSumDet(det, 'lamp finial') > 0 : descriptors.append('decorative')
    if getSumDet(det, 'lamp body jointed') > 0 : descriptors.append('jointed')
    if getSumDet(det, 'lamp body solid') > 0 : descriptors.append('solid')
    descriptor = rn.choice(descriptors) if len(descriptors) > 0 else ''
    
    heads = max(getSumDet(det, 'lamp head'), getSumDet(det, 'lamp unit'))
    heads_desc =' it has {}'.format(multiScriptor('head', heads, 6))
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = ''
    csize = int(round((cx + cz)/2))
    shape_desc += lampSizes[0][csize]
    shape_desc += ' and ' + lampSizes[1][cy]
        
    return joinPhrases([ descriptor + desc + shape_desc, heads_desc, item_desc])

faucetSize = ['very short', 'short', 'regular', 'tall', 'very tall']   # y axis options (heightwise)  1
def dFaucet(row) :
    det = detToArr(row.details)
    subcat = 'faucet'
    if len(det) > 1 :
        subcat = 'shower faucet' if det[1][0]=='shower faucet' else 'faucet'
    desc = ' {} that is '.format(subcat)
    switches = getSumDet(det, 'switch')
    spouts = getSumDet(det, 'spout')
    vsups = getSumDet(det, 'vertical support')
    hsups = getSumDet(det, 'horizontal support')
    parts = getSumDet(det, 'frame') 
    
    switch_desc = ' it has ' + multiScriptor('switch', switches, 6)
    spout_desc = ' it has ' + multiScriptor('spout', spouts, 6)
    vsup_desc = ' it has ' + multiScriptor('vertical bar', vsups, 5) if vsups > 1 else ''
    hsup_desc = ' it has ' + multiScriptor('horizontal bar', hsups, 5) if hsups > 1 else ''
    part_desc = ' it is made of ' + multiScriptor('part', parts, 6) if parts > 1 else ''   
    
    shape_desc = faucetSize[row.cy]
    return joinPhrases([ desc + shape_desc, part_desc, switch_desc, spout_desc, vsup_desc, hsup_desc])

clockSizes = [['very wide', 'wide', 'regular width', 'long', 'very long'],  # width
               ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size
               ['very thin', 'thin', 'average thickness', 'thick', 'very thick'],   # thickness options
               ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dClock(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=10)
    desc = ' {} that is '.format(subcat)
    chains = getSumDet(det, 'chain')
    boxs = getSumDet(det, 'box')
    screens = getSumDet(det, 'screen')
    frames = getSumDet(det, 'frame')
    bases = getSumDet(det, 'base') + getSumDet(det, 'base surface') 
    pendulums = 0 if 'pendulum clock' in subcat else getSumDet(det, 'pendulum clock')
    foots = getSumDet(det, 'foot')
    
    chain_desc = ' it has ' + multiScriptor('chain', chains, 10) if chains > 0 else ''
    box_desc = ' it has a box ' if boxs > 0 else ''
    pendulum_desc = ' it is a pendulum clock ' if pendulums > 0 else ''
    screen_desc = ' it has a screen ' if screens > 0 else ''
    frame_desc = ' it has a frame ' if frames > 0 else ''
    base_desc = ' it has a base ' if bases > 0 else ''
    foot_desc = ' it has ' + multiScriptor('foot', foots) if foots > 0 else ''
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = ''
    if ('grand' in row.subcats or 'pendul' in row.subcats) :
        shape_desc = '' + clockSizes[3][cy]
        clong = clamp(round((cy / (cx+.0001)) * 2), 0, len(clockSizes[0])-1)
        shape_desc += ' and ' + clockSizes[0][clong]
    elif abs(dx - dy)>.05 :
        shape_desc = clockSizes[3][cy]
        clong = clamp(round((cy / (cx+.0001)) * 2), 0, len(clockSizes[0])-1)
        shape_desc += ' and ' + clockSizes[0][clong]
    else :
        shape_desc = clockSizes[2][cz]
        cdia = int(round((cx + cy)/2))
        shape_desc += ' and ' + clockSizes[1][cdia]
        
    return joinPhrases([ desc + shape_desc, pendulum_desc, chain_desc, box_desc, base_desc, frame_desc, screen_desc, foot_desc])

bottleSizes = [['very long', 'long', 'regular length', 'wide', 'very wide'],  # width
               ['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
               ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dBottle(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=10)
    desc = ' {} that is '.format(subcat)
    mouths = getSumDet(det, 'mouth')
    necks = getSumDet(det, 'neck')
    handles = getSumDet(det, 'handle')
    lids = getSumDet(det, 'lid') + getSumDet(det, 'closure')
    
    mouth_desc = ' it has a mouth ' if mouths > 0 else ''
    neck_desc = ' it has a neck ' if necks > 0 else ''
    lid_desc = ' it has a lid ' if lids > 0 else ''
    handles_desc = ' it has ' + multiScriptor('handle', handles, 4) if handles > 0 else ''
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = '' + bottleSizes[2][cy]
    if abs(dx - dz)>.05 :
        clong = clamp(round((cz / (cx+.0001)) * 2), 0, len(bottleSizes[0])-1)
        shape_desc += ' and ' + bottleSizes[0][clong]
    else :
        cdia = int(round((cx + cz)/2))
        shape_desc += ' and ' + bottleSizes[1][cdia]
        
    return joinPhrases([ desc + shape_desc, lid_desc, mouth_desc, neck_desc, handles_desc])

vaseSizes = [['very long', 'long', 'regular length', 'wide', 'very wide'],  # width
            ['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dVase(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row)
    desc = ' {} that is '.format(subcat)
    plants = getSumDet(det, 'plant')
    stuff = getSumDet(det, 'liquid or soil')
    bases = getSumDet(det, 'base')
    foots = getSumDet(det, 'foot')
    lids = getSumDet(det, 'lid')
    
    foot_desc = ' it has ' + multiScriptor('foot', foots) if foots > 0 else ''
    lid_desc = ' it has a lid ' if lids > 0 else ''
    base_desc = ' it has a base ' if bases > 0 else ''
    
    contains_desc = ''
    if plants >  0 and stuff >  0 : contains_desc = ' it contains a plant and some soil'
    if plants == 0 and stuff >  0 : contains_desc = ' it is full'
    if plants >  0 and stuff == 0 : contains_desc = ' it contains a plant'
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = '' + vaseSizes[2][cy]
    if abs(dx - dz)>.05 :
        clong = clamp(round((cz / (cx+.0001)) * 2), 0, len(vaseSizes[0])-1)
        shape_desc += ' and ' + vaseSizes[0][clong]
    else :
        cdia = int(round((cx + cz)/2))
        shape_desc += ' and ' + vaseSizes[1][cdia]
    
    return joinPhrases([ desc + shape_desc, contains_desc, lid_desc, base_desc, foot_desc])

laptopSizes = [['very skinny', 'skinny', 'average width', 'wide', 'very wide']]
def dLaptop(row) :
    det = detToArr(row.details)
    desc = ' {} that is '.format('laptop')
    tpad = getSumDet(det, 'touchpad')
    kboard = getSumDet(det, 'keyboard')
    screens = getSumDet(det, 'screen')
    item_desc = ''
    if tpad > 0 and kboard > 0 : item_desc = ' it has a screen, a touchpad, and a keyboard'
    if tpad == 0 and kboard > 0 : item_desc = ' it has a screen and a keyboard but no touchpad'
    if tpad > 0 and kboard == 0 : item_desc = ' it has a screen and a touchpad but no keyboard'
    if screens == 0 : item_desc = item_desc + ' it is closed'
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq    
    
    shape_desc = laptopSizes[0][cx]
    
    return joinPhrases([desc + shape_desc, item_desc])

mugSizes = [['very small handle', 'small handle', 'average handle', 'large handle', 'very large handle'], # handle size
            ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dMug(row) :
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=4)
    desc = ' {} that is '.format(subcat)
    contains = getSumDet(det, 'containing things')
    handles = getSumDet(det, 'handle')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    shape_desc = ''
    cdia = cz
    chandle = clamp(round((cz / (cx+.0001)) * 2), 0, len(mugSizes[0])-1)
    shape_desc += mugSizes[1][cdia]
    shape_desc += ' and ' + mugSizes[2][cy]
    if (handles > 0) : shape_desc += ' . it has a ' + mugSizes[0][chandle]
    
    handles_desc = '' if handles == 1 else ' it has ' + multiScriptor('handle', handles, 4)
    contains_desc = ' it is empty ' if contains==0 else ' it is full '
    return joinPhrases([ desc + shape_desc, contains_desc, handles_desc])

bowlSizes = [['round', 'oval'], # roundness
            ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size 
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dBowl(row) :
    det = detToArr(row.details)
    desc = ' bowl that is '
    contains = getSumDet(det, 'containing things')
    base = getSumDet(det, 'bottom')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq    
    
    shape_desc = (bowlSizes[0][0] if abs(dx - dz)<.005 else bowlSizes[0][1])
    cdia = int(round((cx + cz)/2))
    shape_desc += ' and ' + bowlSizes[1][cdia]
    shape_desc += ' and ' + bowlSizes[2][cy]
    
    contains_desc = ' it is empty ' if contains==0 else ' it is full '
    base_desc = '' if base==0 else ' it has a base '    
    return '.'.join([ desc + shape_desc, contains_desc, base_desc])

def clamp(value, minv, maxv) :
    return int(max(min(value, maxv), minv))

def getFreq(name) :
    if name not in corpdict :
        return 0
    return corpdict[name]

def rarestCat(row, min_cutoff=10, exclude='NONE') :
    subcats = row.subcats.split(',')
    cat_freq = [getFreq(cat) if int(getFreq(cat)) >= min_cutoff or cat==exclude else 999999 for cat in subcats]
    if min(cat_freq) >= 999999 : return row.cattext.lower()
    return subcats[np.argmin(cat_freq)]

def joinPhrases(phrases) :
    phrases = [p for p in phrases if not p=='']
    return '. '.join(phrases)

def multiScriptor(thing, qty, many_thresh=7) :
    if qty == 0 : return 'no {} '.format(inflect.plural(thing))
    for i in range(1,many_thresh+1) :
        if qty == i : return '{} {} '.format(inflect.number_to_words(i), inflect.plural(thing, count=i))     
    return 'many {} '.format(inflect.plural(thing))

def getSumDet(det, name) :
    count = sum([int(item[3]) for item in det if item[0]==name])
    return count

def printDetArr(det, max_level=10) :
    for item in det :
        name, level, children, quantity = (item[0]).lower(), int(item[1]), int(item[2]), int(item[3])
        if (level > max_level) : continue
        freq = corpdict[name]
        print('{:1d} :{:2d} :{:2d} : {:5d} : {}{}'.format(level, children, quantity, freq, '  '*int(level+1), name))

# self.name, level, children, quantity
def detToArr(details) :
    dets = [[item.strip() for item in d.split(',')] for d in (' '+details).split('|')[:-1]]
    dets = np.array(dets)
    return dets

def printShapeClasses(row) :
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    template = '   {}  {}   {:13}  {:.3f}'
    print(template.format('X ', cx, sizes[0][cx], row.dx))
    print(template.format('Y ', cy, sizes[0][cy], row.dy))
    print(template.format('Z ', cz, sizes[0][cz], row.dz))
    print(template.format('SQ', csq, sizes[0][csq], row.dsq))
    
def getShapeDesc(row) :
    cx = row.cx
    cy = row.cy
    cz = row.cz
    csq = row.csq      
    descriptors = [sizes[0][cx], sizes[1][cy], sizes[2][cz], sizes[3][csq]]
    extreme = np.array((cx,cy,cz,csq))
    eid = np.argmax(abs(extreme-2))
    ext_descriptor = sizes[eid][extreme[eid]]
    return descriptors, ext_descriptor

def listSubcats(catid) :
    allsubcats = []
    dffilter = dfdesc[dfdesc.cattext==cats_to_load[catid]]
    for i, row in dffilter.iterrows():
        allsubcats.extend(row.subcats.split(','))        
    for item in sorted(set(allsubcats)) :
        freq = 0
        try : freq = corpdict[item]
        except : freq = 'N/A'
        print('{:5}   {}'.format(freq, item))

def deleteSentences(row, rate=.2) :
    desc = row.desc.values[0]
    s = desc.split('.')[:-1]
    for sentence in s[1:] :
        if rn.uniform(0,1) < rate :
            s.remove(sentence)
    
    result = ' . '.join(s) + ' . '
    row.desc = result
    return row        

def shuffleSentences(row) :
    desc = row.desc.values[0]
    s = desc.split('.')[:-1]
    rn.shuffle(s)
    result = ' . '.join(s) + ' . '
    row.desc = result
    return row

fix_replacements = [['.', ' . '],
                    [',', ' , '],
                    ['  ', ' '],
                    ['   ', ' ']]
def fixPuncs(row) :    
    desc = row.desc.values[0]
    for rp in fix_replacements :
        desc = desc.replace(rp[0], rp[1])  
    row.desc = desc
    return row

def buildCorpus(dfdesc):
    global corpdict
    corpus = []
    for index, row in tqdm(dfdesc.iterrows(), total=len(dfdesc)):   # .iloc[:1000]
        try: 
            corpus.extend([str(row.cattext).lower()])
            corpus.extend([word.lower() for word in row.subcats.split(',')])
            dets = [item.split(',')[0].strip().lower() for item in row.details.split('|')[:-1]]
            corpus.extend(dets)        
        except : pass
        
    phrases = Counter(corpus).keys()
    counts =  Counter(corpus).values()
    corpdict = {k:v for k,v in zip(phrases, counts)}
    
    for p,c in zip(phrases, counts) :
        print('{:25}    {}'.format(p, c))
        
    corpdict = {k: v for k, v in sorted(corpdict.items(), key=lambda item: item[1])}    
    return corpdict