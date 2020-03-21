'''
This file is how the descriptions are generated. It must be ran after partnetmeta.py as it uses the data from that file.

It is broken up into 5 subparts:
    1. Loading in spacy and exploring it's capabilities
    2. Loading in the data from the partnetmeta output
    3. Shape class that takes in various information about an object and stores it so that it can be sampled for randomized descriptions later.
    4. Shape description methods that gather the information for the shape class from the dfmeta file
    5. Generate the descriptions, save them to file, and inspect the descriptions for quality.
'''

#%% Imports
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pandas as pd
import random
from tqdm import tqdm
import random as rn
import inflect
import spacy
from nltk.corpus import wordnet as wn
import math

import utils as ut
import configs as cf






np.set_printoptions(precision=3, suppress=True)

#%% Setup text processing and load in dfmeta
inflect = inflect.engine()
vocab = set()
corpdict = {}

cats_to_load = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']
catids_to_load = [4379243,3001627,3636649,3325088,3046257,2876657,3593526,3642806,2818832,3797390,2880940]

dfmeta = pd.read_csv(cf.META_DATA_CSV)
dfdesc = dfmeta[dfmeta.details.notnull()]

#%% Setup spacy and some related methods
def get_embeddings(vocab):
        max_rank = max(lex.rank for lex in vocab if lex.has_vector)
        vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
        for lex in vocab:
            if lex.has_vector:
                vectors[lex.rank] = lex.vector
        return vectors

def most_similar(word):
   queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15 and w.has_vector]
   by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
   return by_similarity[:10]

def closestVect(vect, num_to_get=10):
   queries = [w for w in nlp.vocab if w.prob >= -15 and w.has_vector]
   by_similarity = sorted(queries, key=lambda w: cosine_similarity(w.vector.reshape(1,-1), vect.reshape(1,-1)), reverse=True)
   return by_similarity[:num_to_get]

nlp = spacy.load('en_core_web_md')
embeddings = get_embeddings(nlp.vocab)

#%% Seeing what the closest vector is in vector space
cvs = closestVect(nlp("high")[0].vector)
for w in cvs : print(w.text)

#%% Using the builtin spacy method to do the same as above for comparison
syns = most_similar(nlp.vocab['high'])
for w in syns:
    print(w.text, w.cluster)

#%% Part of speech tagging testing
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

text = nlp('a large thing')
for token in text:
    print('{:12}  {:5}  {:6}  {}'.format(str(token), str(token.tag_), str(token.pos_), str(spacy.explain(token.tag_))))

print('\n\n')
for token in text:
    print('{:10}  {:8}  {:8}  {:8}  {:8}  {:15}  {}'.format(token.text, token.dep_, token.tag_, token.pos_, token.head.text,
              str([item for item in token.conjuncts]), [child for child in token.children]))

for item in text :
    print('{:10}  {}'.format(item.text, [syn.name().split('.')[0] for syn in wn.synsets(item.text, pos=wn.ADJ)[:10] if not syn.name().split('.')[0] == item.text]))
wn.synset('tall.a.01').lemmas()[0].antonyms()

#%% Shape class that can take in a set of standardized items and generate random descriptions based on those
class shape() :
    replacements = {
    '   ' : ' ',
    '  ' : ' ',
    ' and .' : '.',
    ' or .' : '.',
    ' . ' : '. ',
    ' .' : '.', 
    ' with.' : '.', 
    ' that is.' : '.' }
    
    synonyms = {'long' : 'deep great lengthy',
    'skinny' : 'lean slim',
    'regular' : 'ordinary traditional standard usual common typical average',
    'average' :'regular ordinary traditional standard usual common typical',
    'short' : 'low flat',
    'tall' : 'high',
    'thin' : 'lean narrow skinny slim',
    'thick' : 'broad wide',
    'wide' : 'broad spacious',
    'narrow' : 'slender slim thin',
    'square' : 'round ',
    'rectangular' : 'oval boxy',
    'elongated' : 'stretched extended',
    'small' : 'tiny little',
    'very small' : 'miniature teeny',
    'large' : 'big',
    'very large' : 'enourmous gigantic giant huge massive roomy',
    'very' : 'super extremely quite immensely',
    'chair arms' : 'armrests arms',
    ' bar' : [' support'],
    'tabletop surface made of glass' : ['glass surface', 'surface made of glass', 'glass tabletop'] }

    combiners = [' that has ', ' made of ', ' with ']
    desc_prefixes = [' that is ', ' and is ']

    def __init__(self, mcat='', scat='', sdescriptors=[], gdescriptors=[], spnames=[], negatives=[], contains=[]) :
        self.mcat, self.scat, self.sdescriptors, self.gdescriptors, self.spnames, self.negatives, self.contains = mcat, scat, sdescriptors, gdescriptors, spnames, negatives, contains
    
    def getComplexity(self) :
        return math.ceil(math.log(sum([len(self.sdescriptors), len(self.gdescriptors), len(self.spnames), len(self.contains), len(self.negatives)])) * 3)
    
    def __repr__(self) :
        output = 'Main cat: {:10} Subcat: {:12} Complexity: {}'.format(self.mcat, self.scat, self.getComplexity())
        output += '\nShape Descriptors:\n'       
        for item in self.sdescriptors :
            output += '   ' + item + '\n'
        
        if len(self.gdescriptors) > 0 :
            output += '\nGeneral Descriptors:\n'       
            for item in self.gdescriptors :
                output += '   ' + item + '\n'
        else : output += 'No general descriptors\n'
        
        if len(self.spnames) > 0 :
            output += '\nSub part names:\n'       
            for item in self.spnames :
                output += '   ' + item + '\n'
        else : output += 'No sub parts\n'
        
        if len(self.negatives) > 0 :
            output += '\nNegatives :\n'       
            for item in self.negatives:
                output += '   ' + item + '\n'
        else : output += 'No negatives\n'
    
        if len(self.contains) > 0 :
            output += '\nContains:\n'       
            for item in self.contains:
                output += '   ' + item + '\n'
        else : output += 'Contains nothing\n'
            
        return output
    
    def getDesc(self) :
        scat, sdescriptors, gdescriptors, spnames, negatives, contains = self.scat, self.sdescriptors.copy(), self.gdescriptors.copy(), self.spnames.copy(), self.negatives.copy(), self.contains.copy()
        sdescriptors, gdescriptors, spnames, negatives, contains = self.ablateList(sdescriptors, rate=0.5), self.ablateList(gdescriptors, rate=0.3), self.ablateList(spnames, rate=0.6), self.ablateList(negatives, rate=0.75), self.ablateList(contains, rate=0.5)
        descriptors = sdescriptors + gdescriptors
        
        descs_pre = self.multiFormat(self.subSampleRemove(descriptors, 3)) if len(descriptors) > 0 else ''        
        if (len(contains) > 0 and rn.random() > 0.3) :
            parts_post = self.multiFormat(self.subSampleRemove(contains, at_most=2), prefix=' containing ', ensure_ending_and=True) if len(contains) > 0 else ''    
        else :
            parts_post =self.multiFormat(self.subSampleRemove(spnames, 3), prefix = rn.choice(self.combiners), ensure_ending_and=True ) if len(spnames) > 0 else ''                                
        descs_post= self.multiFormat(self.subSampleRemove(descriptors, 3), prefix = rn.choice(self.desc_prefixes), ensure_ending_and=True) if len(descriptors) > 0 else ''
        posts = [parts_post, descs_post]
        rn.shuffle(posts)
        output = 'a {}{}{}{}. '.format(descs_pre, scat, posts[0], posts[1])
        
        descs_r2 = self.multiFormat(descriptors[:5], prefix = ' it is ', ensure_ending_and=True) if len(descriptors) > 0 else ''
        parts_r2 = self.multiFormat(spnames[:5], prefix = ' it has ', ensure_ending_and=True) if len(spnames) > 0 else ''
        conts_r2 = self.multiFormat(contains, prefix = ' it contains ', ensure_ending_and=True) if len(contains) > 0 else ''
        
        round2 = []
        if len(descs_r2) > 0 : round2.append(descs_r2 + '. ')
        if len(parts_r2) > 0 : round2.append(parts_r2 + '. ')
        if len(conts_r2) > 0 : round2.append(conts_r2 + '. ')
        
        negs_descs = ''
        if len(negatives) > 0 :            
            negs_descs += 'it does not have '
            for thing in negatives :
                if 'a ' in thing :
                    negs_descs += '{} or '.format(thing)
                else :
                    negs_descs += 'any {} or '.format(inflect.plural(thing))
            negs_descs += '. '    
        if len(negs_descs) > 0 : round2.append(negs_descs) 
        
        rn.shuffle(round2)
        for thing in round2 :
            output += thing
        
        output = self.synReplace(output, chance=0.25)
        return self.fixDesc(output)

    'a {}desk'
    'a desk with {}'
    multi_1templates = ['{}{} ']
    multi_2templates = ['{}{} {} ',
                        '{}{} and {} ']
    multi_3templates = ['{}{} {} {} ',
                        '{}{} and {} and {} ',                        
                        '{}{} {} and {} ',]
    multi_4templates = ['{}{} {} {} {} ',
                        '{}{} and {} and {} and {} ',
                        '{}{} {} {} and {} ',]
    multi_5templates = ['{}{} {} {} {} {} ',
                        '{}{} and {} and {} and {} and {} ',
                        '{}{} {} {} {} and {} ' ]
    
    def multiFormat(self, items, prefix='', ensure_ending_and = False) :
        usable_items = items.copy()
        if len(items) > 5 :
            usable_items = rn.sample(items, 5)
        
        min_template_index = 1 if ensure_ending_and else 0            
        if len(usable_items) == 0 :
            return ''
        elif len(usable_items) == 1 :
            return self.multi_1templates[0].format(prefix, usable_items[0])
        elif len(usable_items) == 2 :
            return rn.choice(self.multi_2templates[min_template_index:]).format(prefix, usable_items[0], usable_items[1])
        elif len(usable_items) == 3 :
            return rn.choice(self.multi_3templates[min_template_index:]).format(prefix, usable_items[0], usable_items[1], usable_items[2])
        elif len(usable_items) == 4 :
            return rn.choice(self.multi_4templates[min_template_index:]).format(prefix, usable_items[0], usable_items[1], usable_items[2], usable_items[3])
        elif len(usable_items) == 5 :
            return rn.choice(self.multi_5templates[min_template_index:]).format(prefix, usable_items[0], usable_items[1], usable_items[2], usable_items[3], usable_items[4])

    def ablateList(self, items, rate=0.3) :
        if len(items) == 0 :
            return []
        ablated = [item for item in items if rn.random() > rate]
        rn.shuffle(ablated)
        return ablated
        
    def subSampleRemove(self, items, at_most) :
        samples = rn.sample(items, min(at_most, rn.randint(0,len(items))))
        for thing in samples : items.remove(thing)
        return samples

    def synReplace(self, desc, chance=0.3) :
        for syn in self.synonyms.keys() :
            if rn.random() < chance :
                if type(self.synonyms[syn]) == list :
                    desc = desc.replace(syn, rn.choice(self.synonyms[syn]))
                else :
                    desc = desc.replace(syn, rn.choice(self.synonyms[syn].split()))
        return desc    

    def fixDesc(self, desc) :
        desc = desc.lower().strip()
        for fix in self.replacements.keys() :
            desc = desc.replace(fix, self.replacements[fix])
        return desc

'''
Table of category info for reference :
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

High level notes for how the descriptions were generated for each subcategory:
    
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

'''

#%% Shape description methods
'''
These methods take in a row of the dfmeta file and populate a standardized set of items that the shape class
then uses to generate randomized descriptions. It may use any subset of the available items.

From dfmeta these methods get :
    1. subcat       : The subcategory. For example, a chair may specfically be a swivel chair.
    2. sdescriptors : Shape descriptors. Changes for each category but could be like tall for chairs or thick for clocks. Every category method is preceded by a ...sizes lists which defines which axes correspond to which shape descriptions.
    3. gdescriptors : General descriptors. For example, could be a decorative or jointed lamp.
    4. spnames      : Sub part names. A listing of all interesting sub parts contained in the tree. Interesting defined as relatively rare.
    5. negatives    : Things that this object doesn't have that other similar objects often do.
    6. contains     : If it contains anything. Only relevant for a few categories like vase which could contain a plant or soil or a bowl that could be full or empty.
'''

# The main method that selects which row to specific method to use
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
    else : return None

chairSizes = [['very small', 'small', 'average size', 'large', 'very large'], # size options 0
              ['very short', 'short', 'regular height', 'tall', 'very tall'],   # y axis options (heightwise)  1
              ['very thin',  'thin',  'regular width', 'wide', 'very wide'],   # x axis options (widthwise)   2
              ['', '', '', 'long', 'very long']] # length options (z)  3
def dChair(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    
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
   
    # item name to look for, what to call it, note qty
    items = [ 
              ['star leg base', 'set of star legs', False, False],     # 0
              ['pedestal base', '', False, False],                     # 1
              # ['central support', '', True, False],                    # 2
              ['caster', 'wheel', True, False],                             # 3
              ['foot', '', True, False],                               # 4
              ['leg', '', True, False],
              ['bar stretcher', 'leg connecting bar', True, False],
              ['runner', '', True, False],
              # ['', '', False, False],
              ['vertical side panel', '', True, False],
              ['vertical front panel', '', True, False],
              ['back panel', '', True, False],
              ['bottom panel', '', True, False],
              # ['', '', True, False],
               ['chair head', 'headrest', False, False],
               ['footrest', '', False, False]
             ]
    
    arms = [ ['chair arm', '', True],
             ['arm sofa style', 'sofa style', True],
             ['arm writing table', 'writing desk', True]
             ]
    arms_qty = [getSumDet(det, item[0]) for item in arms]
    num_arms = arms_qty[0]
    arms_desc = ''
    if (num_arms > 0) :
        arm_style = 'regular'
        for i, arm in enumerate(arms) :
            if arms_qty[i] > 0 : arm_style = arm[1]
        arm_style = '{}{}'.format(arm_style+' ' if len(arm_style)>0 else '', 'chair arm')
        arms_desc = '{}'.format(multiScriptor(arm_style, num_arms))
        spnames.append(arms_desc)
    else :
        negatives.append('chair arm')    
        
    backs = [ ['back frame vertical bar', 'vertical bar', True],
              ['back surface vertical bar', 'vertical bar', True],
              ['back frame horizontal bar', 'horizontal bar', True],
              ['back surface horizontal bar', 'horizontal bar', True],
              ['back connector', 'back connector', True],
              ['back support', 'back support', True],
             ]
    backs_qty = [getSumDet(det, item[0]) for item in backs]
    backs_desc = ''    
    if (sum(backs_qty) > 0) :
        back_items = []
        for i, back in enumerate(backs) :
            if backs_qty[i] > 0 :
                back_item_desc = multiScriptor(back[1], backs_qty[i])
                back_items.append(back_item_desc)
                spnames.append(back_item_desc)
        backs_desc = ' it has a back frame made of {}'.format('and '.join(back_items))
    
    items_qty = [getSumDet(det, item[0]) for item in items]    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc = '{}'.format(multiScriptor(items[i][1] if not items[i][1] == '' else items[i][0] , item_qty, 9))
            else :
                item_desc = 'a {}'.format(items[i][1] if not items[i][1] == '' else items[i][0])
            spnames.append(item_desc)
        elif items[i][3] :
            negatives.append(items[i][1] if items[i][2] else 'a ' + items[i][1])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    csize = int(round((cx + cz)/2)) 
    sdescriptors.append(chairSizes[0][csize])
    sdescriptors.append(chairSizes[1][cy])
    sdescriptors.append(chairSizes[2][cx])
    if cz >= 3 : sdescriptors.append(chairSizes[3][cz])    
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

tableSizes = [['very small', 'small', 'average size', 'large', 'very large'], # size options 0
              ['very short', 'short', 'regular height', 'tall', 'very tall'],   # y axis options (heightwise)  1
              ['very thin',  'thin',  'regular', 'wide', 'very wide'],   # z axis options (widthwise)   2
              ['square', '', 'rectangular', 'long', 'skinny']] # square options (squarewise)  3
def dTable(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
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
    
    # item name to look for, what to call it, note qty
    items = [ ['glass', 'tabletop surface made of glass', False, False],
              # ['board', 'tabletop surface made of a board', False, False],
              ['bar stretcher', 'connecting bar', True, False],
              ['runner', 'runner', True, False],
              ['star leg base', 'set of star legs', False, False],
              ['pedestal base', '', False, False],
              ['central support', '', True, False],
             # ['', '', False, False],
              ['vertical side panel', '', True, False],
              ['vertical front panel', '', True, False],
              ['back panel', '', True, False],
              ['bottom panel', '', True, False],
             # ['', '', False, False],
              ['foot', '', True, False],
              ['leg', '', True, False],
              ['caster', 'wheel', True, False],
              ['drawer', '', True, False],
              # ['drawer base', '', False, False],
               ['handle', 'handle', False, False],
               ['cabinet door', 'cabinet', True, False],
               ['shelf', '', True, False],
              # ['', '', True, False],
               ['circular stretcher', '', True, False],
              # ['', '', True, False],
               ['bench', '', True, False],
                ['pool ball', '', False, False],
               # ['bench', '', True, False],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc = '{}'.format(multiScriptor(items[i][1] if not items[i][1] == '' else items[i][0] , item_qty, 9))
            else :
                item_desc = 'a {}'.format(items[i][1] if not items[i][1] == '' else items[i][0])
            spnames.append(item_desc)
        elif items[i][3] :
            negatives.append(items[i][1] if items[i][2] else 'a ' + items[i][1])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    sdescriptors.append(tableSizes[1][cy])
    if (not csq == 1) : 
        sdescriptors.append(tableSizes[3][csq])
    csize = int(round((cx + cz)/2))
    sdescriptors.append(tableSizes[0][csize])
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

bedSizes = [['very skinny', 'skinny', 'average width', 'wide', 'very wide'], # bed width (x)
            ['very short', 'short', 'average length', 'long', 'very long'],   # length size (z)
            ['very thin', 'thin', 'average thickness', 'thick', 'very thick'],   # y axis options (heightwise) for regular beds
            ['super short', 'very short', 'short', 'average height', 'tall']]   # y axis options (heightwise) for bunks
def dBed(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = 'bed'
    subcat = rarestCat(row, min_cutoff=71)   
    if 'platform' in row.subcats : subcat = 'platform bed'
    if 'headboard' in row.subcats : subcat = 'headboard bed'
    if 'hammock' in row.subcats : subcat = 'hammock'
    
    # item name to look for, what to call it, note qty
    items = [['headboard', 'headboard', False, True],
             ['ladder', 'ladder', False, False],
             ['pillow', 'pillow', True, True],
             ['foot', 'foot', True, True],
             ['bed post', 'post', True, False],
             ['bed unit', 'bed', True, False],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    item_desc = ''
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc = '{}'.format(multiScriptor(items[i][1], item_qty, 6))
            else :
                item_desc = 'a {}'.format(items[i][1])
            spnames.append(item_desc)
        elif items[i][3] :
            negatives.append(items[i][1] if items[i][2] else 'a ' + items[i][1])
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    if ('bunk' in row.subcats) :
        sdescriptors.append(bedSizes[3][cy])
        sdescriptors.append(bedSizes[1][cz])
        sdescriptors.append(bedSizes[0][cx])
    else :
        sdescriptors.append(bedSizes[2][cy])
        sdescriptors.append(bedSizes[1][cz])
        sdescriptors.append(bedSizes[0][cx]  )
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

lampSizes = [['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
             ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dLamp(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
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
  
    # item name to look for, what to call it, note qty
    items = [['lamp arm curved bar', 'curved bar', False, False],
             ['lamp arm straight bar', 'straight bar', False, False],
             ['lamp cover', 'cover', False, True],
             ['power cord', 'power cord', False, False],
             ['lamp body jointed', 'jointed body', False, False],
             ['lamp body solid', 'solid body', False, False],
             ['lamp holistic base', 'base', False, False],
             ['chain', 'chain', True, False],
             ['foot', 'foot', True, True],
             ['leg', 'leg', True, True],
             ['lamp pole', 'pole', True, False],
             ['lamp arm', 'arm', True, False],
             ]
    items_qty = [getSumDet(det, item[0]) for item in items]
    
    for i, item_qty in enumerate(items_qty) :
        if item_qty > 0 :
            if items[i][2] :
                item_desc = '{}'.format(multiScriptor(items[i][1], item_qty, 6))
            else :
                item_desc = 'a {}'.format(items[i][1])
            spnames.append(item_desc)
        elif items[i][3] :
            negatives.append(items[i][1] if items[i][2] else 'a ' + items[i][1])
    
    if getSumDet(det, 'lamp finial') > 0 : gdescriptors.append('decorative')
    if getSumDet(det, 'lamp body jointed') > 0 : gdescriptors.append('jointed')
    if getSumDet(det, 'lamp body solid') > 0 : gdescriptors.append('solid')
    
    heads = max(getSumDet(det, 'lamp head'), getSumDet(det, 'lamp unit'))
    spnames.append(multiScriptor('head', heads, 6))
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    csize = int(round((cx + cz)/2))
    sdescriptors.append(lampSizes[0][csize])
    sdescriptors.append(lampSizes[1][cy])
        
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

faucetSize = ['very short', 'short', 'regular height', 'tall', 'very tall']   # y axis options (heightwise)  1
def dFaucet(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = 'faucet'
    if len(det) > 1 :
        subcat = 'shower faucet' if det[1][0]=='shower faucet' else 'faucet'
        
    switches = getSumDet(det, 'switch')
    spouts = getSumDet(det, 'spout')
    vsups = getSumDet(det, 'vertical support')
    hsups = getSumDet(det, 'horizontal support')
    parts = getSumDet(det, 'frame') 
    
    if switches > 0 : spnames.append(multiScriptor('switch', switches, 6))
    else : negatives.append('switch')
    if spouts > 0 : spnames.append(multiScriptor('spout', spouts, 6))
    else : negatives.append('spout')
    if vsups > 1 : spnames.append(multiScriptor('vertical bar', vsups, 5))
    if hsups > 1 : spnames.append(multiScriptor('horizontal bar', hsups, 5))
    if parts > 1 : spnames.append(multiScriptor('part', parts, 6))
    
    sdescriptors.append(faucetSize[row.cy])
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

clockSizes = [['very wide', 'wide', 'regular width', 'long', 'very long'],  # width
               ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size
               ['very thin', 'thin', 'average thickness', 'thick', 'very thick'],   # thickness options
               ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dClock(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=10)
    subcat = 'pendulum clock' if sum([getSumDet(det, 'pendulum clock'), getSumDet(det, 'pendulum')]) > 0 else subcat
    
    chains = getSumDet(det, 'chain')
    boxs = getSumDet(det, 'box')
    screens = getSumDet(det, 'screen')
    frames = getSumDet(det, 'frame')
    bases = getSumDet(det, 'base') + getSumDet(det, 'base surface') 
    foots = getSumDet(det, 'foot')
    
    if chains > 0 : spnames.append(multiScriptor('chain', chains, 10))
    if boxs > 0 : spnames.append('a box')
    if screens > 0 : spnames.append('a screen')
    else : negatives.append('screen')
    if frames > 0 : spnames.append('a frame')
    else : negatives.append('a frame')
    if bases > 0 : spnames.append('a base')
    else : negatives.append('a base')
    if foots > 0 : spnames.append(multiScriptor('foot', foots))
    else : negatives.append('foot')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    if ('grand' in row.subcats or 'pendul' in row.subcats) :
        sdescriptors.append(clockSizes[3][cy])
        clong = clamp(round((cy / (cx+.0001)) * 2), 0, len(clockSizes[0])-1)
        sdescriptors.append(clockSizes[0][clong])
    elif abs(dx - dy)>.05 :
        sdescriptors.append(clockSizes[3][cy])
        clong = clamp(round((cy / (cx+.0001)) * 2), 0, len(clockSizes[0])-1)
        sdescriptors.append(clockSizes[0][clong])
    else :
        sdescriptors.append(clockSizes[2][cz])
        cdia = int(round((cx + cy)/2))
        sdescriptors.append(clockSizes[1][cdia])
        
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

bottleSizes = [['very long', 'long', 'regular length', 'wide', 'very wide'],  # width
               ['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
               ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dBottle(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=10)
    
    mouths = getSumDet(det, 'mouth')
    necks = getSumDet(det, 'neck')
    handles = getSumDet(det, 'handle')
    lids = getSumDet(det, 'lid') + getSumDet(det, 'closure')
    
    if mouths > 0 : spnames.append('a mouth')
    else : negatives.append('a mouth')
    if necks > 0 : spnames.append('a neck')
    else : negatives.append('a neck')
    if lids > 0 : spnames.append('a lid')
    else : negatives.append('a lid')
    if handles > 0 : spnames.append(multiScriptor('handle', handles, 4))
    else : negatives.append('handle')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    sdescriptors.append(bottleSizes[2][cy])
    if abs(dx - dz)>.05 :
        clong = clamp(round((cz / (cx+.0001)) * 2), 0, len(bottleSizes[0])-1)
        sdescriptors.append(bottleSizes[0][clong])
    else :
        cdia = int(round((cx + cz)/2))
        sdescriptors.append(bottleSizes[1][cdia])
        
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

vaseSizes = [['very long', 'long', 'regular length', 'wide', 'very wide'],  # width
            ['very skinny', 'small', 'average size', 'large', 'very large'],   # diameter size
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dVase(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = rarestCat(row)
    
    plants = getSumDet(det, 'plant')
    stuff = getSumDet(det, 'liquid or soil')
    bases = getSumDet(det, 'base')
    foots = getSumDet(det, 'foot')
    lids = getSumDet(det, 'lid')
    
    if foots > 0 : spnames.append(multiScriptor('foot', foots))
    else : negatives.append('foot')
    if lids > 0 : spnames.append('lid')
    else : negatives.append('lid')
    if bases > 0 : spnames.append('base')
    else : negatives.append('a base')
    
    if plants >  0 and stuff >  0 : contains.append('a plant and some soil')
    if plants >  0 and stuff == 0 : contains.append('a plant')
        
    if plants == 0 and stuff >  0 : gdescriptors.append('full')
    if plants == 0 and stuff == 0 : gdescriptors.append('empty')    
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    sdescriptors.append(vaseSizes[2][cy])
    if abs(dx - dz)>.05 :
        clong = clamp(round((cz / (cx+.0001)) * 2), 0, len(vaseSizes[0])-1)
        sdescriptors.append(vaseSizes[0][clong])
    else :
        cdia = int(round((cx + cz)/2))
        sdescriptors.append(vaseSizes[1][cdia])
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

laptopSizes = [['very skinny', 'skinny', 'average width', 'wide', 'very wide']]
def dLaptop(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = 'laptop'
    
    tpad = getSumDet(det, 'touchpad')
    kboard = getSumDet(det, 'keyboard')
    screens = getSumDet(det, 'screen')
    
    if tpad > 0 : spnames.append('a touchpad')
    if kboard > 0 : spnames.append('a keyboard')
    if screens > 0 : spnames.append('a screen')
    
    if tpad == 0 : negatives.append('a touchpad')
    if kboard == 0 : negatives.append('a keyboard')
    if screens == 0 : gdescriptors.append('closed')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq    
    
    sdescriptors.append(laptopSizes[0][cx])
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

mugSizes = [['very small', 'small', 'average', 'large', 'very large'], # handle size
            ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dMug(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = rarestCat(row, min_cutoff=4)
    
    contain_things = getSumDet(det, 'containing things')
    handles = getSumDet(det, 'handle')
    
    if contain_things > 0 : gdescriptors.append('full')
    else : gdescriptors.append('empty')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq
    
    cdia = cz
    chandle = clamp(round((cz / (cx+.0001)) * 2), 0, len(mugSizes[0])-1)
    sdescriptors.append(mugSizes[1][cdia])
    sdescriptors.append(mugSizes[2][cy])    
        
    if handles == 0 : negatives.append('handle')
    elif handles == 1 : spnames.append('a {} handle'.format(mugSizes[0][chandle]))
    elif handles >  1 : spnames.append(multiScriptor('handle', handles, 4))
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

bowlSizes = [['round', 'oval'], # roundness
            ['very small', 'small', 'average size', 'large', 'very large'],   # diameter size 
            ['very short', 'short', 'average height', 'tall', 'very tall']]   # y axis options (heightwise)
def dBowl(row) :
    subcat, sdescriptors, gdescriptors, spnames, negatives, contains = '', [], [], [], [], []
    det = detToArr(row.details)
    subcat = 'bowl'
    
    contain_things = getSumDet(det, 'containing things')
    base = getSumDet(det, 'bottom')
    
    if contain_things > 0 : gdescriptors.append('full')
    else : gdescriptors.append('empty')
    if base > 0 : spnames.append('base')
    else : negatives.append('a base')
    
    cx, cy, cz, csq = row.cx, row.cy, row.cz, row.csq
    dx, dy, dz, dsq = row.dx, row.dy, row.dz, row.dsq    
    
    sdescriptors.append((bowlSizes[0][0] if abs(dx - dz)<.005 else bowlSizes[0][1]))
    cdia = int(round((cx + cz)/2))
    sdescriptors.append(bowlSizes[1][cdia])
    sdescriptors.append(bowlSizes[2][cy]) 
    
    cshape = shape(row.cattext, subcat, sdescriptors, gdescriptors, spnames, negatives, contains)
    return cshape

#%% Helper methods
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

# The details array contains: [self.name, level, children, quantity]
def detToArr(details) :
    dets = [[item.strip() for item in d.split(',')] for d in (' '+details).split('|')[:-1]]
    dets = np.array(dets)
    return dets

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

if len(corpdict) < 5 :
    buildCorpus(dfdesc)
    

#%% Setup class balancer. These numbers are based on estimates for how much attention each category requires relative to how many samples are available.
balancer = { 
'Table' : 1,
'Chair' : 1.4,
'Lamp' : 1.5,
'Faucet' : 1.8,
'Clock' : 1.3,
'Bottle' : 1.4,
'Vase' : 1.4,
'Laptop' : 1.2,
'Bed' : 3.5,
'Mug' : 2.5,
'Bowl' : 2.9 }

#%% Generate all descriptions and stack them into a numpy array
all_descs = []
for mid in tqdm(dfmeta.mid) :
    if (type(mid) == float) : continue
    row = dfmeta[dfmeta.mid == mid].iloc[0]
    if (not type(row.details) == float and len(row.details) > 1) :
        shape_object = dRow(row)
        for i in range(int(shape_object.getComplexity() * balancer[row.cattext])) :
            all_descs.append([mid, shape_object.getDesc()])
            
dnp = np.stack(all_descs)

#%% Get word count stats for the generated descriptions
total = 0
maxlen = 0
for d in dnp[:,1] :
    numwords = len(d.split())
    total += numwords
    maxlen = max(numwords, maxlen)
    
num_unique_descs = np.unique(dnp[:,1]).shape[0]
print('Average words per desc: {:.1f} \nMax words: {} \nUnique descs: {} / {}  = {:.1f}%'.format(total / len(dnp[:,1]), maxlen, num_unique_descs, len(dnp), 100 * num_unique_descs / len(dnp)))

#%% Save descriptions to file
np.save(os.path.join(cf.DATA_DIR, 'alldnp.npy'), dnp, allow_pickle=True)

#%% Inspect some objects and their generated descriptions
catid = 0

while True:
    index = rn.randint(1, len(dfdesc)-1)
    row = dfdesc.iloc[index]
    if (not row.cattext == cats_to_load[catid]) : continue
    print('-------------------------------------------{}------------------------------------------------'.format(index))
    print('{:20}    {}'.format(row.cattext, row.mid))

    sh = dRow(row)
    print(sh)
    for i in range(int(sh.getComplexity() * balancer[row.cattext] * 1)) :
        print(sh.getDesc())
    
    ut.showPic(dfdesc.iloc[index].mid,title=index)
    try : 
        i = input('')
        if i=='s' : 
            print('showing binvox...')
            ut.showBinvox(row.mid)
    except (KeyboardInterrupt, SystemExit): break