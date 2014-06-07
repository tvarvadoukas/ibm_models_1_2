#!/usr/bin/python
# -*- coding: utf-8 -*- 

from __future__ import division
from collections import Counter
from math import log
import numpy as np
import codecs
import sys

##
## Input:   corpus of parallel sentence pairs PC = <<E, F>>
## Output:  translational probabilities t(e | f) over the 
##          vocabularies E and F, where F includes the NULL word. 
##          For a given f in F =>  sum(t(e|f), forall e) = 1

if len(sys.argv) < 4:
    print "Usage: %s <source> <target> <output>" % sys.argv[0]
    sys.exit(-1)

frenchfile = sys.argv[1]
englishfile = sys.argv[2]
translationoutput = sys.argv[3]


# Since data are small do:
#   - load sentence pairs into memory. 
#   - create a 2d matrix FxE to keep translation probabilities.
#     (in large and sparse data we should create forward/inverted indexes).
pairs = [] 
eng = {}
engcnt = 0
with codecs.open(englishfile, encoding='utf-8') as E:
    for e in E:
        s = e.lower().split()
        pairs.append([s])

        # Store each word's index in the translational table.
        for w in s:
            if w not in eng:
                eng[w] = engcnt
                engcnt += 1

fr = {} 
cnt = 0 
frcnt = 0 
with codecs.open(frenchfile, encoding='utf-8') as F:
    for f in F:
        s = f.lower().split()
        s.append("null")
        pairs[cnt].append(s)
        cnt += 1

        # Store each word's index in the translation table.
        for w in s:
            if w not in fr:
                fr[w] = frcnt
                frcnt += 1

# Create the translation table and initialize uniformly.
translation = np.ones([frcnt, engcnt]) / engcnt


# Data structures to work with from now on:
#   - translation:  numpy probability matrix FxE, each row sums to 1.
#   - pairs:        list of lists. Each element is a list with 2 lists 
#                   for each Eng-Fr pairs.   i.e. [[ENG, FR], ...]
#   - fr & eng:     mapping between word and index in translation table.

# Run EM steps until convergence.
# Convergence defined as "no increase in the log-likelihood of the corpus. 
iterations = 0
old_likelihood = -2 # dummy values
new_likelihood = -1

epsilon = 10**(-5)
while iterations < 2 or abs(new_likelihood - old_likelihood) > epsilon:    

        iterations += 1        
        if iterations > 1: old_likelihood = new_likelihood

        if iterations > 30: break

        ## E-STEP
        new_likelihood = 0
        count_e_f = np.zeros([frcnt, engcnt])
        total_f = np.zeros(frcnt)        

        for pair in pairs:

            # Get french and english indexes.
            indeng = Counter([eng[p] for p in pair[0]])            
            indfr  = Counter([fr[p] for p in pair[1]])            

            # Compute the weighted translation sub-matrix to account for duplicates.
            sortedeng = sorted(indeng)
            sortedfr = sorted(indfr)
            indexes = np.ix_(sortedfr, sortedeng)
            weighted_translation = translation[indexes]

            for pos, y in enumerate(sortedeng): 
                if indeng[y] > 1: weighted_translation[:, pos] *= indeng[y]           

            for pos, y in enumerate(sortedfr): 
                if indfr[y] > 1:  weighted_translation[pos, :] *= indfr[y]           

            # Compute the normalization constant for each word e.
            z = np.sum(weighted_translation, axis=0)

            # log-likelihood of the corpus = sumlog(normalization constant for each e)
            new_likelihood += np.sum(np.log(z))
            
            # Collect counts.
            temp = weighted_translation / z
            count_e_f[indexes] += temp
            total_f[sortedfr] += np.sum(temp, axis=1)

        ## M-STEP
        translation = (count_e_f.T / total_f).T
        #print iterations, new_likelihood, new_likelihood - old_likelihood


# Print translation table.
print "EM converged."
print "Writing translation probabilities to the output file..."
minpr = 10 ** (-15)
with codecs.open(translationoutput, 'w+', encoding='utf-8') as T:
    for f in fr:
        for e in eng:
            pr = translation[fr[f], eng[e]]
            if pr > 0:
                if pr < minpr: pr = minpr
                T.write('%s %s %.15f\n' % (f, e, pr))
print "The End."

