# Output the Viterbi alignments.
# That is mapping:  target -> source indexes.

import sys
import codecs
import numpy as np
from collections import Counter

if len(sys.argv) < 5:
    print "Usage: %s <source> <target> <translation_table> <output>"
    sys.exit(-1)

frenchfile = sys.argv[1]
englishfile = sys.argv[2]
translationfile = sys.argv[3]
outputfile = sys.argv[4]


# Since data are small, load sentences into memory.
pairs = [] 
eng = {}
enginv = {}
engcnt = 0
with codecs.open(englishfile, encoding='utf-8') as E:
    for e in E:
        s = e.lower().split()
        pairs.append([s])

        # Store each word's index in the translational table.
        for w in s:
            if w not in eng:
                eng[w] = engcnt
                enginv[engcnt] = w
                engcnt += 1

fr = {} 
frinv = {}
cnt = 0 
frcnt = 0 
with codecs.open(frenchfile, encoding='utf-8') as F:
    for f in F:
        s = ["null"]
        s.extend(f.lower().split())
        pairs[cnt].append(s)
        cnt += 1

        # Store each word's index in the translation table.
        for w in s:
            if w not in fr:
                fr[w] = frcnt
                frinv[frcnt] = w
                frcnt += 1

# Swap NULL with the element at index 0.
fr[frinv[0]] = fr["null"]
frinv[fr["null"]] = frinv[0]
fr["null"] = 0
frinv[0] = "null"

# Load the translation table.
translation = np.zeros([frcnt, engcnt])
with codecs.open(translationfile, encoding='utf-8') as T:
    for t in T:
        s = t.lower().split()
        pr = float(s[2])
        if pr > 0:
            translation[fr[s[0]], eng[s[1]]] = pr
        

# Print viterbi alignment for each sentence.
maxlikelihood = np.argmax(translation, axis=1)

with codecs.open(outputfile, 'w+', encoding='utf-8') as O:
    for pair in pairs:
        indeng = Counter([eng[p] for p in pair[0]])
        indfr  = Counter([fr[p] for p in pair[1]])

        sortedeng = sorted(indeng)
        sortedfr = sorted(indfr)
        indexes = np.ix_(sortedfr, sortedeng)

        # Indexes in the array of most likely french words.
        maxlikelihood = np.argmax(translation[indexes], axis=0)

        viterbi = []
        for e in pair[0]:
            indsource = sortedeng.index(eng[e])
            indtarget = pair[1].index(frinv[sortedfr[maxlikelihood[indsource]]])
            viterbi.append(str(indtarget))

        O.write(' '.join(viterbi) + '\n')
