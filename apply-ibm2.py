# Output the viterbi alignments.

from __future__ import division
import sys
import codecs
from math import log
from collections import Counter, defaultdict


if len(sys.argv) < 6:
    print "Usage: %s <source> <target> <translation_table> <distortion_table> <output>" 
    sys.exit(-1)

frenchfile = sys.argv[1]
englishfile = sys.argv[2]
translationfile = sys.argv[3]
distortionfile = sys.argv[4]
output = sys.argv[5]


# Since data are small, load sentences into memory.
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
        s = ["null"]
        s.extend(f.lower().split())
        pairs[cnt].append(s)
        cnt += 1

        # Store each word's index in the translation table.
        for w in s:
            if w not in fr:
                fr[w] = frcnt
                frcnt += 1

# Load the translation table.
translation = {}
with codecs.open(translationfile, encoding='utf-8') as T:
    for t in T:
        s = t.lower().split()
        pr = float(s[2])
        if pr > 0:
            translation[(s[1], s[0])] = pr

# Load the distortion table.
distortion = {}
with codecs.open(distortionfile, encoding='utf-8') as D:
    for d in D:
        s = d.lower().split()
        distortion[(int(s[0]), int(s[1]), int(s[2]), int(s[3]))] = float(s[4])


with codecs.open(output, 'w+', encoding='utf-8') as O:
    for pair in pairs:
        I = len(pair[0])
        K = len(pair[1])
        viterbi = []

        for i, e in enumerate(pair[0]):
            curmax = -100
            for k, f in enumerate(pair[1]):
                try: p = distortion[(k, i, I, K)] * translation[(e, f)] 
                except KeyError: 
                    p = 0

                if p > curmax:
                    curmax = p
                    curpos = k

            viterbi.append(str(curpos))

        # Write to output.
        O.write(' '.join(viterbi) + '\n')
