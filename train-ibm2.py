# Output the Viterbi alignments.
# That is mapping:  target -> source indexes.
from __future__ import division
import sys
import codecs
from math import log
from collections import Counter, defaultdict


if len(sys.argv) < 6:
    print "Usage: %s <source> <target> <translation_table> <output1_translation> <output2_distortion>" 
    sys.exit(-1)

frenchfile = sys.argv[1]
englishfile = sys.argv[2]
translationfile = sys.argv[3]
outputtrans = sys.argv[4]
outputdistor = sys.argv[5]


# Since data are small, load sentences into memory.
pairs = [] 
eng = set([])
with codecs.open(englishfile, encoding='utf-8') as E:
    for e in E:
        s = e.lower().split()
        pairs.append([s])
        for w in s: eng.add(w)
engcnt = len(eng)


fr = set([])
cnt = 0 
with codecs.open(frenchfile, encoding='utf-8') as F:
    for f in F:
        s = ["null"]
        s.extend(f.lower().split())
        for w in s: fr.add(w)
        pairs[cnt].append(s)
        cnt += 1
frcnt = len(fr)

# Load the translation table.
translation = {}
with codecs.open(translationfile, encoding='utf-8') as T:
    for t in T:
        s = t.lower().split()
        pr = float(s[2])
        if pr > 0:
            translation[(s[1], s[0])] = pr


# Run EM steps until convergence.
# Convergence defined as very small increase in the log-likelihood of the corpus. 
distortion = defaultdict(lambda: 1 / frcnt) # Uniform initialization
iterations = 0
old_likelihood = -2 # dummy values
new_likelihood = -1

epsilon = 10**(-5)
while iterations < 2 or abs(new_likelihood - old_likelihood) > epsilon:    

        iterations += 1        
        if iterations > 1: old_likelihood = new_likelihood

        if iterations > 150: break

        ## E-STEP
        new_likelihood = 0
        count_t = defaultdict(float)
        total_t = defaultdict(float)     

        count_a = defaultdict(float)
        total_a = defaultdict(float)

        for pair in pairs:
            I = len(pair[0])
            K = len(pair[1])

            # Compute normalization constant. 
            z = defaultdict(float)
            for i, e in enumerate(pair[0]):
                for k, f in enumerate(pair[1]):
                    try: z[e] += distortion[(k, i, I, K)] * translation[(e, f)]
                    except KeyError: pass # combination has 0 probability.
                
            # Collect link counts weighted by the posterior link probability.
            for i, e in enumerate(pair[0]):
                for k, f in enumerate(pair[1]):

                    try: 
                        if z[e] != 0:
                            p = distortion[(k, i, I, K)] * translation[(e, f)] / z[e]
                        else:
                            p = 0
                    except KeyError: p = 0

                    count_t[(e, f)] += p 
                    total_t[f] += p
                    count_a[(k, i, I, K)] += p
                    total_a[(i, I, K)] += p

                if z[e] != 0:
                    new_likelihood += log(z[e])
            

        ## M-STEP
        for f in fr:
            for e in eng:
                if total_t[f] != 0:
                    translation[(e, f)] = count_t[(e, f)] / total_t[f]

        for pair in pairs:
            I = len(pair[0])
            K = len(pair[1])
            for i in range(I):
                for k in range(K):
                    if total_a[(i, I, K)] != 0:
                        distortion[(k, i, I, K)] = count_a[(k, i, I, K)] / total_a[(i, I, K)]

        #print iterations, new_likelihood, abs(new_likelihood - old_likelihood)

print 'EM converged.'
print 'Writing translation probabilities to file...'
minval = 10 ** (-15)
with codecs.open(outputtrans, 'w+', encoding='utf-8') as T:
    for t in translation:
        if translation[t] > 0:
            if translation[t] < minval:
                translation[t] = minval
            T.write('%s %s %.15f\n' % (t[1], t[0], translation[t]))

print 'Done.'
print 'Writing distortion table to file...'
with codecs.open(outputdistor, 'w+', encoding='utf-8') as D:
    for d in distortion:
        if distortion[d] < minval:
            distortion[d] = minval
        D.write(('%d %d %d %d %.15f\n') % (d[0], d[1], d[2], d[3], distortion[d]))
print 'Done.'
