#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple

def main(w, nbest_list):

    translation = namedtuple("translation", "english, score")
    nbests = []
    for line in nbest_list:
        (i, sentence, features) = line.strip().split("|||")
        if len(nbests) <= int(i):
            nbests.append([])
        features = [float(h) for h in features.strip().split()]
        if w is None:
            w = [1.0/len(features) for _ in xrange(len(features))]
        nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

    score_list = []
    translation_list = []
    for nbest in nbests:
        t = sorted(nbest, key=lambda x: -x.score)[0]
        score_list.append(t.score)
        translation_list.append(t.english)

    return (score_list, translation_list)


