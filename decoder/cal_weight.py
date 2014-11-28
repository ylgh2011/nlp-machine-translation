#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple, defaultdict
# from bleu import bleu_stats, bleu, smoothed_bleu
import bleu
from operator import itemgetter
import itertools
import random
from math import fabs


def get_sample(nbest, opts):
    '''
    nbest is a list of [setence, bleu_score, smoothed_bleu_score, featrue_list]
    We only use bleu_socre and smoothed_bleu_score here
    '''

    # generate all the pairs combinations
    # len(pairs) = len(nbest) * (len(nbest) - 1) / 2
    pairs = list(itertools.combinations(range(0, len(nbest)), 2))
    random.shuffle(pairs)


    samples = [];
    for pair in pairs:
        # pair will be random pair index from nbest
        if len(samples) >= opts.tau:
            break
        if fabs(nbest[pair[0]].smoothed_bleu - nbest[pair[1]].smoothed_bleu) > opts.alpha:
            if nbest[pair[0]].smoothed_bleu > nbest[pair[1]].smoothed_bleu:
                samples.append((nbest[pair[0]], nbest[pair[1]]))
            else:
                samples.append((nbest[pair[1]], nbest[pair[0]]))
        else:
            continue
    return samples


def dot_product(l1, l2):
    if (len(l1) != len(l2)):
        raise(ValueError, "product of dif length of vectors")
    ans = 0.0
    for i in xrange(len(l1)):
        ans += l1[i] * l2[i]
    return ans


def vector_plus(v1, v2, multiply=1):
    ans = []
    for i in xrange(len(v1)):
        ans.append(v1[i] + multiply * v2[i])
    return ans


def main(opts, references, input_nbest):
    entry = namedtuple("entry", "sentence, smoothed_bleu, feature_list")
    nbests = None
    if nbests is None:
        nbests = []
        sys.stderr.write("No nbests on disk, so calculating ndests ... \n")
        for j,line in enumerate(input_nbest):
            (i, sentence, features) = line.strip().split("|||")
            i = int(i)
            stats = list(bleu.bleu_stats(sentence, references[i]))
            # bleu_score = bleu.bleu(stats)
            smoothed_bleu_score = bleu.smoothed_bleu(stats)
            # making the feature string to float list
            feature_list = [float(x) for x in features.split()]
            if len(nbests)<=i:
                nbests.append([])
            # nbests[i].append(entry(sentence, bleu_score, smoothed_bleu_score, feature_list))
            nbests[i].append(entry(sentence, smoothed_bleu_score, feature_list))

            if j%5000 == 0:
                sys.stderr.write(".")

    arg_num = len(nbests[0][0].feature_list)
    theta = [1.0/arg_num for _ in xrange(arg_num)] #initialization

    avg_theta = [ 0.0 for _ in xrange(arg_num)]
    avg_cnt = 0
    sys.stderr.write("\nTraining...\n")
    for j in xrange(opts.epo):
        mistake = 0;
        for nbest in nbests:
            sample = get_sample(nbest, opts)
            sample.sort(key=lambda i: i[0].smoothed_bleu - i[1].smoothed_bleu, reverse=True)
            for i in xrange(min(len(sample), opts.xi)):
                v1 = sample[i][0].feature_list
                v2 = sample[i][1].feature_list
                if dot_product(theta, v1) <= dot_product(theta, v2):
                    mistake += 1
                    theta = vector_plus(theta, vector_plus(v1, v2, -1), opts.eta)
                    
                avg_theta = vector_plus(avg_theta, theta)
                avg_cnt += 1

        sys.stderr.write("Mistake:  %s\n" % (mistake,))
    

    weights = [ avg / avg_cnt if avg_cnt !=0 else 1/float(arg_num) for avg in avg_theta ]
    sys.stderr.write("Computing best BLEU score and outputing...\n")
    # instead of return the averaged-out weights, return the weights that maximize the BLEU score    
    return "\n".join([str(weight) for weight in weights])

# if __name__ == '__main__':
#     # main()
