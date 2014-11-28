#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple, defaultdict
# from bleu import bleu_stats, bleu, smoothed_bleu
import bleu
from operator import itemgetter
import itertools
import random
from math import fabs
from library import pre_process
from library import read_ds_from_file
from library import write_ds_to_file



optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("--en", dest="en", default=os.path.join("data", "train.en"), help="target language references for learning how to rank the n-best list")
optparser.add_option("-t", "--tau", dest="tau", type="int", default=5000, help="samples generated from n-best list per input sentence (default 5000)")
optparser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.1, help="sampler acceptance cutoff (default 0.1)")
optparser.add_option("-x", "--xi", dest="xi", type="int", default=100, help="training data generated from the samples tau (default 100)")
optparser.add_option("-e", "--eta", dest="eta", type="float", default=0.1, help="perceptron learning rate (default 0.1)")
optparser.add_option("-p", "--epo", dest="epo", type="int",default=5, help="number of epochs for perceptron training (default 5)")


optparser.add_option("--fr", dest="fr", default=os.path.join("data", "train.fr"), help="train French file")
optparser.add_option("--testnbest", dest="testnbest", default=os.path.join("data", "test.nbest"), help="test N-best file")
optparser.add_option("--testfr", dest="testfr", default=os.path.join("data", "test.fr"), help="test French file")
optparser.add_option("--nbestDic", dest="nbestDS", default=os.path.join("data", "nbest.ds.gz"), help="dumping file of the data structure that storing scores for nbestDic")
optparser.add_option("--testen", dest="testen", default=os.path.join("data", "test.en"), help="test en")

(opts, _) = optparser.parse_args()
# entry = namedtuple("entry", "sentence, bleu_score, smoothed_bleu, feature_list")
entry = namedtuple("entry", "sentence, smoothed_bleu, feature_list")

pre_process( [(opts.fr, opts.nbest, opts.en), (opts.testfr, opts.testnbest, opts.testen)] )


def get_sample(nbest):
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


def main():
    references = []
    sys.stderr.write("Reading English Sentences\n")
    for i, line in enumerate(open(opts.en)):
        '''Initialize references to correct english sentences'''
        references.append(line)
        if i%100 == 0:
            sys.stderr.write(".")

    sys.stderr.write("\nTry reading nbests datastructure from disk ... \n")
    nbests = read_ds_from_file(opts.nbestDS)
    if nbests is None:
        nbests = []
        sys.stderr.write("No nbests on disk, so calculating ndests ... \n")
        for j,line in enumerate(open(opts.nbest)):
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
        write_ds_to_file(nbests, opts.nbestDS)

    arg_num = len(nbests[0][0].feature_list)
    theta = [1.0/arg_num for _ in xrange(arg_num)] #initialization

    avg_theta = [ 0.0 for _ in xrange(arg_num)]
    avg_cnt = 0
    sys.stderr.write("\nTraining...\n")
    for j in xrange(opts.epo):
        mistake = 0;
        for nbest in nbests:
            sample = get_sample(nbest)
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
    # instead of print the averaged-out weights, print the weights that maximize the BLEU score    
    print "\n".join([str(weight) for weight in weights])

if __name__ == '__main__':
    main()
