#!/usr/bin/env python
import optparse
import sys
import copy
from collections import namedtuple

import beam
import cal_weight
import models
import library
import rerank

optparser = optparse.OptionParser()

########################################################################################################################################
## Parameters for iter
optparser.add_option("-i", "--iter", dest="iter", default=5, type="int", help="Number of iterations between decoder and reranker")

########################################################################################################################################
## Parameters for decoder part
optparser.add_option("--input", dest="input", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")

# optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
# optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/small/phrase-table/moses/phrase-table.gz", help="File containing translation model (default=data/tm)")
optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")

optparser.add_option("--language-model", dest="lm", default="/usr/shared/CMPT/nlp-class/project/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
# optparser.add_option("--language-model", dest="lm", default="/usr/shared/CMPT/nlp-class/project/lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")

optparser.add_option("--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("--translations-per-phrase", dest="k", default=20, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("--heap-size", dest="s", default=1000, type="int", help="Maximum heap size (default=1)")
optparser.add_option("--disorder", dest="disord", default=5, type="int", help="Disorder limit (default=6)")
optparser.add_option("--beam width", dest="bwidth", default=1.0,  help="beamwidth")
optparser.add_option("--mute", dest="mute", default=0, type="int", help="mute the output")
optparser.add_option("--nbest", dest="nbest", default=100, type="int", help="print out nbest results")

########################################################################################################################################
## Parameters for reranker part
optparser.add_option("--en", dest="en", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.en0", help="target language references for learning how to rank the n-best list")
optparser.add_option("--epo", dest="epo", type="int",default=5, help="number of epochs for perceptron training (default 5)")
optparser.add_option("--eta", dest="eta", type="float", default=0.1, help="perceptron learning rate (default 0.1)")
optparser.add_option("--xi", dest="xi", type="int", default=100, help="training data generated from the samples tau (default 100)")
optparser.add_option("--alpha", dest="alpha", type="float", default=0.1, help="sampler acceptance cutoff (default 0.1)")
optparser.add_option("--tau", dest="tau", type="int", default=5000, help="samples generated from n-best list per input sentence (default 5000)")

########################################################################################################################################
## opts
opts = optparser.parse_args()[0]




########################################################################################################################################
## init for decoder part
tm = models.TM(opts.tm, opts.k, opts.mute)
lm = models.LM(opts.lm, opts.mute)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
bound_width = float(opts.bwidth)

for word in set(sum(french,())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, [0.0, 0.0, 0.0, 0.0])]

ibm_t = library.init('./data/ibm2.t.ds.gz')


########################################################################################################################################
## init for reranker part
references = []
sys.stderr.write("Reading English Sentences\n")
for i, line in enumerate(open(opts.en)):
    # Initialize references to correct english sentences
    references.append(line)
    if i%100 == 0:
        sys.stderr.write(".")
sys.stderr.write("\n")


########################################################################################################################################
## start doing iteration between decoder and reranker
w = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
nbest_sentences = []
for i in range(opts.iter):
    sys.stderr.write("Iteration %d\n" % i)
    nbest_sentences = beam.main(opts, w, tm, lm, french, ibm_t)
    w_str = cal_weight.main(opts, references, nbest_sentences)
    w = [float(item) for item in w_str.split('\n')]
    sys.stderr.write("w = " + str(w) + '\n')

(score_list, translation_list) = rerank.main(w, nbest_sentences)
for (score, translation) in zip(score_list, translation_list):
    sys.stderr.write(str(score) + '\n')
    print translation

# print '\n'.join([str(item) for item in w])
