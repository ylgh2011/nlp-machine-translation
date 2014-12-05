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
import score_reranker_avg

optparser = optparse.OptionParser()

########################################################################################################################################
## Parameters for iter
optparser.add_option("-i", "--iter", dest="iter", default=5, type="int", help="Number of iterations between decoder and reranker")

########################################################################################################################################
## Parameters for decoder part
# optparser.add_option("--input", dest="input", default="/usr/shared/CMPT/nlp-class/project/toy/train.cn", help="File containing sentences to translate (default=data/input)")
optparser.add_option("--input", dest="input", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
# optparser.add_option("--input", dest="input", default="/usr/shared/CMPT/nlp-class/project/small/train.cn", help="File containing sentences to translate (default=data/input)")
# optparser.add_option("--input", dest="input", default="../segmenter/train.cn", help="File containing sentences to translate (default=data/input)")

# optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/toy/phrase-table/phrase_table.out", help="File containing translation model (default=data/tm)")
optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
# optparser.add_option("--translation-model", dest="tm", default="/usr/shared/CMPT/nlp-class/project/large/phrase-table/test-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")

# optparser.add_option("--language-model", dest="lm", default="/usr/shared/CMPT/nlp-class/project/lm/en.tiny.3g.arpa", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("--language-model", dest="lm", default="/usr/shared/CMPT/nlp-class/project/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")

optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("--heap-size", dest="s", default=100, type="int", help="Maximum heap size (default=1)")
optparser.add_option("--disorder", dest="disord", default=3, type="int", help="Disorder limit (default=3)")
optparser.add_option("--diseta", dest="diseta", type="float", default=0.1, help="perceptron learning rate (default 0.1)")
optparser.add_option("--beam width", dest="bwidth", default=600,  help="beamwidth")
optparser.add_option("--mute", dest="mute", default=0, type="int", help="mute the output")
optparser.add_option("--nbest", dest="nbest", default=100, type="int", help="print out nbest results")

########################################################################################################################################
## Parameters for reranker part
# optparser.add_option("--en", dest="en", default="/usr/shared/CMPT/nlp-class/project/toy/train.cn", help="target language references for learning how to rank the n-best list")
# optparser.add_option("--en", dest="en", default="/usr/shared/CMPT/nlp-class/project/small/train.en", help="target language references for learning how to rank the n-best list")

optparser.add_option("--en0", dest="en0", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.en0", help="target language references for learning how to rank the n-best list")
optparser.add_option("--en1", dest="en1", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.en1", help="target language references for learning how to rank the n-best list")
optparser.add_option("--en2", dest="en2", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.en2", help="target language references for learning how to rank the n-best list")
optparser.add_option("--en3", dest="en3", default="/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.en3", help="target language references for learning how to rank the n-best list")

optparser.add_option("--epo", dest="epo", type="int", default=5, help="number of epochs for perceptron training (default 5)")
optparser.add_option("--eta", dest="eta", type="float", default=0.1, help="perceptron learning rate (default 0.1)")
optparser.add_option("--xi", dest="xi", type="int", default=100, help="training data generated from the samples tau (default 100)")
optparser.add_option("--alpha", dest="alpha", type="float", default=0.1, help="sampler acceptance cutoff (default 0.1)")
optparser.add_option("--tau", dest="tau", type="int", default=5000, help="samples generated from n-best list per input sentence (default 5000)")

########################################################################################################################################
## opts
opts = optparser.parse_args()[0]




########################################################################################################################################
## init for decoder part
lm = models.LM(opts.lm, opts.mute)
tm = models.TM(opts.tm, opts.k, opts.mute)


french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
bound_width = float(opts.bwidth)

for word in set(sum(french,())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, [0.0, 0.0, 0.0, 0.0])]



# ibm_t = {}
ibm_t = library.init('./data/ibm.t.gz')


########################################################################################################################################
## init for reranker part
references = [[], [], [], []]
sys.stderr.write("Reading English Sentences ... \n")
def readReference(ref_fileName):
    ref = []
    for i, line in enumerate(open(ref_fileName)):
        # Initialize references to correct english sentences
        ref.append(line)
        if i%1000 == 0:
            sys.stderr.write(".")
    sys.stderr.write("\n")
    return ref


references[0] = readReference(opts.en0)
references[1] = readReference(opts.en1)
references[2] = readReference(opts.en2)
references[3] = readReference(opts.en3)


########################################################################################################################################
## start doing iteration between decoder and reranker
# w = [1.0, -0.01, 1.0, 1.0, 1.0, 1.0, 1.0]
# w = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]



reload(beam)
reload(cal_weight)
reload(models)
reload(library)
reload(rerank)
reload(score_reranker_avg)


w = [1.0/6] * 6
nbest_sentences_0 = beam.main(opts, w, tm, lm, french, ibm_t)
(score_list, translation_list) = rerank.main(w, nbest_sentences_0)
best_bleu_score = score_reranker_avg.main(opts, translation_list)
print "w = " + str(w) +  ", score before iteration: " + str(best_bleu_score)

for i in range(opts.iter):
    sys.stderr.write("Iteration %d\n" % i)
    # beam decode and output nbest file
    nbest_sentences = beam.main(opts, w, tm, lm, french, ibm_t)
    # calculate weight and output the result weight
    new_w_str = cal_weight.main(opts, references[0], nbest_sentences, w)
    new_w = [float(item) for item in new_w_str.split('\n')]
    # rerank using the output weight 
    (score_list, translation_list) = rerank.main(new_w, nbest_sentences)
    # calculate the BLEU score for the test set
    new_bleu_score = score_reranker_avg.main(opts, translation_list)
    sys.stderr.write("new_w = " + str(new_w) + ", new_bleu_score = " + str(new_bleu_score) + '\n')
    if best_bleu_score < new_bleu_score:
        best_bleu_score = new_bleu_score
        w = new_w
    else:
        pass
        # break


print "best score: " + str(best_bleu_score)
for item in w:
    print item






# (score_list, translation_list) = rerank.main(w, nbest_sentences)
# counter = 0
# for (score, translation) in zip(score_list, translation_list):
#     counter += 1
    # sys.stderr.write('______________ # ' + str(counter) + ' score: ' + str(score) + '\n')
    # print translation

# print '\n'.join([str(item) for item in w])


reload(beam)
reload(cal_weight)
reload(models)
reload(library)
reload(rerank)
reload(score_reranker_avg)



