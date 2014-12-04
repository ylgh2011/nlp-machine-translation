#!/usr/bin/env python
import optparse
import sys
import models
import copy
from collections import namedtuple
from models import getDotProduct

from library import read_ds_from_file
from library import write_ds_to_file
from library import init
from library import t_f_given_e
from library import q_j_given_i_l_m
from library import ibm_model_1_score
from library import ibm_model_1_w_score

def bitmap(sequence):
    """ Generate a coverage bitmap for a sequence of indexes """
    return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def onbits(b):
    """ Count number of on bits in a bitmap """
    return 0 if b==0 else (1 if b&1==1 else 0) + onbits(b>>1)

def prefix1bits(b):
    """ Count number of bits encountered before first 0 """
    return 0 if b&1==0 else 1+prefix1bits(b>>1)

def last1bit(b):
    """ Return index of highest order bit that is on """
    return 0 if b==0 else 1+last1bit(b>>1)

# def stateeq(state1, state2):
#     return (state1.lm_state == state2.lm_state) and (state1.end == state2.end) and (state1.coverage == state2.coverage)
hypothesis = namedtuple("hypothesis", "lm_state, logprob, coverage, end, predecessor, phrase, distortionPenalty")

def main(opts, w, tm, lm, french, ibm_t):
    # tm should translate unknown words as-is with probability 1
    # if w is None:
    #     w = [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    nbest_output = []
    total_prob = 0
    if opts.mute == 0:
        sys.stderr.write("Start decoding %s ...\n" % (opts.input,))
    for idx,f in enumerate(french):
        if opts.mute == 0 and idx % 500 == 0:
            sys.stderr.write("Decoding sentence #%s ...\n" % (str(idx)))
        initial_hypothesis = hypothesis(lm.begin(), 0.0, 0, 0, None, None, None)
        heaps = [{} for _ in f] + [{}]
        heaps[0][lm.begin(), 0, 0] = initial_hypothesis
        for i, heap in enumerate(heaps[:-1]):
            # maintain beam heap
            front_item = sorted(heap.itervalues(), key=lambda h: -h.logprob)[0]

            for h in sorted(heap.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
                if h.logprob < front_item.logprob - float(opts.bwidth): continue

                fopen = prefix1bits(h.coverage)
                for j in xrange(fopen,min(fopen+1+opts.disord, len(f)+1)):
                    for k in xrange(j+1, len(f)+1):
                        if f[j:k] in tm:
                            if (h.coverage & bitmap(range(j, k))) == 0:
                                for phrase in tm[f[j:k]]:
                                    lm_prob = 0
                                    lm_state = h.lm_state
                                    for word in phrase.english.split():
                                        (lm_state, prob) = lm.score(lm_state, word)
                                        lm_prob += prob
                                    lm_prob += lm.end(lm_state) if k == len(f) else 0.0
                                    coverage = h.coverage | bitmap(range(j, k))
                                    # logprob = h.logprob + lm_prob*w[0] + getDotProduct(phrase.several_logprob, w[2:6]) + abs(h.end+1-j)*w[1] + ibm_model_1_w_score(ibm_t, f, phrase.english)*w[6]
                                    logprob  = h.logprob
                                    logprob += lm_prob*w[0]
                                    logprob += getDotProduct(phrase.several_logprob, w[2:6])
                                    logprob += opts.diseta*abs(h.end+1-j)*w[1]
                                    logprob += ibm_model_1_w_score(ibm_t, f, phrase.english)*w[6]

                                    new_hypothesis = hypothesis(lm_state, logprob, coverage, k, h, phrase, abs(h.end + 1 - j))

                                    # add to heap
                                    num = onbits(coverage)
                                    if (lm_state, coverage, k) not in heaps[num] or new_hypothesis.logprob > heaps[num][lm_state, coverage, k].logprob:
                                        heaps[num][lm_state, coverage, k] = new_hypothesis


        winners = sorted(heaps[-1].itervalues(), key=lambda h: h.logprob)[0:opts.nbest]

        def get_lm_logprob(test_list):
            stance = []
            for i in test_list:
                stance += (i.split())
            stance = tuple(stance)
            lm_state = (stance[0],)
            score = 0.0
            for word in stance:
                (lm_state, word_score) = lm.score(lm_state, word)
                score += word_score
            return score
        def get_list_and_features(h):
            lst = [];
            features = [0, 0, 0, 0, 0, 0, 0]
            current_h = h;
            while current_h.phrase is not None:
                # print current_h
                lst.append(current_h.phrase.english);
                features[1] += current_h.distortionPenalty              # distortion penaulty
                features[2] += current_h.phrase.several_logprob[0]      # translation feature 1
                features[3] += current_h.phrase.several_logprob[1]      # translation feature 2
                features[4] += current_h.phrase.several_logprob[2]      # translation feature 3
                features[5] += current_h.phrase.several_logprob[3]      # translation feature 4
                current_h = current_h.predecessor
            lst.reverse()
            features[0] = get_lm_logprob(lst)                           # language model score
            features[6] = ibm_model_1_score(ibm_t, f, lst)
            return (lst, features)

        for win in winners:
            s = str(idx) + " ||| "
            (lst, features) = get_list_and_features(win)
            for word in lst:
                s += word + ' '
            s += '||| '
            for fea in features:
                s += str(fea) + ' '
            nbest_output.append(s)

    # log_file = open('./log.txt','wb')
    # log_file.write(str(nbest_output))

    return nbest_output

# if __name__ == "__main__":
#     # main()

