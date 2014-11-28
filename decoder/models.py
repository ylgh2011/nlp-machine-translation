#!/usr/bin/env python
# Simple translation model and language model data structures
import sys
import gzip
from collections import namedtuple, defaultdict
from math import log

# A translation model is a dictionary where keys are tuples of French words
# and values are lists of (english, logprob) named tuples. For instance,
# the French phrase "que se est" has two translations, represented like so:
# tm[('que', 'se', 'est')] = [
#   phrase(english='what has', logprob=-0.301030009985), 
#   phrase(english='what has been', logprob=-0.301030009985)]
# k is a pruning parameter: only the top k translations are kept for each f.
phrase = namedtuple("phrase", "english, several_logprob")

def getDotProduct(several_logprob, w=None):
    if w is None:
        w = [1]*len(several_logprob)
    dotProduct = 0;
    for i, logprob in enumerate(several_logprob):
        dotProduct += logprob * w[i];
    return dotProduct


def TM(filename, k, mute=1):
    if (mute == 0):
        sys.stderr.write("Reading translation model from %s...\n" % (filename,))
    tm = {}
    fp = gzip.open(filename) if filename[-3:] == '.gz' else open(filename)
    for line in fp.readlines():
        # (f, e, several_logprob_str) = line.strip().split(" ||| ")
        (f, e, several_logprob_str)= line.strip().split(" ||| ")[0:3]
        tm.setdefault(tuple(f.split()), []).append(phrase(e, tuple([float(x) for x in several_logprob_str.strip().split()[0:4]]) ))
    for f in tm: # prune all but top k translations
        # print tm[f]
        tm[f].sort(key=lambda x: -getDotProduct( x.several_logprob ) )
        del tm[f][k:] 
    return tm

# # A language model scores sequences of English words, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = models.LM(filename)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for word in sentence.split():
#   (lm_state, word_logprob) = lm.score(lm_state, word)
#   logprob += word_logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")

class LM:
    def __init__(self, filename, mute=1):
        if mute == 0:
            sys.stderr.write("Reading language model from %s...\n" % (filename,))
        self.table = defaultdict(lambda: ngram_stats(0, 0))
        fp = gzip.open(filename) if filename[-3:] == '.gz' else open(filename)
        for line in fp:
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))
                self.table[ngram] = ngram_stats(logprob, backoff)

    def begin(self):
        return ("<s>",)

    def score(self, state, word):
        ngram = state + (word,)
        score = 0.0
        while len(ngram)> 0:
            if ngram in self.table:
                return (ngram[-2:], score + self.table[ngram].logprob)
            else: #backoff
                # print ngram[:-1]
                score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
                ngram = ngram[1:]
        return ((), score + self.table[("<unk>",)].logprob)
        
    def end(self, state):
        return self.score(state, "</s>")[1]

