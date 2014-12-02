from __future__ import division
import sys, codecs
from math import log
from collections import defaultdict

from entry import Entry
from pdist import Pdist

sys.stderr.write("Reading dictionarys...\n")
p1w = Pdist('./grams/unigram.txt')
sys.stderr.write("Unigram finished!\n")
p2w = Pdist('./grams/bigram.txt')
sys.stderr.write("Bigram finished!\n")
p3w = Pdist('./grams/trigram.txt')
sys.stderr.write("Trigram finished!\n")

lam3 = 0.618
lam2 = (1 - lam3) * lam3
lam1 = 1 - lam3 - lam2
word_len = p1w.maxlen

source_path = '/usr/shared/CMPT/nlp-class/project/test/all.cn-en.cn'
#source_path = './train.cn.unseg'
seg_out = codecs.open("./test.cn", "w", "utf-8")


def back_trace(entry):
    if entry.word == "<s>":
        return ""
    return u"{} {}".format(back_trace(entry.back_pnt).strip(), entry.word)


def count_prob(word, p):
    sum1 = p1w(word)
    if sum1 is False:
        sum1 = p1w.default(word)
        if sum1 == False:
            return False
        return sum1 * lam1
    sum1 *= lam1

    word2 = u"{} {}".format(p.word, word)
    sum2 = p2w(word2, p1 = p1w)
    if sum2 is False:
        return sum1
    sum2 = sum2 * lam2 + sum1

    pp = p.back_pnt
    word3 = u"{} {}".format(pp.word, word2)
    sum3 = p3w(word3, p2 = p2w)
    if sum3 is False:
        return sum2
    return sum3 * lam3 + sum2

sys.stderr.write("Start working on segmentation...\n")
with open(source_path) as f:
    cnt = 0
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')
        utf8line.replace(' ', '')
        liness = Entry("<ss>", 0.0, None)
        lines = Entry("<s>", 0.0, liness)
        entry_list = [liness, lines]

        for index in range(len(utf8line)):
            max_sum = float('-Inf')
            max_ety = 1
            max_word = utf8line[: index + 1]

            for i in range(index , max(-1, index - word_len), -1):
                word = utf8line[i : index + 1]
                this_log = count_prob(word, entry_list[i+1])
                if (this_log == False):
                    continue
                sum_log = entry_list[i+1].logP + log(this_log)
                # sys.stderr.write(u"{}: {}? {}\n".format(word, sum_log, p1w.get(word, 0)))
                if sum_log > max_sum:
                    max_sum = sum_log
                    max_ety = i + 1
                    max_word = word

            entry_list.append(Entry(max_word, max_sum, entry_list[max_ety]))
        sentence = u"{}\n".format(back_trace(entry_list[-1]).strip())
        seg_out.write(sentence)
        if cnt%500 == 0:
            sys.stderr.write('.')
            cnt = 0
        cnt += 1
sys.stderr.write("Finished!\n")
