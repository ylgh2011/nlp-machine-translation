import sys
import glob
import errno
from collections import defaultdict
import codecs
path1 = '/usr/shared/CMPT/nlp-class/project/seg/cityu_train.utf8'
path2 = '/usr/shared/CMPT/nlp-class/project/seg/msr_train.utf8'
path3 = '/usr/shared/CMPT/nlp-class/project/seg/upenn_train.utf8'
BOI_path_list = [path1, path2, path3]

dev_path = '/usr/shared/CMPT/nlp-class/project/dev/all.cn-en.cn'
test_path = '/usr/shared/CMPT/nlp-class/project/large/train.cn'
article_path_list = [dev_path, test_path]

unigram = defaultdict(int)
bigram = defaultdict(int)
trigram = defaultdict(int)

def trace_sentence(s):
    if (len(s) == 2):
        return
    for i in range(len(s)):
        unigram[s[i]] += 1
        if i >= 1:
            key = "{} {}".format(s[i-1], s[i])
            bigram[key] += 1
        if i >= 2:
            key = "{} {} {}".format(s[i-2], s[i-1], s[i])
            trigram[key] += 1
    return

for path in BOI_path_list:
    with open(path) as f:
        word = ""
        sentence = []
        cnt = 0
        for line in f:
            match = line.strip().split()
            if len(match) <= 1:
                trace_sentence(sentence)
                if cnt % 500 == 0:
                    sys.stderr.write('.')
                    cnt = 0
                cnt += 1
                sentence = []
                continue
            if match[1] == 'B' or match[1] == 'O':
                if len(word) <= 0:
                    continue
                sentence.append(word)
                word = match[0]
            if match[1] == 'I':
                word += match[0]
    sys.stderr.write('\n')

for path in article_path_list:
    with open(path) as f:
        cnt = 0
        for line in f:
            sentence = line.strip().split()
            sentence[:0] = []
            trace_sentence(sentence)
            if cnt % 500 == 0:
                sys.stderr.write('.')
                cnt = 0
            cnt += 1
    sys.stderr.write('\n')


uni_out = codecs.open("./grams/unigram.txt", "w", "utf-8")
for key in unigram:
    uni_out.write(u"{}    {}\n".format(unicode(key, "utf-8"), unigram[key]))
sys.stderr.write('Unigram finished!\n')

bi_out = codecs.open("./grams/bigram.txt", "w", "utf-8")
for key in bigram:
    bi_out.write(u"{}    {}\n".format(unicode(key, "utf-8"), bigram[key]))
sys.stderr.write('Bigram finished!\n')

tri_out = codecs.open("./grams/trigram.txt", "w", "utf-8")
for key in trigram:
    tri_out.write(u"{}    {}\n".format(unicode(key, "utf-8"), trigram[key]))
sys.stderr.write('Trigram finished!\n')
