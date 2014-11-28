import sys
import glob
import errno
from collections import defaultdict
import json
import codecs
path1 = '/usr/shared/CMPT/nlp-class/project/seg/cityu_train.utf8'   
path2 = '/usr/shared/CMPT/nlp-class/project/seg/msr_train.utf8'   
path3 = '/usr/shared/CMPT/nlp-class/project/seg/upenn_train.utf8'   
path_list = [path1, path2, path3]
unigram = defaultdict(int)
bigram = defaultdict(int)

for path in path_list:
	with open(path) as f:
		prevword = ""
		word = ""
		for line in f:
			if len(line.split()) <= 1:
				continue
			if line.split()[1] == 'B' or line.split()[1] == 'O':
				#print word
				if len(word) <= 0:
					continue
				unigram[word] += 1
				word = ""
				word = word + line.split()[0]
			if line.split()[1] == 'I':
				word = word + line.split()[0]
	print "One file completed"

a = codecs.open("unigram.txt", "w", "utf-8")
for key in unigram:
	a.write(unicode(key, "utf-8"))
	a.write("	")
	#print unigram[key]
	a.write(str(unigram[key]))
	a.write('\n')
	# codecs.open("unigram.txt", "w", "utf8").write(" ")
	# codecs.open("unigram.txt", "w", "utf8").write(unigram[key])

#json.dump(unigram, open("unigram.txt",'w'))
