import sys
import glob
import errno
from collections import defaultdict
import json
import codecs
path1 = '/usr/shared/CMPT/nlp-class/project/seg/cityu_train.utf8'   
path2 = '/usr/shared/CMPT/nlp-class/project/seg/msr_train.utf8'   
path3 = '/usr/shared/CMPT/nlp-class/project/seg/upenn_train.utf8'   
path4 = 'cityu_train.utf8'

path_list = [path1, path2, path3]
#path_list = [path4]
bigram = defaultdict(int)

for path in path_list:
	with open(path) as f:
		prevword = "<s>"
		word = ""
		for line in f:
			if len(line.split()) <= 1:
				if len(word) <= 0:
					word = word + line.split()[0]
					continue
				if len(prevword) != 0:
					key = prevword + " " + word
					bigram[key] += 1
				prevword = word
				#word = ""
				word = '<\s>'

			elif line.split()[1] == 'B' or line.split()[1] == 'O':
				#print word
				if len(word) <= 0:
					word = word + line.split()[0]
					continue
				if len(prevword) != 0:
					if prevword == '<\s>':
						prevword = '<s>'
					key = prevword + " " + word
					bigram[key] += 1
				prevword = word
				word = ""
				word = word + line.split()[0]
			elif line.split()[1] == 'I':
				word = word + line.split()[0]
	print "One file completed"

a = codecs.open("bigram.txt", "w", "utf-8")
for key in bigram:
	a.write(unicode(key, "utf-8"))
	a.write("	")
	#print unigram[key]
	a.write(str(bigram[key]))
	a.write('\n')
	# codecs.open("unigram.txt", "w", "utf8").write(" ")
	# codecs.open("unigram.txt", "w", "utf8").write(unigram[key])

#json.dump(unigram, open("unigram.txt",'w'))
