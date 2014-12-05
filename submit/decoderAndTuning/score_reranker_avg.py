#!/usr/bin/env python
import optparse
import sys
import bleu

def main(opts, sysstdin):
	system = [line.strip().split() for line in sysstdin]
	score = 0.0
	ref = [line.strip().split() for line in open(opts.en0)]
	sys.stderr.write('socre for en0: ' + str(cal_store(ref, system)) + '\n')
	score += cal_store(ref, system)
	ref = [line.strip().split() for line in open(opts.en1)]
	sys.stderr.write('socre for en1: ' + str(cal_store(ref, system)) + '\n')
	score += cal_store(ref, system)
	ref = [line.strip().split() for line in open(opts.en2)]
	sys.stderr.write('socre for en2: ' + str(cal_store(ref, system)) + '\n')
	score += cal_store(ref, system)
	ref = [line.strip().split() for line in open(opts.en3)]
	sys.stderr.write('socre for en3: ' + str(cal_store(ref, system)) + '\n')
	score += cal_store(ref, system)
	return score/4


def cal_store(ref, system):
	stats = [0 for i in xrange(10)]
	for (r,s) in zip(ref, system):
		stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]
	return bleu.bleu(stats)

if __name__ == "__main__":
	optparser = optparse.OptionParser()
	# ptparser.add_option("--ref", dest="en", default="/usr/shared/CMPT/nlp-class/project/toy/train.cn", help="target language references for learning how to rank the n-best list")
	optparser.add_option("--en0", dest="en0", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.en0", help="target language references for learning how to rank the n-best list")
	optparser.add_option("--en1", dest="en1", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.en1", help="target language references for learning how to rank the n-best list")
	optparser.add_option("--en2", dest="en2", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.en2", help="target language references for learning how to rank the n-best list")
	optparser.add_option("--en3", dest="en3", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.en3", help="target language references for learning how to rank the n-best list")
	(opts,_) = optparser.parse_args()
	score = main(opts, sys.stdin)
	print score
	sys.stderr.write('average BLEU socre: ' + str(score) + '\n')


