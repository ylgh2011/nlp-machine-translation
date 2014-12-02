#!/usr/bin/env python
import optparse
import sys
import bleu

def main(opts, sysstdin):
	ref = [line.strip().split() for line in open(opts.en)]
	system = [line.strip().split() for line in sysstdin]

	stats = [0 for i in xrange(10)]
	for (r,s) in zip(ref, system):
	    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]
	
	return bleu.bleu(stats)

if __name__ == "__main__":
	optparser = optparse.OptionParser()
	# ptparser.add_option("--ref", dest="en", default="/usr/shared/CMPT/nlp-class/project/toy/train.cn", help="target language references for learning how to rank the n-best list")
	optparser.add_option("--en", dest="en", default="/usr/shared/CMPT/nlp-class/project/test/all.cn-en.en0", help="target language references for learning how to rank the n-best list")
	(opts,_) = optparser.parse_args()
	print main(opts, sys.stdin)

