#!/usr/bin/env python
import optparse, sys, os, logging, copy
from collections import defaultdict
import pickle
import gzip

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="/usr/shared/CMPT/nlp-class/project/medium", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="train", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="cn", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
sys.stderr.write("Training with Baseline method coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


def main():
    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    # Initialization step
    voc_f = defaultdict()
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            voc_f[f_i] = 1

    init_prob = 1.0 / float(len(voc_f.keys()))
    t = defaultdict(float)

    for iter_cnt in range(5):
        sys.stderr.write("\nTraining")
        # inherit last iteration

        # init count
        fe_count = defaultdict(float)
        e_count = defaultdict(float)
        for (n, (f, e)) in enumerate(bitext):
            for f_i in f:
                norm_z = 0
                for e_j in e:
                    norm_z += t.get((f_i, e_j), init_prob)

                for e_j in e:
                    cnt = t.get((f_i, e_j), init_prob)/norm_z
                    fe_count[f_i, e_j] += cnt
                    e_count[e_j] += cnt

            # process indicator
            if n % 500 == 0:
                sys.stderr.write(".")

        # clean up t_prev
        t = defaultdict(float)
        sys.stderr.write("\nAssigning variable")
        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            t[f_i, e_j] = fe_count[f_i, e_j]/e_count[e_j]
            if k % 5000 == 0:
                sys.stderr.write(".")


    sys.stderr.write("\nDumping\n")
    output = gzip.open('ibm1.ds', 'wb')
    pickle.dump(t, output)
    output.close()

if __name__ == "__main__":
    main()
