import optparse, sys, os
from collections import namedtuple, defaultdict
from math import log

def read_ds_from_file(filename):
    import pickle
    import gzip
    pfile = None
    dataStructure = None
    try:
        pfile = gzip.open(filename, 'rb') if filename[-3:] == '.gz' \
                else open(filename, 'rb')
    except:
        dataStructure = None
    try:
        dataStructure = pickle.load(pfile)
    except:
        dataStructure = None
    if pfile is not None:
        pfile.close()
    return dataStructure


def write_ds_to_file(dataStructure, filename):
    import pickle
    import gzip
    output = gzip.open(filename, 'wb') if filename[-3:] == '.gz' \
             else open(filename, 'wb')
    pickle.dump(dataStructure, output)
    output.close()


def init(tFileName):
    sys.stderr.write("Reading dump dictionary %s for ibm model ... \n" % tFileName)
    d = read_ds_from_file(tFileName)
    if d is None:
        sys.stderr.write("There is no such file %s to initialize ibm model dictionary" % tFileName)
        exit(1)
    sys.stderr.write("Finish reading dump dictionary, len=" + str(len(d)) + '\n')
    return d


init_prob = 1.0 / 30.0
def t_f_given_e(t, fw, ew):
    if (fw, ew) in t:
        return t[fw, ew]
    else:
        return init_prob
def q_j_given_i_l_m(q, j, i, l, m):
    if (j, i, l, m) in q:
        return q[j, i, l, m]
    else:
        return init_prob

epsi = 1.0
def ibm_model_1_w_score(t, f, ew):
    score = 0
    for fw in f:
        score += log(t_f_given_e(t, fw, ew))
    return score
def ibm_model_1_score(t, f, e):
    l = float(len(e))
    m = float(len(f))
    score = log(epsi)
    for i in len(f):
        score -= log(l+1)
    for fw in f:
        fw_sum = 0
        for ew in e:
            fw_sum += t_f_given_e(t, fw, ew)
        score += log(fw_sum)
    return score


