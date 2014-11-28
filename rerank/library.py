#!/usr/bin/env python
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




def untranslatedWords(f, e, std_e):
    count = 0
    for word in e.split():
        if (word not in std_e) and (word in f):
            count += 1

    return -1 * count


def ss(s):
    return s.strip().split()


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
def ibm_model_1_score(t, f, e):
    l = float(len(ss(e)))
    m = float(len(ss(f)))
    score = log(epsi / float((l+1)**len(ss(f))))
    for fw in ss(f):
        fw_sum = 0
        for ew in ss(e):
            fw_sum += t_f_given_e(t, fw, ew)
        score += log(fw_sum)
    return score

def ibm_model_2_score(t, q, f, e):
    l = float(len(ss(e)))
    m = float(len(ss(f)))
    score = log(epsi / float((l+1)**len(ss(f))))
    for i, fw in enumerate(ss(f)):
        fw_sum = 0
        for j, ew in enumerate(ss(e)):
            fw_sum += t_f_given_e(t, fw, ew)*q_j_given_i_l_m(q, j, i, l, m)
        score += log(fw_sum)
    return score    


Expected_number_of_features = 9
def pre_process(fileNameList):
    shouldInit_this_item = []
    # sys.stderr.write(str(fileNameList))

    for item in fileNameList:
        frFileName = item[0]
        nBestFileName = item[1]
        enFileName = item[2]
        sys.stderr.write(frFileName +', ' + nBestFileName + ', ' + enFileName + ': ')
        for line in file(nBestFileName):
            (i, sentence, features) = line.strip().split("|||")
            if len(features.strip().split()) == Expected_number_of_features:
                shouldInit_this_item.append(0)
                sys.stderr.write("0\n")
            else:
                shouldInit_this_item.append(1)
                sys.stderr.write("1\n")
            break
        # shouldInit_this_item.append(1)

    if sum(shouldInit_this_item) > 0:
        t = init('./data/ibm2.t.ds.gz')
        q = init('./data/ibm2.q.ds.gz')
        for i, item in enumerate(fileNameList):
            sys.stderr.write(str(item) + '\n')
            frFileName = item[0]
            nBestFileName = item[1]
            enFileName = item[2]
            sys.stderr.write(frFileName +', ' + nBestFileName + ', ' + enFileName + '\n')
            if shouldInit_this_item[i] == 1:
                real_pre_process(frFileName, nBestFileName, enFileName, t, q)
            else:
                sys.stderr.write("Number of features is already %d, so %s is not updated\n" % (Expected_number_of_features, nBestFileName))
    else:
        sys.stderr.write("No .nbest files need to be updated\n")


def real_pre_process(frFileName, nBestFileName, enFileName, t, q):
    sys.stderr.write("Pre-processing new features for %s: (sentence length, number of untranslated words, IBM model 1 score, IBM model 2 score)\n" % nBestFileName)
    frenches = []
    englishes = []
    nBestLines = []
    for line in file(frFileName):
        frenches.append(line)
    for line in file(enFileName):
        englishes.append(line)

    # initialization ibm model t dictionary and q dictionary

    newlines = []
    sys.stderr.write("Calculating new features for %s ... \n" % nBestFileName)
    for j, line in enumerate(file(nBestFileName)):
        if j % 2000 == 1:
            sys.stderr.write('.')

        (i, sentence, features) = line.strip().split("|||")
        feature_list = [x for x in ss(features)][0:6]

        index = int(i) if int(i) < len(englishes) else len(englishes) - 1
        feature_list.append(str(-1 * abs(len(ss(sentence)) - len(ss(englishes[index])))))
        feature_list.append(str(untranslatedWords(frenches[int(i)], sentence, englishes[index])))
        # feature_list.append(str(ibm_model_1_score(t, frenches[int(i)], sentence))) # IBM model 1 score
        feature_list.append(str(ibm_model_2_score(t, q, frenches[int(i)], sentence))) # IBM model 2 score

        newlines.append(i + '|||' + sentence + '||| ' + ' '.join(feature_list) + '\n')

    # print newlines
    sys.stderr.write("\nWriting new content to %s ... \n" % nBestFileName)
    nBestFile = open(nBestFileName, 'wb')
    nBestFile.writelines(newlines)
    nBestFile.close()
    sys.stderr.write("Finish writing %s\n" % nBestFileName)


if __name__ == '__main__':
   pre_process('./data/test.fr', './data/test.nbest')

