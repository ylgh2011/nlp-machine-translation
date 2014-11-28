from __future__ import division

class Pdist(dict):
    def __init__(self, filename):
        self.maxlen = 0
        self.sumword = 0
        sep = "    "
        for line in file(filename):
            key, freq = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Error in Reading {}".format(filename))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(self.maxlen, len(utf8key))
            self.sumword += int(freq)


    def __call__(self, key, p1 = None, p2 = None):
        if key in self:
            words = key.split()
            if len(words) == 1:
                return self[key]/self.sumword
            if len(words) == 2:
                p1n = p1.get(words[0], 0)
                if p1n == 0:
                    return 0
                return self[key]/p1n
            p2n = p2.get(u"{} {}".format(words[0], words[1]), 0)
            if p2n == 0:
                return 0
            return self[key]/p2n
        else:
            return False

    def default(self, word):
        l = len(word)
        if l == 1:
            return 0.5/self.sumword
        else:
            return False
