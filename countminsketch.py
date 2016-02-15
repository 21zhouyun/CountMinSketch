import numpy as np


class CountMinSketch(object):
    """
    A non GPU implementation of the count min sketch algorithm.
    """
    def __init__(self, d, w, hash_functions, M=None):
        self.d = d
        self.w = w
        self.hash_functions = hash_functions
        if len(hash_functions) != d:
            raise ValueError("The number of hash functions must match match the depth. (%s, %s)" % (d, len(hash_functions)))
        if M is None:
            self.M = np.zeros([d, w], dtype=np.int32)
        else:
            self.M = M

    def add(self, x, delta=1):
        for i in range(self.d):
            self.M[i][self.hash_functions[i](x) % self.w] += delta

    def batch_add(self, lst):
        pass

    def query(self, x):
        return min([self.M[i][self.hash_functions[i](x) % self.w] for i in range(self.d)])

    def get_matrix(self):
        return self.M
