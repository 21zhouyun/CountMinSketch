import random
import hashlib

_memomask = {}


def hash_function(n):
    """
    :param n: the index of the hash function
    :return: a generated hash function
    """
    mask = _memomask.get(n)

    if mask is None:
        random.seed(n)
        mask = _memomask[n] = random.getrandbits(32)

    def my_hash(x):
        return hash(str(x) + str(n)) ^ mask

    return my_hash


def gpu_hash_function(j, rand):
    """
    This is a python duplicate of the string hash function used
    in gpu_countminsketch.py
    :param j: the index of the hash function
    :param rand: a list of generated random numbers, must be at least as long as j
    :return: a generated hash function
    """
    def my_hash(s):
        value = rand[j]

        for c in s:
            value = (((value << 5) + value) + ord(c)) % 2**32

        return value

    return my_hash
