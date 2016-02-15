import multiprocessing as mp
import re
from datetime import datetime
from countminsketch import CountMinSketch
from gpu_countminsketch import GPUCountMinSketch
from hashfactory import hash_function
from hashfactory import gpu_hash_function
import numpy as np
import time
import random
import pickle
from collections import Counter
from collections import defaultdict

TIMESTEMP_RE = re.compile(r'timestamp=\"(.*)\"')
WORD_RE = re.compile(r'\w+')

DEPTH = 8
WIDTH = 2**22
HASH_FUNCTIONS = [hash_function(i) for i in range(DEPTH)]


def worker(index, path):
    global counter
    """
    :param index: the index of the dump this worker should work on.
    :return:
    """
    print "Process %d start processing" % index
    with open("%s/wiki_0%s" % (path, index), "r") as f:
        batch = Counter()
        batch_limit = 10000
        sketch = CountMinSketch(DEPTH, WIDTH, HASH_FUNCTIONS)
        current = datetime.now().date()
        for line in f:
            # Extrat timestamp from header
            if line[:4] == "<doc":
                m = TIMESTEMP_RE.search(line)
                if m:
                    current = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").date()
                continue
            elif line[:5] == "</doc>":
                continue
            else:
                for pair in map(lambda word: (current, word.lower()), WORD_RE.findall(line)):
                    batch[pair] += 1
            if len(batch) > batch_limit:
                for key, count in batch.iteritems():
                    sketch.add(key, count)
                batch.clear()

            counter.value += 1
            if counter.value % 10000 == 0:
                print "Processed %s lines" % counter.value

        for key, count in batch.iteritems():
            sketch.add(key, count)
        batch.clear()

    print "Process %d finished" % index
    return sketch.get_matrix()

if __name__ == '__main__':
    global counter

    numthreads = 8

    counter = mp.Value('i', 0)
    pool = mp.Pool(processes=numthreads)

    start_time = time.time()
    results = pool.map(worker, range(numthreads))

    print "--- %s seconds ---" % (time.time() - start_time)
    print len(results)
    print "Reducing matrices"

