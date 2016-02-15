# CountMinSketch

This is a python implementation of the count min sketch algorithm. The original implementation is done as a programming [assignment](http://www.cs.rice.edu/~as143/COMP441_Spring16/Assignment/Assignment1.pdf) for the class COMP441 offered at Rice University instructed by Professor [Anshumali Shrivastava](http://www.cs.rice.edu/~as143/).

##### Usage
To construct an CountMinSketch object, you need to supply three parameters.
```python
depth = 8
width = 2**22
hash_functions = [hash_function(i) for i in range(DEPTH)]
sketch = CountMinSketch(depth, width, hash_functions)
```

Notice that under the hood, the hashtable is implemented as a numpy matrix. This provide a rather simple way run multiple CountMinSketch objects in parallel and merge the results together.

##### Example
In main.py, I include an example of using 8 CountMinSketch objects to count the word frequency of an entire wikipedia dump. The dump is preprocessed into 8 chunks using a modified version of [WikiExtractor](https://github.com/attardi/wikiextractor)(added functionality to output timestamp information). The preprocessing is necessary because the hashing is very fast and the task is mostly IO bound. The speed up gained from the parallelism is mainly from the parallel reading of the dump.