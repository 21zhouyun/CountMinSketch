# CountMinSketch

This is a python implementation of the count min sketch algorithm. The original implementation is done as a programming [assignment](http://www.cs.rice.edu/~as143/COMP441_Spring16/Assignment/Assignment1.pdf) for the class COMP441 offered at Rice University instructed by Professor [Anshumali Shrivastava](http://www.cs.rice.edu/~as143/).

##### Usage
To construct an CountMinSketch object, you need to supply three parameters.
```python
from hashfactory import hash_function

depth = 8
width = 2**22
hash_functions = [hash_function(i) for i in range(depth)]
sketch = CountMinSketch(depth, width, hash_functions)
```

Notice that under the hood of this implementation is a numpy matrix. This provide a rather simple way run multiple CountMinSketch objects in parallel and merge the results together.

##### Example
In main.py, I include an example of using 8 CountMinSketch objects to count the word frequency of an entire wikipedia dump. The dump is preprocessed into 8 chunks using a modified version of [WikiExtractor](https://github.com/attardi/wikiextractor)(added functionality to output timestamp information). The preprocessing is necessary because the hashing is very fast and the task is mostly IO bound. The speed up gained from the parallelism is mainly from the parallel reading of the dump.

##### Other files:
######gpu_countminsketch.py:

I tried to leverage the massive parallelism provided by a GPU to speed up the hashing process. There are two main concerns:

1. Since the hashing function need access to entire M matrix, this is not a typical problem that can be chunked and feed to a GPU and get massive speedup. As you can see in the opencl kernel I wrote, I used an atomic_add to solve the race problem. However, if the collision rate is high, the GPU implementation may degenerate into a slower sequential version of CMS.
 
2. Since calculating a hash function is not very time consuming, the time used to copy the memory from CPU to GPU may be the dominating factor. In fact, in my own experient, the GPU implementation is slower than a single thread CPU implementation on the same data.

######hashfactory.py:

A simple utility that generate hash functions. For the gpu_hash_function, you need to store the random seed you used to generate the rand parameter. You will need it when querying the generated sketch.
