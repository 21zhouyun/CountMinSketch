import numpy as np
import pyopencl as cl
from collections import Counter


class GPUCountMinSketch(object):
    """
    A GPU implementation of the count min sketch algorithm.
    It will batch input strings into a list until the buffer
    reaches a limit. Then it will invoke the opencl kernel
    to calculate the hashes.
    """

    def __init__(self, d, w, batch_limit, rand, hash_functions, M=None):
        """
        :param d: the depth of the sketch
        :param w: the width of the sketch
        :param batch_limit: size limit of the batch buffer
        :param kernel: the kernel code for hashing the given data.
        :param M: provided matrix of counts
        :return:
        """
        self.d = d
        self.w = w
        self.rand = rand
        self.hash_functions = hash_functions
        self.batch_limit = batch_limit
        self.batch = Counter()
        if M is not None:
            self.M = M
        else:
            self.M = np.zeros([d, w], dtype=np.int32)

        # Initialize kernel
        self.ctx = cl.create_some_context()

        self.queue = cl.CommandQueue(self.ctx)

        # kernel code
        code = """
        #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

        __kernel void increment(__global const unsigned int* rand, __global char* keys, __global int* counts, __global int* out)
        {
            int i = get_global_id(0);
            int j = get_global_id(1);

            int width = %d;
            int str_size = %d;

            // calculate hash inline
            unsigned int hash = rand[j];
            char c;

            for(int k = 0; k < str_size; k++){
                c = keys[i*str_size + k];
                if (c != 0) {
                    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
                }
            }
            long index = hash %% width;

            atomic_add(&out[j * width + index], counts[i]);
        }
        """ % (self.w, 32)
        # build the Kernel
        self.bld = cl.Program(self.ctx, code).build()

    def add(self, x, delta=1):
        if len(self.batch) < self.batch_limit:
            self.batch[x] += delta
        else:
            self.dump_batch()

    def dump_batch(self):
        keys = np.array(self.batch.keys(), dtype='S32')
        counts = np.array(self.batch.values(), dtype=np.int32)
        out = np.zeros([self.d, self.w], dtype=np.int32)

        # create the buffers to hold the values of the input
        rand_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.rand)
        keys_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=keys)
        counts_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=counts)

        # create output buffer
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, out.nbytes)

        # Kernel is now launched
        launch = self.bld.increment(self.queue, (len(keys), self.d), None, rand_buf, keys_buf, counts_buf, out_buf)
        # wait till the process completes
        launch.wait()

        cl.enqueue_read_buffer(self.queue, out_buf, out).wait()

        self.M += out
        self.batch.clear()

    def query(self, x):
        return min([self.M[i][self.hash_functions[i](x) % self.w] for i in range(self.d)])

    def get_matrix(self):
        return self.M

