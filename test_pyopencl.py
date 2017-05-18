#!/usr/bin/env python3

from pyopencl import cl

if __name__ == "__main__":
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    prg = cl.Program(ctx, """
        __kernel void sq(__global const float *a, __global float *c) 
        {
            int gid = get_global_id(0);
            c[gid] = a[gid] * a[gid];
        }
        """).build()

    import ipdb; ipdb.set_trace()
