#!/usr/bin/env python3

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":
    if rank == 0:
        for i in range(0, 10):
            for proc_id in range(1, size):
                data = {'player': 0, 'state': [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0]}
                req = comm.isend(data, dest=proc_id, tag=11)

        for i in range(0, 10):
            req = comm.irecv(tag=12)
            data = req.wait()
            print(f'Root, Received:', data)

        for proc_id in range(1, size):
            req = comm.isend(None, dest=proc_id, tag=11)
    else:
        while True:
            req = comm.irecv(source=0, tag=11)
            data = req.wait()
            if not data:
                break
            print(f'Worker {rank}, Received:', data)
            time.sleep(2)
            req = comm.isend(rank+10, dest=0, tag=12)

