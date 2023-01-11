import sys
import os.path
import numpy as np

#import shift
#import fiesta
import mpiutils


MPI = mpiutils.MPI()

val = MPI.send_down(MPI.rank)

MPI.mpi_print('Rank ', MPI.rank, val)
