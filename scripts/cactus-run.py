import sys
import os.path
import numpy as np

import cactus
from cactus.ext import mpiutils

MPI = mpiutils.MPI()

yaml_fname = str(sys.argv[1])

CCT = cactus.main.CaCTus(MPI)

CCT.run(yaml_fname)
