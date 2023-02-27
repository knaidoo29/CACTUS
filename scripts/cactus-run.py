import sys
import os.path
import numpy as np

import cactus
import shift
import fiesta
import mpiutils

MPI = mpiutils.MPI()

yaml_fname = str(sys.argv[1])

CCT = cactus.main.CaCTus(MPI)
# CCT.start()
# CCT.read_paramfile(yaml_fname)
# CCT.prepare()
# # CCT.read_particles()
# # CCT.calculate_density()
# CCT._load_dens()
# CCT.end()

CCT.run(yaml_fname)

# import mistree as mist
#
# size = 10000
# x, y, z = mist.get_levy_flight(size, mode='3D')

# np.savez('lf_'+str(MPI.rank)+'.npz', x=x, y=y, z=z)
#
# np.savetxt('levy_flight_'+str(MPI.rank)+'.dat', np.column_stack([x, y, z]), header="X, Y, Z")
