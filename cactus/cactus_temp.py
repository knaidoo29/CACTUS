import numpy as np
import shift


class CaCTus:

    def __init__(self):
        """Initialises class."""
        # setup
        self.boxsize = None
        self.ngrid = None
        self.outpath = None
        # grids
        self.x3d = None
        self.y3d = None
        self.z3d = None
        self.kx3d = None
        self.ky3d = None
        self.kz3d = None
        # density field
        self.dens = None

    def setup(self, boxsize, ngrid, outpath='', method='NEXUS+'):
        self.boxsize = boxsize
        self.ngrid = ngrid
        self.outpath = outpath

    def _get_real_grid(self):
        self.x3d, self.y3d, self.z3d = shift.cart.grid3D(self.boxsize, self.ngrid)

    def _get_fourier_grid(self):
        self.kx3d, self.ky3d, self.kz3d = shift.cart.kgrid3D(self.boxsize, self.ngrid)
        self.kmag = np.sqrt(self.kx3d**2. + self.ky3d**2. + self.kz3d**2.)

    def input_dens(self, dens):


    def clean(self):
        """Re-initialises class."""
        self.__init__()
