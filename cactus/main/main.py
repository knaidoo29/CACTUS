import os
import time

import numpy as np

from scipy.interpolate import interp1d

from ..ext import fiesta, magpie, shift

from . import files
from . import read
from . import inout
from .. import src


"""
The CaCTus 'graphics' are inspired/based on the ascii artwork:

    ,*-.
    |  |
,.  |  |
| |_|  | ,.
`---.  |_| |
    |  .--`
    |  |
    |  | kra

"""

cactus_beg = """
 ______________________________________________________________________________
|                                                                              |
|                                        ,*-.                                  |
|                                        |  |                                  |
|                                    ,.  |  |                                  |
|                    ,.__       _*__ | |_|  | ,.                               |
|                   / ___|__ _ / ___|`---.  |_| | _   * ___                    |
|                  | |   , _` | |        |  .--` | | | / *_|                   |
|                  |  *_| (_| | |___     |  |    | *_| \__ \                   |
|___________________\____\__,_|\____|____|  |_____\__,_|___/___________________|
|                                                                              |
|                      Cosmic web Classification Toolkit                       |
|______________________________________________________________________________|

"""


cactus_end = """
 ______________________________________________________________________________
|                                                                              |
|           ,*.                                                                |
|        ,. | |                                                                |
|        ||_| | ,.                                          ,*.                |
|        `--. |_||                                       /|_| | ,.             |
|           | .--`                                       `--. |_||             |
|___________| |_____________________________________________| .--`_____________|
|                                                                              |
|                                 Finished                                     |
|______________________________________________________________________________|
"""


class CaCTus:


    def __init__(self, MPI):
        """Initialise the CaCTus main class."""
        # Global variables
        self.MPI = MPI
        self.ERROR = False

        # Time Variables
        self.time = {
            "Start": None,
            "Particle_Start": None,
            "Particle_End": None,
            "Density_Start": None,
            "Density_End": None,
            "CosmicWeb_Start": None,
            "CosmicWeb_End": None,
            "End": None
        }
        # Parameters
        self.params = None
        # Cosmology
        self.cosmo = {
            "H0": None,
            "Omega_m": None
            }
        # Siminfo
        self.siminfo = {
            "Boxsize": None,
            "Ngrid": None,
            "x3D": None,
            "y3D": None,
            "z3D": None,
            "avgmass": None
        }
        # Switch
        self.what2run = {
            "particles": False,
            "density": False,
            "cosmicweb": False
        }
        # Particles
        self.particles = {
            "Type": None,
            "Fileslist": None,
            "NFiles": None,
            "Fnames": None,
            "ASCII_Columns": None,
            "NPZ_Keys": None,
            "Gadget_Factor": None,
            "x": None,
            "y": None,
            "z": None,
            "npart": None,
            "mass": None
        }
        # Density
        self.density = {
            "Type": None,
            "MPI_Split": None,
            "Buffer_Type": None,
            "Buffer_Length": None,
            "Subsampling": None,
            "Saveas": None,
            "dens": None,
            "mass": None,
        }
        # CosmicWeb
        self.cosmicweb = {
            "Density_Prefix": None,
            "Density_Nfiles": None,
            "Filter": {
                "Type": None,
                "R": None,
            },
            "Type": None,
            "Nexus": {
                "Signature": {
                    "_Run": False,
                    "R0": None,
                    "Nmax": None,
                    "Logsmooth": None,
                    "Output": None,
                    "Sc": None, "Sf":None, "Sw":None,
                },
                "Thresholds": {
                    "_Run": False,
                    "SigFile": {
                        "Prefix": None,
                        "Nfiles": None
                    },
                    "Clusters": {
                        "Minmass": None,
                        "Mindens": None,
                        "Minvol": None,
                        "nbins": 100,
                        "Neval": None
                    },
                    "Filaments": {
                        "nbins": 100,
                        "Minvol": None
                    },
                    "Walls": {
                        "nbins": 100,
                        "Minvol": None
                    },
                    "Output": None
                }
            },
            "Vweb": {},
            "Tweb": {},
            "web_flag": None,
        }


    def start(self):
        """Starts the run and timers."""
        self.time["Start"] = time.time()
        self.MPI.mpi_print_zero(cactus_beg)


    def _break4error(self):
        """Breaks the class run if an error is detected."""
        if self.ERROR is True:
            exit()


    def _check_particle_files(self):
        """Check whether particle file exists."""
        for fname in self.particles["Fnames"]:
            if files.check_exist(fname) is False:
                self.MPI.mpi_print_zero(" ERROR: File '"+fname+"' does not exist, aborting.")
                self.ERROR = True
            else:
                self.ERROR = False


    def _construct_particle_fnames(self):
        """Construct particle filenames."""
        if self.particles["Filelist"] is not None:
            if files.check_exist(self.particles["Filelist"]) is False:
                self.MPI.mpi_print_zero(" ERROR: Filelist file does not exist, aborting.")
                self.ERROR = True
            else:
                self.ERROR = False
                with open(self.particles["Filelist"]) as fname:
                    self.particles["Fnames"] = fname.read().splitlines()
                    self.particles["Nfiles"] = len(self.particles["Fnames"])
        else:
            if self.particles["Type"] == "ASCII" or self.particles["Type"] == "NPZ":
                self.particles["Fnames"] = [self.particles["Fileprefix"]+str(n)+self.particles["Filesuffix"] for n in range(0,self.particles["Nfiles"])]
            elif self.particles["Type"] == "Gadget":
                self.particles["Fnames"] = [self.particles["Fname"]+"."+str(n) for n in range(0,self.particles["Nfiles"])]
        self._break4error()
        self._check_particle_files()
        self._break4error()


    def _check_param_key(self, params, key):
        """Check param key exists in dictionary, and if key is not None."""
        if key in params:
            if params[key] is not None:
                return True
            else:
                return False
        else:
            return False


    def read_params(self, params):
        """Reads parameter file."""

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Parameters")
        self.MPI.mpi_print_zero(" ==========")

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" MPI:")
        self.MPI.mpi_print_zero(" -", self.MPI.size, "Processors")

        # Read in Cosmological parameters
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Cosmology:")

        self.cosmo["H0"] = float(params["Cosmology"]["H0"])
        self.cosmo["Omega_m"] = float(params["Cosmology"]["Omega_m"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - H0 \t\t\t=", self.cosmo["H0"])
        self.MPI.mpi_print_zero(" - Omega_m\t\t=", self.cosmo["Omega_m"])

        # Read in Siminfo
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Siminfo:")

        self.siminfo["Boxsize"] = float(params["Siminfo"]["Boxsize"])
        self.siminfo["Ngrid"] = int(params["Siminfo"]["Ngrid"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - Boxsize \t\t=", self.siminfo["Boxsize"])
        self.MPI.mpi_print_zero(" - Ngrid \t\t=", self.siminfo["Ngrid"])

        # Read Particles
        if self._check_param_key(params, "Particles"):
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" Particles:")

            self.particles["Type"] = params["Particles"]["Type"]
            self.particles["Filelist"] = params["Particles"]["Filelist"]

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Type \t\t=", self.particles["Type"])
            self.MPI.mpi_print_zero(" - Filelist \t\t=", self.particles["Filelist"])

            if self.particles["Type"] == "ASCII":
                pass
            elif self.particles["Type"] == "NPZ":
                pass
            elif self.particles["Type"] == "Gadget":
                pass
            else:
                self.MPI.mpi_print_zero(" ERROR: Type must be ASCII, NPZ or Gadget")
                self.ERROR = True
            self._break4error()

            if self.particles["Filelist"] is None:

                if self.particles["Type"] == "ASCII" or self.particles["Type"] == "NPZ":
                    self.particles["Fileprefix"] = params["Particles"]["Fileprefix"]
                    self.particles["Filesuffix"] = params["Particles"]["Filesuffix"]
                    self.particles["Nfiles"] = params["Particles"]["Nfiles"]
                    self.MPI.mpi_print_zero(" - Fileprefix \t\t=", self.particles["Fileprefix"])
                    self.MPI.mpi_print_zero(" - Filesuffix \t\t=", self.particles["Filesuffix"])

                elif self.particles["Type"] == "Gadget":
                    self.particles["Fname"] = params["Particles"]["Fname"]
                    self.particles["Nfiles"] = params["Particles"]["Nfiles"]
                    self.MPI.mpi_print_zero(" - Fname \t\t=", self.particles["Fname"])

            self._construct_particle_fnames()
            self.MPI.mpi_print_zero(" - Nfiles \t\t=", self.particles["Nfiles"])

            if self.particles["Type"] == "ASCII":
                self.particles["ASCII_Columns"] = params["Particles"]["ASCII_Columns"]
                self.MPI.mpi_print_zero(" - ASCII_Columns \t=", self.particles["ASCII_Columns"])

            elif self.particles["Type"] == "NPZ":
                self.particles["NPZ_Keys"] = params["Particles"]["NPZ_Keys"]
                self.MPI.mpi_print_zero(" - NPZ_Keys \t\t=", self.particles["NPZ_Keys"])

            elif self.particles["Type"] == "Gadget":
                self.particles["Gadget_Factor"] = params["Particles"]["Gadget_Factor"]
                self.MPI.mpi_print_zero(" - Gadget_Factor \t=", self.particles["Gadget_Factor"])

            self.what2run["particles"] = True

        # Read Density
        if self._check_param_key(params, "Density"):
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" Density:")

            self.density["Type"] = params["Density"]["Type"]

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Type \t\t=", self.density["Type"])

            if self.density["Type"] == "DTFE":
                self.density["MPI_Split"] = params["Density"]["MPI_Split"]
                self.density["Buffer_Type"] = params["Density"]["Buffer_Type"]
                self.density["Buffer_Length"] = params["Density"]["Buffer_Length"]
                self.density["Subsampling"] = params["Density"]["Subsampling"]

                self.MPI.mpi_print_zero(" - MPI_Split \t\t=", self.density["MPI_Split"])
                self.MPI.mpi_print_zero(" - Buffer_Type \t\t=", self.density["Buffer_Type"])
                self.MPI.mpi_print_zero(" - Buffer_Length \t=", self.density["Buffer_Length"])
                self.MPI.mpi_print_zero(" - Subsampling \t\t=", self.density["Subsampling"])

            self.density["Saveas"] = params["Density"]["Saveas"]
            self.MPI.mpi_print_zero(" - Saveas \t\t=", self.density["Saveas"])

            self.what2run["density"] = True

        # Read CosmicWeb info
        if self._check_param_key(params, "CosmicWeb"):
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" CosmicWeb:")

            self.cosmicweb["Density_Prefix"] = params["CosmicWeb"]["Density_Prefix"]
            self.cosmicweb["Density_Nfiles"] = params["CosmicWeb"]["Density_Nfiles"]

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Density_Prefix\t=", self.cosmicweb["Density_Prefix"])
            self.MPI.mpi_print_zero(" - Density_Nfiles\t=", self.cosmicweb["Density_Nfiles"])

            self.what2run["cosmicweb"] = True

            if self._check_param_key(params["CosmicWeb"], "Filter"):

                self.cosmicweb["Filter"]["Type"] = str(params["CosmicWeb"]["Filter"]["Type"])

                if self._check_param_key(params["CosmicWeb"]["Filter"], "R"):
                    self.cosmicweb["Filter"]["R"] = float(params["CosmicWeb"]["Filter"]["R"])
                else:
                    self.MPI.mpi_print_zero(" ERROR: Must define 'R' for Filter.")
                    self.ERROR = True
                    self._break4error()

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" - Filter")

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Type\t\t\t=", self.cosmicweb["Filter"]["Type"])
                self.MPI.mpi_print_zero(" -- R   \t\t\t=", self.cosmicweb["Filter"]["R"])


            self.cosmicweb["Type"] = params["CosmicWeb"]["Type"]

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Type\t\t\t=", self.cosmicweb["Type"])

            if self.cosmicweb["Type"] == "Nexus":

                if self._check_param_key(params["CosmicWeb"]["Nexus"], "Signature"):

                    self.cosmicweb["Nexus"]["Signature"]["_Run"] = True

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" - Signature")

                    self.cosmicweb["Nexus"]["Signature"]["R0"] = params["CosmicWeb"]["Nexus"]["Signature"]["R0"]
                    self.cosmicweb["Nexus"]["Signature"]["Nmax"] = params["CosmicWeb"]["Nexus"]["Signature"]["Nmax"]
                    self.cosmicweb["Nexus"]["Signature"]["Logsmooth"] = params["CosmicWeb"]["Nexus"]["Signature"]["Logsmooth"]
                    self.cosmicweb["Nexus"]["Signature"]["Output"] = params["CosmicWeb"]["Nexus"]["Signature"]["Output"]

                    self.MPI.mpi_print_zero(" -- R0\t\t\t=", self.cosmicweb["Nexus"]["Signature"]["R0"])
                    self.MPI.mpi_print_zero(" -- Nmax\t\t=", self.cosmicweb["Nexus"]["Signature"]["Nmax"])
                    self.MPI.mpi_print_zero(" -- Logsmooth\t\t=", self.cosmicweb["Nexus"]["Signature"]["Logsmooth"])
                    self.MPI.mpi_print_zero(" -- Output\t\t=", self.cosmicweb["Nexus"]["Signature"]["Output"])

                if self._check_param_key(params["CosmicWeb"]["Nexus"], "Thresholds"):

                    self.cosmicweb["Nexus"]["Thresholds"]["_Run"] = True

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" - Thresholds")

                    self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["SigFile"]["Prefix"]
                    self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"] = int(params["CosmicWeb"]["Nexus"]["Thresholds"]["SigFile"]["Nfiles"])

                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minmass"] = float(params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Minmass"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Mindens"] = float(params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Mindens"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvol"] = float(params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Minvol"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Nbins"] = int(params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Nbins"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Neval"] = int(params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Neval"])

                    self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Nbins"] = int(params["CosmicWeb"]["Nexus"]["Thresholds"]["Filaments"]["Nbins"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Minvol"] = float(params["CosmicWeb"]["Nexus"]["Thresholds"]["Filaments"]["Minvol"])

                    self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Nbins"] = int(params["CosmicWeb"]["Nexus"]["Thresholds"]["Walls"]["Nbins"])
                    self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Minvol"] = float(params["CosmicWeb"]["Nexus"]["Thresholds"]["Walls"]["Minvol"])

                    self.cosmicweb["Nexus"]["Thresholds"]["Output"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Output"]

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- SigFile")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Prefix\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"])
                    self.MPI.mpi_print_zero(" --- Nfiles\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Clusters")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Minmass\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minmass"])
                    self.MPI.mpi_print_zero(" --- Mindens\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Mindens"])
                    self.MPI.mpi_print_zero(" --- Minvol\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvol"])
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Nbins"])
                    self.MPI.mpi_print_zero(" --- Neval\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Neval"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Filaments")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Nbins"])
                    self.MPI.mpi_print_zero(" --- Minvol\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Minvol"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Walls")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Nbins"])
                    self.MPI.mpi_print_zero(" --- Minvol\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Minvol"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Output\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Output"])


    def _bool2yesno(self, _x):
        if _x is True:
            return "Yes"
        else:
            return "No"


    def read_paramfile(self, yaml_fname):
        """Reads parameter file."""
        self.params, self.ERROR = read.read_paramfile(yaml_fname, self.MPI)
        if self.ERROR is False:
            self.read_params(self.params)
        self._break4error()


    def prepare(self):
        """Prepare grid divisions."""
        self.SBX = fiesta.coords.MPI_SortByX(self.MPI)
        self.SBX.settings(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.SBX.limits4grid()
        self.siminfo["x3D"], self.siminfo["y3D"], self.siminfo["z3D"] = shift.cart.mpi_grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)

    # Particle Functions ----------------------------------------------------- #

    def _particle_range(self):
        """Get the particle range for each processor."""
        if self.particles["x"] is None:
            self.MPI.mpi_print(" -> Processor", self.MPI.rank, "particle range: [n/a, n/a, n/a, n/a, n/a, n/a]")
        else:
            lims = (np.min(self.particles["x"]), np.max(self.particles["x"]),
                    np.min(self.particles["y"]), np.max(self.particles["y"]),
                    np.min(self.particles["z"]), np.max(self.particles["z"]))
            self.MPI.mpi_print(" -> Processor", self.MPI.rank, "particle range: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % lims)


    def _get_npart(self):
        """Get the number of particles."""
        self.particles["npart"] = self.MPI.sum(len(self.particles["x"]))
        if self.MPI.rank == 0:
            self.MPI.send(self.particles["npart"], tag=11)
        else:
            self.particles["npart"] = self.MPI.recv(0, tag=11)
        self.MPI.wait()


    def _particle_mass(self):
        """Compute particle mass."""
        Omega_m = self.cosmo["Omega_m"]
        boxsize = self.siminfo["Boxsize"]
        self._get_npart()
        npart = self.particles["npart"]
        self.particles["mass"] = src.density.average_mass_per_cell(Omega_m, boxsize, npart**(1./3.))


    def read_particles(self):
        """Read particles."""
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Particles")
        self.MPI.mpi_print_zero(" =========")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.MPI.mpi_print_zero(" > Reading particles")
        self.time["Particle_Start"] = time.time()
        fnames = self.MPI.split_array(self.particles["Fnames"])
        if len(fnames) != 0:
            for i in range(0, len(fnames)):
                if self.particles["Type"] == "ASCII":
                    _pos = read.read_ascii(fnames[i], self.particles["ASCII_Columns"])
                elif self.particles["Type"] == "NPZ":
                    _pos = read.read_npz(fnames[i], self.particles["NPZ_Keys"])
                elif self.particles["Type"] == "Gadget":
                    _pos = read.read_gadget(fnames[i])
                    _pos *= self.particles["Gadget_Factor"]
                _x, _y, _z = _pos[:,0], _pos[:,1], _pos[:,2]
                if i == 0:
                    self.particles["x"] = _x
                    self.particles["y"] = _y
                    self.particles["z"] = _z
                else:
                    self.particles["x"] = np.concatenate([self.particles["x"], _x])
                    self.particles["y"] = np.concatenate([self.particles["y"], _y])
                    self.particles["z"] = np.concatenate([self.particles["z"], _z])
            data = np.column_stack([self.particles["x"], self.particles["y"], self.particles["z"]])
        else:
            self.particles["x"] = None
            self.particles["y"] = None
            self.particles["z"] = None
            data = None
        self._particle_range()
        self.MPI.wait()

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" > Distributing particles")
        self.SBX.input(data)
        data = self.SBX.distribute()
        self.particles["x"] = data[:,0]
        self.particles["y"] = data[:,1]
        self.particles["z"] = data[:,2]
        self._particle_range()
        self.MPI.wait()

        self._particle_mass()
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" -> NPart     : %i " % self.particles["npart"])
        self.MPI.mpi_print_zero(" -> Mass      : %.4e 10^10 M_solar h^-1" % self.particles["mass"])
        self.MPI.mpi_print_zero(" -> Mean Sep. : %0.4f " % (self.siminfo["Boxsize"]/((self.particles["npart"])**(1./3.))) )

        self.time["Particle_End"] = time.time()


    # Density Functions ------------------------------------------------------ #


    def _save_dens(self):
        """Save density to file."""
        inout.save_dens(self.density["Saveas"], self.MPI.rank, self.siminfo["x3D"],
            self.siminfo["y3D"], self.siminfo["z3D"], self.density["dens"],
            self.siminfo["Ngrid"], self.siminfo["Boxsize"])


    def calculate_density(self):
        """Calculate density."""
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Density")
        self.MPI.mpi_print_zero(" =======")
        self.MPI.mpi_print_zero()
        self.MPI.wait()

        self.MPI.mpi_print_zero(" > Running "+self.density["Type"])
        self.MPI.mpi_print_zero()

        self.time["Density_Start"] = time.time()

        if self.density["Type"] == "NGP" or self.density["Type"] == "CIC" or self.density["Type"] == "TSC":

            dens = fiesta.p2g.mpi_part2grid3D(self.particles["x"], self.particles["y"],
                self.particles["z"], self.particles["mass"]*np.ones(len(self.particles["x"])),
                self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI,
                method=self.density["Type"], periodic=True, origin=0.)

        elif self.density["Type"] == "DTFE":

            dens = fiesta.dtfe.mpi_dtfe4grid3D(self.particles["x"], self.particles["y"], self.particles["z"],
                self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI, self.density["MPI_Split"],
                mass=self.particles["mass"]*np.ones(len(self.particles["x"])), buffer_type=self.density["Buffer_Type"],
                buffer_length=self.density["Buffer_Length"], buffer_val=0., origin=0.,
                subsampling=self.density["Subsampling"], outputgrid=False, verbose=True, verbose_prefix=" -> ")

        self.density["dens"] = dens
        self.MPI.wait()

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" > Saving to "+self.density["Saveas"]+"{0-"+str(self.MPI.size-1)+"}.npz")
        self._save_dens()

        self.time["Density_End"] = time.time()
        self.MPI.wait()


    def _load_dens(self):
        """Load density files."""
        self.ERROR = inout.check_file(self.cosmicweb["Density_Prefix"], 0, self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI)
        self._break4error()
        dens_ind = np.arange(self.cosmicweb["Density_Nfiles"])
        dens_ind = self.MPI.split_array(dens_ind)
        if len(dens_ind) != 0:
            for i in range(0, len(dens_ind)):
                _x3D, _y3D, _z3D, _dens, Ngrid, Boxsize = inout.load_dens(self.cosmicweb["Density_Prefix"], dens_ind[i])
                if i == 0:
                    x3D = _x3D.flatten()
                    y3D = _y3D.flatten()
                    z3D = _z3D.flatten()
                    dens = _dens.flatten()
                else:
                    x3D = np.concatenate([x3D, _x3D.flatten()])
                    y3D = np.concatenate([y3D, _y3D.flatten()])
                    z3D = np.concatenate([z3D, _z3D.flatten()])
                    dens = np.concatenate([dens, _dens.flatten()])
        else:
            x3D, y3D, z3D, dens = None, None, None, None
        self.density["dens"] = self.SBX.distribute_grid3D(x3D, y3D, z3D, dens)


    def _norm_dens(self):
        self.density["dens"] = src.density.mpi_norm_dens(self.density["dens"], self.MPI)


    def _apply_filter(self):
        if self._check_param_key(self.cosmicweb["Filter"], "Type"):

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" # Apply Filter")
            self.MPI.mpi_print_zero(" # ============")
            self.MPI.mpi_print_zero()

            if self.cosmicweb["Filter"]["Type"] == "Tophat":

                self.MPI.mpi_print_zero(" -> Applying Tophat filter")

                self.density["dens"] = src.filters.mpi_tophat3D(self.density["dens"],
                    self.cosmicweb["Filter"]["R"], self.siminfo["Boxsize"],
                    self.siminfo["Ngrid"], self.MPI)

            elif self.cosmicweb["Filter"]["Type"] == "Gaussian":

                self.MPI.mpi_print_zero(" -> Applying Gaussian filter")

                self.density["dens"] = src.filters.mpi_smooth3D(self.density["dens"],
                    self.cosmicweb["Filter"]["R"], self.siminfo["Boxsize"],
                    self.siminfo["Ngrid"], self.MPI)

            elif self.cosmicweb["Filter"]["Type"] == "LogGaussian":

                self.MPI.mpi_print_zero(" -> Applying LogGaussian filter")

                self.density["dens"] = src.filters.mpi_logsmooth3D(self.density["dens"],
                    self.cosmicweb["Filter"]["R"], self.siminfo["Boxsize"],
                    self.siminfo["Ngrid"], self.MPI, setzeroto=None, zero2min=True)

            else:

                self.MPI.mpi_print_zero(" ERROR: Filter %s not supported, must be 'Tophat', 'Gaussian' or 'LogGaussian'." % self.cosmicweb["Filter"]["Type"])
                self.ERROR = True
                self._break4error()


    # Cosmic Web Functions --------------------------------------------------- #

    def _load_sig(self):
        """Load significance files."""
        self.ERROR = inout.check_file(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"],
            0, self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI)
        self._break4error()
        sig_ind = np.arange(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"])
        sig_ind = self.MPI.split_array(sig_ind)
        if len(sig_ind) != 0:
            for i in range(0, len(sig_ind)):
                Ngrid, Boxsize, _x3D, _y3D, _z3D, _Sc, _Sf, _Sw = inout.load_nexus_sig(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"], sig_ind[i])
                if i == 0:
                    x3D = _x3D.flatten()
                    y3D = _y3D.flatten()
                    z3D = _z3D.flatten()
                    Sc = _Sc.flatten()
                    Sf = _Sf.flatten()
                    Sw = _Sw.flatten()
                else:
                    x3D = np.concatenate([x3D, _x3D.flatten()])
                    y3D = np.concatenate([y3D, _y3D.flatten()])
                    z3D = np.concatenate([z3D, _z3D.flatten()])
                    Sc = np.concatenate([Sc, _Sc.flatten()])
                    Sf = np.concatenate([Sf, _Sf.flatten()])
                    Sw = np.concatenate([Sw, _Sw.flatten()])
        else:
            x3D, y3D, z3D, Sc, Sf, Sw = None, None, None, None, None, None
        Sc = self.SBX.distribute_grid3D(x3D, y3D, z3D, Sc)
        Sf = self.SBX.distribute_grid3D(x3D, y3D, z3D, Sf)
        Sw = self.SBX.distribute_grid3D(x3D, y3D, z3D, Sw)
        return Sc, Sf, Sw


    def _average_mass_per_cell(self):
        """Get average mass in each cell."""
        self.siminfo["avgmass"] = src.density.average_mass_per_cell(self.cosmo["Omega_m"],
            self.siminfo["Boxsize"], self.siminfo["Ngrid"])


    def _density2mass(self):
        """Convert density to mass."""
        self.density["mass"] = src.density.mpi_dens2mass(self.density["dens"],
            self.cosmo["Omega_m"], self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)


    def _run_cweb_summary(self):
        """Output cosmic web information."""

        vol_frac = src.cweb.mpi_get_vol_fraction(self.cosmicweb["web_flag"], self.MPI)

        self._density2mass()

        mass_frac = src.cweb.mpi_get_mass_fraction(self.cosmicweb["web_flag"], self.density["mass"], self.MPI)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" # Cosmic Web Summary")
        self.MPI.mpi_print_zero(" # ==================")

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" -> Volume Fraction")

        if self.MPI.rank == 0:

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" --> Clusters  : %0.4f %%" % (vol_frac[3]*1e2))
            self.MPI.mpi_print_zero(" --> Filaments : %0.4f %%" % (vol_frac[2]*1e2))
            self.MPI.mpi_print_zero(" --> Sheets    : %0.4f %%" % (vol_frac[1]*1e2))
            self.MPI.mpi_print_zero(" --> Voids     : %0.4f %%" % (vol_frac[0]*1e2))

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" -> Mass Fraction")

        if self.MPI.rank == 0:

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" --> Clusters  : %0.4f %%" % (mass_frac[3]*1e2))
            self.MPI.mpi_print_zero(" --> Filaments : %0.4f %%" % (mass_frac[2]*1e2))
            self.MPI.mpi_print_zero(" --> Sheets    : %0.4f %%" % (mass_frac[1]*1e2))
            self.MPI.mpi_print_zero(" --> Voids     : %0.4f %%" % (mass_frac[0]*1e2))


    def _save_cweb(self):
        """Save cosmic web environments."""
        if self.cosmicweb["Type"] == "Nexus":
            fname = self.cosmicweb["Nexus"]["Thresholds"]["Output"]+str(self.MPI.rank)+".npz"
            np.savez(fname, web_flag=self.cosmicweb["web_flag"])


    def _run_nexus_signature(self):
        """Compute Nexus signature."""

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ## Calculating NEXUS Signatures")
        self.MPI.mpi_print_zero(" ## ============================")

        Ngrids = [self.siminfo["Ngrid"], self.siminfo["Ngrid"], self.siminfo["Ngrid"]]

        cond = np.where(np.array(self.cosmicweb["Nexus"]["Signature"]["Logsmooth"]) == False)[0]

        if len(cond) > 0:
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" --> Run Multiscale Hessian on density")

            _Sc, _Sf, _Sw = src.nexus.mpi_get_nexus_sig(self.density["dens"],
                self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI,
                logsmooth=False, R0=self.cosmicweb["Nexus"]["Signature"]["R0"],
                Nmax=self.cosmicweb["Nexus"]["Signature"]["Nmax"],
                verbose=True, verbose_prefix=' ---> ')

        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][0] == False:
            Sc = _Sc
        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][1] == False:
            Sf = _Sf
        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][2] == False:
            Sw = _Sw

        cond = np.where(np.array(self.cosmicweb["Nexus"]["Signature"]["Logsmooth"]) == True)[0]

        if len(cond) > 0:
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" --> Run Multiscale Hessian on log10(density)")

            _Sc, _Sf, _Sw = _Sc, _Sf, _Sw = src.nexus.mpi_get_nexus_sig(self.density["dens"],
                self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI,
                logsmooth=True, R0=self.cosmicweb["Nexus"]["Signature"]["R0"],
                Nmax=self.cosmicweb["Nexus"]["Signature"]["Nmax"],
                verbose=True, verbose_prefix=' ---> ')

        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][0] == True:
            Sc = _Sc
        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][1] == True:
            Sf = _Sf
        if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][2] == True:
            Sw = _Sw

        self.cosmicweb["Nexus"]["Signature"]["Sc"] = Sc
        self.cosmicweb["Nexus"]["Signature"]["Sf"] = Sf
        self.cosmicweb["Nexus"]["Signature"]["Sw"] = Sw

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" --> Saving NEXUS signature: "+self.cosmicweb["Nexus"]["Signature"]["Output"]+"{0-%i}.npz"%(self.MPI.size-1))

        prefix = self.cosmicweb["Nexus"]["Signature"]["Output"]

        inout.save_nexus_sig(prefix, self.MPI.rank, self.siminfo["x3D"], self.siminfo["y3D"],
            self.siminfo["z3D"], Sc, Sf, Sw, self.siminfo["Ngrid"], self.siminfo["Boxsize"])


    def _get_nexus_signature(self):
        """Read Nexus signature."""

        if self.cosmicweb["Nexus"]["Signature"]["_Run"] is False:

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" --> Load NEXUS signature: "+self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"]+"{0-%i}.npz"%(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"]-1))

            Sc, Sf, Sw = self._load_sig()

        else:

            Sc = self.cosmicweb["Nexus"]["Signature"]["Sc"]
            Sf = self.cosmicweb["Nexus"]["Signature"]["Sf"]
            Sw = self.cosmicweb["Nexus"]["Signature"]["Sw"]

        return Sc, Sf, Sw


    def _run_nexus_threshold(self):
        """Compute Nexus thresholds."""

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ## Calculating Environment Thresholds")
        self.MPI.mpi_print_zero(" ## ==================================")

        Sc, Sf, Sw = self._get_nexus_signature()

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" --> Convert density to mass")

        self._density2mass()
        mass_total = self.MPI.sum(np.sum(self.density["mass"]))

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ### Computing Cluster Environment")
        self.MPI.mpi_print_zero(" ### =============================")
        self.MPI.mpi_print_zero()

        Sc_lims, Num, Num_dlim, Num_mlim, Num_mlim_dlim = \
            src.nexus.mpi_get_Sc_group_info(Sc, self.density["dens"], self.cosmo["Omega_m"],
                self.siminfo["Boxsize"], self.siminfo["Ngrid"],
                self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvol"],
                self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Mindens"],
                self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minmass"],
                self.MPI, neval=self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Neval"],
                overide_min_sum_M=None, overide_max_sum_M=None,
                verbose=True, prefix=' ---> ')

        Sc_lim = src.nexus.get_clust_threshold(Sc_lims, Num_mlim, Num_mlim_dlim)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ---> Threshold log10(Sc) = %.4f" % np.log10(Sc_lim))

        clust_map = src.nexus.mpi_get_clust_map(Sc, Sc_lim, self.density["dens"],
            self.cosmo["Omega_m"], self.siminfo["Boxsize"], self.siminfo["Ngrid"],
            self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvol"],
            self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Mindens"],
            self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minmass"], self.MPI)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ### Computing Filament Environment")
        self.MPI.mpi_print_zero(" ### ==============================")

        Sf_lim, logSf_lim, dM2 = src.nexus.mpi_get_filam_threshold(Sf, self.density["dens"],
            self.cosmo["Omega_m"], self.siminfo["Boxsize"], self.siminfo["Ngrid"],
            clust_map, self.MPI,  nbins=self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["nbins"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ---> Threshold log10(Sf) = %.4f" % np.log10(Sf_lim))

        filam_map = src.nexus.mpi_get_filam_map(Sf, Sf_lim, self.density["dens"],
            self.siminfo["Boxsize"], self.siminfo["Ngrid"],
            self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Minvol"],
            clust_map, self.MPI)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ### Computing Sheet Environment")
        self.MPI.mpi_print_zero(" ### ===========================")

        Sw_lim, logSw_lim, dM2 = src.nexus.mpi_get_sheet_threshold(Sw, self.density["dens"],
            self.cosmo["Omega_m"], self.siminfo["Boxsize"], self.siminfo["Ngrid"],
            clust_map, filam_map, self.MPI, nbins=self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["nbins"])

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ---> Threshold log10(Sw) = %.4f" % np.log10(Sw_lim))

        sheet_map = src.nexus.mpi_get_sheet_map(Sw, Sw_lim, self.density["dens"],
            self.siminfo["Boxsize"], self.siminfo["Ngrid"],
            self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Minvol"], clust_map,
            filam_map, self.MPI)

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" ### Set Cosmic Web Environment")
        self.MPI.mpi_print_zero(" ### ==========================")

        self.cosmicweb["web_flag"] = src.nexus.get_cweb_map(clust_map, filam_map, sheet_map)

        self.MPI.mpi_print_zero()
        fname = self.cosmicweb["Nexus"]["Thresholds"]["Output"] + "{0-%i}.npz" % (self.MPI.size-1)
        self.MPI.mpi_print_zero(" ---> Saving cosmicweb environments to "+fname)

        self._save_cweb()


    def _run_nexus(self):
        """Compute Nexus."""

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" # Running NEXUS")
        self.MPI.mpi_print_zero(" # =============")

        if self.cosmicweb["Nexus"]["Signature"]["_Run"] is True:
            self._run_nexus_signature()

        if self.cosmicweb["Nexus"]["Thresholds"]["_Run"] is True:
            self._run_nexus_threshold()
            self._run_cweb_summary()


    def calculate_cosmicweb(self):
        """Compute Cosmic Web."""
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" CosmicWeb")
        self.MPI.mpi_print_zero(" =========")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.time["CosmicWeb_Start"] = time.time()

        self.MPI.mpi_print_zero(" > Loading density")
        self._load_dens()
        self._norm_dens()
        self._apply_filter()

        if self.cosmicweb["Type"] == "Nexus":
            self._run_nexus()

        self.time["CosmicWeb_End"] = time.time()


    def _print_time(self, prefix, time):
        """Compute print time.

        Parameters
        ----------
        prefix: str
            Prefix to time ouptut.
        time : float
            Time.
        """
        if time < 1.:
            self.MPI.mpi_print_zero(prefix, "%0.6f seconds" % time)
        elif time < 60:
            self.MPI.mpi_print_zero(prefix, "%0.2f seconds" % time)
        elif time < 60*60:
            time /= 60
            self.MPI.mpi_print_zero(prefix, "%0.2f minutes" % time)
        else:
            time /= 60*60
            self.MPI.mpi_print_zero(prefix, "%0.2f hours" % time)


    def run(self, yaml_fname):
        """Run CaCTus."""
        self.start()

        self.read_paramfile(yaml_fname)

        self.prepare()

        if self.what2run["particles"] is True:
            self.read_particles()

        if self.what2run["density"] is True:

            if self.what2run["particles"] is True:
                self.calculate_density()
            else:
                self.MPI.mpi_print_zero(" ERROR: must input particles to run density")
                self.ERROR = True

            self._break4error()

        if self.what2run["cosmicweb"] is True:
            self.calculate_cosmicweb()

        self.end()


    def end(self):
        """Ends the run."""
        self.MPI.wait()
        self.time["End"] = time.time()
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Running Time")
        self.MPI.mpi_print_zero(" ============")
        self.MPI.mpi_print_zero()
        if self.time["Particle_Start"] is not None:
            self._print_time(" -> Particle\t= ", self.time["Particle_End"] - self.time["Particle_Start"])
        if self.time["Density_Start"] is not None:
            self._print_time(" -> Density\t= ", self.time["Density_End"] - self.time["Density_Start"])
        if self.time["CosmicWeb_Start"] is not None:
            self._print_time(" -> CosmicWeb\t= ", self.time["CosmicWeb_End"] - self.time["CosmicWeb_Start"])
        self._print_time(" -> Total\t= ", self.time["End"] - self.time["Start"])
        self.MPI.mpi_print_zero(cactus_end)
