import os
import time

import numpy as np
import matplotlib.pylab as plt

from scipy.interpolate import interp1d

import fiesta
import magpie
import shift
import cactus

from . import files
from . import read
from . import inout


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
        # Global variables
        self.MPI = MPI
        self.FFT = False
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
            "dens": None
        }
        # CosmicWeb
        self.cosmicweb = {
            "Density_Prefix": None,
            "Density_Nfiles": None,
            "Type": None,
            "Nexus": {
                "Signature": {
                    "_Run": False,
                    "R0": None,
                    "Nmax": None,
                    "Logsmooth": None,
                    "Output": None
                },
                "Thresholds": {
                    "_Run": False,
                    "SigFile": {
                        "Prefix": None,
                        "Nfiles": None
                    },
                    "Clusters": {
                        "Minvirdens": None,
                        "nbins": 100,
                        "Eval": None,
                    },
                    "Filaments": {
                        "nbins": 100,
                    },
                    "Walls": {
                        "nbins": 100,
                    },
                    "Output": None
                }
            },
            "cweb": None,
        }


    def start(self):
        self.time["Start"] = time.time()
        self.MPI.mpi_print_zero(cactus_beg)


    def _break4error(self):
        if self.ERROR is True:
            exit()


    def _check_particle_files(self):
        for fname in self.particles["Fnames"]:
            if files.check_exist(fname) is False:
                self.MPI.mpi_print_zero(" ERROR: File '"+fname+"' does not exist, aborting.")
                self.ERROR = True
            else:
                self.ERROR = False


    def _construct_particle_files(self):
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


    def read_params(self, params):

        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Parameters")
        self.MPI.mpi_print_zero(" ----------")

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
        if params["Particles"] is not None:
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

            self._construct_particle_files()
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
        if params["Density"] is not None:
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
        if params["CosmicWeb"] is not None:
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" CosmicWeb:")

            self.cosmicweb["Density_Prefix"] = params["CosmicWeb"]["Density_Prefix"]
            self.cosmicweb["Density_Nfiles"] = params["CosmicWeb"]["Density_Nfiles"]

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Density_Prefix\t=", self.cosmicweb["Density_Prefix"])
            self.MPI.mpi_print_zero(" - Density_Nfiles\t=", self.cosmicweb["Density_Nfiles"])

            self.what2run["cosmicweb"] = True

            self.cosmicweb["Type"] = params["CosmicWeb"]["Type"]
            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Type\t\t\t=", self.cosmicweb["Type"])

            if self.cosmicweb["Type"] == "Nexus":

                if params["CosmicWeb"]["Nexus"]["Signature"] is not None:

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

                if params["CosmicWeb"]["Nexus"]["Thresholds"] is not None:

                    self.cosmicweb["Nexus"]["Thresholds"]["_Run"] = True

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" - Thresholds")

                    self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["SigFile"]["Prefix"]
                    self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["SigFile"]["Nfiles"]

                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvirdens"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Minvirdens"]
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Nbins"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Nbins"]
                    self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Evaluate"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Clusters"]["Evaluate"]

                    self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Nbins"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Filaments"]["Nbins"]

                    self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Nbins"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Walls"]["Nbins"]

                    self.cosmicweb["Nexus"]["Thresholds"]["Output"] = params["CosmicWeb"]["Nexus"]["Thresholds"]["Output"]

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- SigFile")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Prefix\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"])
                    self.MPI.mpi_print_zero(" --- Nfiles\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Clusters")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Minvirdens\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvirdens"])
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Nbins"])
                    self.MPI.mpi_print_zero(" --- Evaluate\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Evaluate"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Filaments")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Nbins"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Walls")

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" --- Nbins\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Nbins"])

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Output\t\t=", self.cosmicweb["Nexus"]["Thresholds"]["Output"])


    def read_paramfile(self, yaml_fname):
        self.params, self.ERROR = read.read_paramfile(yaml_fname, self.MPI)
        if self.ERROR is False:
            self.read_params(self.params)
        self._break4error()


    def prepare(self):
        self.SBX = fiesta.coords.MPI_SortByX(self.MPI)
        self.SBX.settings(self.siminfo["Boxsize"], self.siminfo["Ngrid"])
        self.SBX.limits4grid()
        self.siminfo["x3D"], self.siminfo["y3D"], self.siminfo["z3D"] = shift.cart.mpi_grid3D(self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI)


    def _particle_range(self):
        if self.particles["x"] is None:
            self.MPI.mpi_print(" - Processor", self.MPI.rank, "particle range: [n/a, n/a, n/a, n/a, n/a, n/a]")
        else:
            lims = (np.min(self.particles["x"]), np.max(self.particles["x"]),
                    np.min(self.particles["y"]), np.max(self.particles["y"]),
                    np.min(self.particles["z"]), np.max(self.particles["z"]))
            self.MPI.mpi_print(" - Processor", self.MPI.rank, "particle range: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]" % lims)


    def _get_npart(self):
        self.particles["npart"] = self.MPI.sum(len(self.particles["x"]))
        if self.MPI.rank == 0:
            self.MPI.send(self.particles["npart"], tag=11)
        else:
            self.particles["npart"] = self.MPI.recv(0, tag=11)
        self.MPI.wait()


    def _particle_mass(self):
        Omega_m = self.cosmo["Omega_m"]
        boxsize = self.siminfo["Boxsize"]
        self._get_npart()
        npart = self.particles["npart"]
        G_const = 6.6743e-11
        self.particles["mass"] = 3.*Omega_m*(boxsize**3.)/(8.*np.pi*G_const*npart)
        self.particles["mass"] *= 3.0857e2/1.9891
        self.particles["mass"] /= 1e10


    def read_particles(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Particles")
        self.MPI.mpi_print_zero(" ---------")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.MPI.mpi_print_zero(" - Reading particles")
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
        self.MPI.mpi_print_zero(" - Distributing particles")
        self.SBX.input(data)
        data = self.SBX.distribute()
        self.particles["x"] = data[:,0]
        self.particles["y"] = data[:,1]
        self.particles["z"] = data[:,2]
        self._particle_range()
        self.MPI.wait()

        self._particle_mass()
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" - NPart:", self.particles["npart"])
        self.MPI.mpi_print_zero(" - Mass: %.4e" % (self.particles["mass"]*1e10), "M_solar h^-1")
        self.MPI.mpi_print_zero(" - Mean Sep:", self.siminfo["Boxsize"]/((self.particles["npart"])**(1./3.)))

        self.time["Particle_End"] = time.time()


    def _save_dens(self):
        inout.save_dens(self.density["Saveas"], self.MPI.rank, self.siminfo["x3D"], self.siminfo["y3D"], self.siminfo["z3D"],
                        self.density["dens"], self.siminfo["Ngrid"], self.siminfo["Boxsize"])


    def calculate_density(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Density")
        self.MPI.mpi_print_zero(" -------")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.MPI.mpi_print_zero(" - Running "+self.density["Type"])
        self.time["Density_Start"] = time.time()
        if self.density["Type"] == "NGP" or self.density["Type"] == "CIC" or self.density["Type"] == "TSC":
            dens = fiesta.p2g.mpi_part2grid3D(self.particles["x"], self.particles["y"],
                self.particles["z"], self.particles["mass"]*np.ones(len(self.particles["x"])),
                self.siminfo["Boxsize"], self.siminfo["Ngrid"], self.MPI,
                method=self.density["Type"], periodic=True, origin=0.)
        elif self.density["Type"] == "DTFE":
            dens = fiesta.dtfe.mpi_dtfe4grid3D(self.particles["x"], self.particles["y"], self.particles["z"],
                self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI, self.density["MPI_Split"],
                mass=self.particles["mass"]*np.ones(len(self.particles["x"])),
                buffer_type=self.density["Buffer_Type"],
                buffer_length=self.density["Buffer_Length"], buffer_val=0., origin=0.,
                subsampling=self.density["Subsampling"], outputgrid=False, verbose=True, verbose_prefix=" - ")
        self.density["dens"] = dens
        self.MPI.wait()
        self.MPI.mpi_print_zero(" - Saving to "+self.density["Saveas"]+"{0-"+str(self.MPI.size-1)+"}.npz")
        self._save_dens()
        self.time["Density_End"] = time.time()
        self.MPI.wait()


    def _load_dens(self):
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
        dens = self.SBX.distribute_grid3D(x3D, y3D, z3D, dens)
        self.density["dens"] = dens


    def _load_sig(self):
        self.ERROR = inout.check_file(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"], 0, self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI)
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
        Omega_m = self.cosmo["Omega_m"]
        boxsize = self.siminfo["Boxsize"]
        ncells = self.siminfo["Ngrid"]**3
        G_const = 6.6743e-11
        self.siminfo["avgmass"] = 3.*Omega_m*(boxsize**3.)/(8.*np.pi*G_const*ncells)
        self.siminfo["avgmass"] *= 3.0857e2/1.9891
        self.siminfo["avgmass"] /= 1e10


    def _density2mass(self):
        dv = (self.siminfo["Boxsize"]/self.siminfo["Ngrid"])**3
        mass = self.density["dens"]*dv
        return mass


    def _get_histogram(self, x, minval, maxval, ngrid):
        """Returns binned (histograms) from an input data set.

        Parameters
        ----------
        x : array
            X-values to be binned.
        minval, maxval : float
            Minimum and maximum values.
        ngrid : int
            Grid dimensions.

        Returns
        -------
        hx, hy : array
            Histogram grid (hx) and binned values (hy).
        """
        pixID = magpie.pixels.pos2pix_cart1d(x, maxval-minval, ngrid, origin=minval)
        hy = magpie.pixels.bin_pix(pixID, ngrid)
        hy = self.MPI.sum(hy)
        hxedges, hx = shift.cart.grid1D(maxval-minval, ngrid, origin=minval)
        if self.MPI.rank == 0:
            return hx, hy
        else:
            return None, None


    def _get_cdf(self, x, minval, maxval, ngrid):
        hx, hy = self._get_histogram(x, minval, maxval, ngrid)
        if self.MPI.rank == 0:
            dx = hx[1]-hx[0]
            cdf_x = np.zeros(len(hx)+1)
            cdf_x[1:] = hx + 0.5*dx
            cdf_x[0] = hx[0] - 0.5*dx
            cdf_y = np.zeros(len(cdf_x))
            cdf_y[1:] = np.cumsum(hy)
            cdf_y /= cdf_y[-1]
            return cdf_x, cdf_y
        else:
            return None, None


    def _bin_mass_by_signature(self, mass, logS, minlogS, maxlogS, ngrid):
        """Returns binned (histograms) from an input data set.

        Parameters
        ----------
        mass : array
            Mass of pixel.
        logS : array
            Signature value.
        minlogS, maxlogS : float
            Minimum and maximum logS values.
        ngrid : int
            Grid dimensions.

        Returns
        -------
        logSx, M : array
            Histogram grid (logSc) and binned values (M).
        """
        pixID = magpie.pixels.pos2pix_cart1d(logS, maxlogS-minlogS, ngrid, origin=minlogS)
        M = magpie.pixels.bin_pix(pixID, ngrid, weights=mass)
        M = self.MPI.sum(M)
        if self.MPI.rank == 0:
            M = np.cumsum(M[::-1])[::-1]
        logSx_edges, logSx = shift.cart.grid1D(maxlogS-minlogS, ngrid, origin=minlogS)
        if self.MPI.rank == 0:
            return logSx_edges[:-1], M
        else:
            return None, None


    def _get_dM_dlogS(self, mass, logS, minlogS, maxlogS, ngrid):
        """Returns binned (histograms) from an input data set.

        Parameters
        ----------
        mass : array
            Mass of pixel.
        logS : array
            Signature value.
        minlogS, maxlogS : float
            Minimum and maximum logS values.
        ngrid : int
            Grid dimensions.

        Returns
        -------
        logSx, dM : array
            Differential of Mass vs logS.
        """
        logSx, M = self._bin_mass_by_signature(mass, logS, minlogS, maxlogS, ngrid)
        if self.MPI.rank == 0:
            dM = fiesta.maths.dfdx(logSx, M)**2.
            return logSx, dM
        else:
            return None, None


    def _save_cweb(self):
        if self.cosmicweb["Type"] == "Nexus":
            fname = self.cosmicweb["Nexus"]["Thresholds"]["Output"]+str(self.MPI.rank)+".npz"
            np.savez(fname, cweb=self.cosmicweb["cweb"])


    def calculate_cosmicweb(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" CosmicWeb")
        self.MPI.mpi_print_zero(" ---------")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.time["CosmicWeb_Start"] = time.time()

        self.MPI.mpi_print_zero(" - Loading density")
        self._load_dens()

        if self.cosmicweb["Type"] == "Nexus":

            self.MPI.mpi_print_zero()
            self.MPI.mpi_print_zero(" - Running NEXUS")

            if self.cosmicweb["Nexus"]["Signature"]["_Run"] is True:

                self.MPI.mpi_print_zero(" - Initialising FFT object")
                Ngrids = [self.siminfo["Ngrid"], self.siminfo["Ngrid"], self.siminfo["Ngrid"]]
                self.FFT = self.MPI.mpi_fft_start(Ngrids)

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" - Calculating NEXUS Signatures")

                cond = np.where(np.array(self.cosmicweb["Nexus"]["Signature"]["Logsmooth"]) == False)[0]
                if len(cond) > 0:
                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Run Multiscale Hessian on density")
                    self.MPI.mpi_print_zero()
                    _Sc, _Sf, _Sw = cactus.nexus.mpi_get_nexus_sig(self.density["dens"],
                        self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI, self.FFT,
                        logsmooth=False, R0=self.cosmicweb["Nexus"]["Signature"]["R0"],
                        Nmax=self.cosmicweb["Nexus"]["Signature"]["Nmax"],
                        verbose=True, verbose_prefix=' --- ')
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][0] == False:
                    Sc = _Sc
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][1] == False:
                    Sf = _Sf
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][2] == False:
                    Sw = _Sw
                cond = np.where(np.array(self.cosmicweb["Nexus"]["Signature"]["Logsmooth"]) == True)[0]
                if len(cond) > 0:
                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Run Multiscale Hessian on log10(density)")
                    self.MPI.mpi_print_zero()
                    _Sc, _Sf, _Sw = cactus.nexus.mpi_get_nexus_sig(self.density["dens"],
                        self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI, self.FFT,
                        logsmooth=True, R0=self.cosmicweb["Nexus"]["Signature"]["R0"],
                        Nmax=self.cosmicweb["Nexus"]["Signature"]["Nmax"],
                        verbose=True, verbose_prefix=' --- ')
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][0] == True:
                    Sc = _Sc
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][1] == True:
                    Sf = _Sf
                if self.cosmicweb["Nexus"]["Signature"]["Logsmooth"][2] == True:
                    Sw = _Sw
                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Saving NEXUS signature: "+self.cosmicweb["Nexus"]["Signature"]["Output"]+"{0-%i}.npz"%(self.MPI.size-1))
                prefix = self.cosmicweb["Nexus"]["Signature"]["Output"]
                inout.save_nexus_sig(prefix, self.MPI.rank, self.siminfo["x3D"], self.siminfo["y3D"],
                    self.siminfo["z3D"], Sc, Sf, Sw, self.siminfo["Ngrid"], self.siminfo["Boxsize"])

            if self.cosmicweb["Nexus"]["Thresholds"]["_Run"] is True:

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" - Calculating Environment Thresholds")

                if self.cosmicweb["Nexus"]["Signature"]["_Run"] is False:

                    self.MPI.mpi_print_zero()
                    self.MPI.mpi_print_zero(" -- Load NEXUS signature: "+self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Prefix"]+"{0-%i}.npz"%(self.cosmicweb["Nexus"]["Thresholds"]["SigFile"]["Nfiles"]-1))

                    Sc, Sf, Sw = self._load_sig()

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Convert density to mass")

                mass = self._density2mass()
                mass_total = self.MPI.sum(np.sum(mass))

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Computing Cluster Thresholds")

                cond = np.where(Sc.flatten() != 0.)[0]

                NumSc = self.MPI.sum(len(cond))

                if self.MPI.rank != 0:
                    NumSc = 0.

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Pixels with Sc != 0: %.2f %%" % (100.*NumSc/(self.siminfo["Ngrid"]**3.)))

                minSc = self.MPI.min(Sc.flatten()[cond])
                maxSc = self.MPI.max(Sc.flatten()[cond])

                minlogSc, maxlogSc = np.log10(minSc), np.log10(maxSc)
                logSc = np.log10(Sc.flatten()[cond])

                self.MPI.mpi_print_zero(" --- log10(Sc) range: [%.2f, %.2f]" % (minlogSc, maxlogSc))

                binlen = self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Nbins"]
                cdf_x, cdf_y = self._get_cdf(logSc, minlogSc, maxlogSc, binlen)

                percent = np.linspace(0., 1., self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Evaluate"]+2)[1:-1]

                if self.MPI.rank == 0:
                    f = interp1d(cdf_y, cdf_x)
                    logSc_percent = f(percent)
                    self.MPI.send(logSc_percent, tag=11)
                else:
                    logSc_percent = self.MPI.recv(0, tag=11)

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Running GroupFinder at %i scales" % self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Evaluate"])

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- #Num\t| min[log10(Sc)]\t| Pixels Frac.\t| Ngroups \t| Cluster Frac.\t|")

                f_good_clusters = []

                for i in range(0, len(logSc_percent)):

                    logSc_val = logSc_percent[i]

                    binmap = np.zeros(np.shape(Sc))
                    cond = np.where(Sc > 10.**logSc_val)
                    binmap[cond] = 1.

                    N_pix_nonzero = self.MPI.sum(np.sum(binmap))

                    if self.MPI.rank == 0:
                        f_pix_nonzero = N_pix_nonzero/(self.siminfo["Ngrid"]**3.)
                        str_val = " --- %i/%i\t| %.4f \t\t| %.4f %%\t|" % (i+1, len(logSc_percent), logSc_val, 100.*f_pix_nonzero)

                    groupID = cactus.groups.mpi_groupfinder(binmap, self.MPI)
                    maxID = self.MPI.max(groupID.flatten())

                    if self.MPI.rank == 0:
                        str_val += " %i \t\t|" % maxID

                    group_N = cactus.groups.mpi_get_ngroup(groupID, self.MPI)
                    group_mass = cactus.groups.mpi_sum4group(groupID, mass, self.MPI)

                    if self.MPI.rank == 0:
                        dV = (self.siminfo["Boxsize"]/self.siminfo["Ngrid"])**3.
                        group_vol = group_N*dV
                        group_dens = group_mass/group_vol
                        cond = np.where(group_dens > self.cosmicweb["Nexus"]["Thresholds"]["Clusters"]["Minvirdens"])[0]
                        fval_good_clusters = len(cond)/len(group_mass)
                        str_val += " %.4f %%\t|" % (100.*fval_good_clusters)
                        f_good_clusters.append(fval_good_clusters)
                    else:
                        str_val = ""

                    self.MPI.mpi_print_zero(str_val)

                if self.MPI.rank == 0:
                    f_good_clusters = np.array(f_good_clusters)
                    f = interp1d(f_good_clusters, logSc_percent)
                    logSc_threshold = f(0.5)
                    self.MPI.send(logSc_threshold, tag=11)
                else:
                    logSc_threshold = self.MPI.recv(0, tag=11)

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Threshold log10(Sc) = %.4f" % logSc_threshold)

                self.cosmicweb["cweb"] = np.zeros(np.shape(self.density["dens"]), dtype="int")
                cond = np.where(Sc >= 10.**logSc_threshold)
                self.cosmicweb["cweb"][cond] = 1

                ## might need to edit this.
                Sf[cond] = 0.
                Sw[cond] = 0.

                Nc = self.MPI.sum(len(cond[0]))
                Mc = self.MPI.sum(np.sum(mass[cond]))

                self.MPI.mpi_print_zero()
                if self.MPI.rank == 0:
                    Cluster_Vol_frac = Nc/(self.siminfo["Ngrid"]**3)
                    Cluster_Mass_frac = Mc/mass_total
                    self.MPI.mpi_print_zero(" --- Clusters Volume Fraction = %.4f %%" % (100.*Cluster_Vol_frac))
                    self.MPI.mpi_print_zero(" --- Clusters Mass Fraction   = %.4f %%" % (100.*Cluster_Mass_frac))

                # Remove groups that are too small, a few points.

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Computing Filament Thresholds")

                cond = np.where(Sf.flatten() != 0.)[0]

                NumSf = self.MPI.sum(len(cond))

                if self.MPI.rank != 0:
                    NumSf = 0.

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Pixels with Sf != 0: %.2f %%" % (100.*NumSf/(self.siminfo["Ngrid"]**3.)))

                minSf = self.MPI.min(Sf.flatten()[cond])
                maxSf = self.MPI.max(Sf.flatten()[cond])

                minlogSf, maxlogSf = np.log10(minSf), np.log10(maxSf)
                logSf = np.log10(Sf.flatten()[cond])

                logSf, dM2 = self._get_dM_dlogS(mass.flatten()[cond], logSf, minlogSf, maxlogSf, self.cosmicweb["Nexus"]["Thresholds"]["Filaments"]["Nbins"])

                if self.MPI.rank == 0:
                    ind = np.argmax(dM2)
                    logSf_threshold = logSf[ind]
                    self.MPI.send(logSf_threshold, tag=11)
                else:
                    logSf_threshold = self.MPI.recv(0, tag=11)

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Threshold log10(Sf) = %.4f" % logSf_threshold)

                cond = np.where((Sf >= 10.**logSf_threshold) & (self.cosmicweb["cweb"] == 0))
                self.cosmicweb["cweb"][cond] = 2
                ## might need to edit this.
                Sw[cond] = 0.

                Nf = self.MPI.sum(len(cond[0]))
                Mf = self.MPI.sum(np.sum(mass[cond]))

                self.MPI.mpi_print_zero()
                if self.MPI.rank == 0:
                    Filament_Vol_frac = Nf/(self.siminfo["Ngrid"]**3)
                    Filament_Mass_frac = Mf/mass_total
                    self.MPI.mpi_print_zero(" --- Filament Volume Fraction = %.4f %%" % (100.*Filament_Vol_frac))
                    self.MPI.mpi_print_zero(" --- Filament Mass Fraction   = %.4f %%" % (100.*Filament_Mass_frac))

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" -- Computing Wall Thresholds")

                cond = np.where(Sw.flatten() != 0.)[0]

                NumSw = self.MPI.sum(len(cond))

                if self.MPI.rank != 0:
                    NumSw = 0.

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Pixels with Sw != 0: %.2f %%" % (100.*NumSw/(self.siminfo["Ngrid"]**3.)))

                minSw = self.MPI.min(Sw.flatten()[cond])
                maxSw = self.MPI.max(Sw.flatten()[cond])

                minlogSw, maxlogSw = np.log10(minSw), np.log10(maxSw)
                logSw = np.log10(Sw.flatten()[cond])

                logSw, dM2 = self._get_dM_dlogS(mass.flatten()[cond], logSw, minlogSw, maxlogSw, self.cosmicweb["Nexus"]["Thresholds"]["Walls"]["Nbins"])

                if self.MPI.rank == 0:
                    ind = np.argmax(dM2)
                    logSw_threshold = logSw[ind]
                    self.MPI.send(logSw_threshold, tag=11)
                else:
                    logSw_threshold = self.MPI.recv(0, tag=11)

                self.MPI.mpi_print_zero()
                self.MPI.mpi_print_zero(" --- Threshold log10(Sw) = %.4f" % logSw_threshold)

                cond = np.where((Sw >= 10.**logSw_threshold) & (self.cosmicweb["cweb"] == 0))
                self.cosmicweb["cweb"][cond] = 3

                Nw = self.MPI.sum(len(cond[0]))
                Mw = self.MPI.sum(np.sum(mass[cond]))

                self.MPI.mpi_print_zero()
                if self.MPI.rank == 0:
                    Wall_Vol_frac = Nw/(self.siminfo["Ngrid"]**3)
                    Wall_Mass_frac = Mw/mass_total
                    self.MPI.mpi_print_zero(" --- Wall Volume Fraction     = %.4f %%" % (100.*Wall_Vol_frac))
                    self.MPI.mpi_print_zero(" --- Wall Mass Fraction       = %.4f %%" % (100.*Wall_Mass_frac))

                self.MPI.mpi_print_zero()
                fname = self.cosmicweb["Nexus"]["Thresholds"]["Output"] + "{0-%i}.npz" % (self.MPI.size-1)
                self.MPI.mpi_print_zero(" -- Saving cosmicweb environments to "+fname)

                self._save_cweb()

        self.time["CosmicWeb_End"] = time.time()


    def _print_time(self, prefix, time):
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
        self.MPI.wait()
        self.time["End"] = time.time()
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" Running Time")
        self.MPI.mpi_print_zero(" ------------")
        self.MPI.mpi_print_zero()
        if self.time["Particle_Start"] is not None:
            self._print_time(" - Particle\t= ", self.time["Particle_End"] - self.time["Particle_Start"])
        if self.time["Density_Start"] is not None:
            self._print_time(" - Density\t= ", self.time["Density_End"] - self.time["Density_Start"])
        if self.time["CosmicWeb_Start"] is not None:
            self._print_time(" - CosmicWeb\t= ", self.time["CosmicWeb_End"] - self.time["CosmicWeb_Start"])
        self._print_time(" - Total\t= ", self.time["End"] - self.time["Start"])
        self.MPI.mpi_print_zero(cactus_end)
