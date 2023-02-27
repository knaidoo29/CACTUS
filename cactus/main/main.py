import os
import time

import numpy as np

import fiesta
import shift

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
            "Density_Nfiles": None
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
            self.MPI.mpi_print_zero(" - Type\t\t\t=", self.cosmicweb["Type"])

            if self.cosmicweb["Type"] == "NEXUS":
                self.cosmicweb["R0"] = params["CosmicWeb"]["R0"]
                self.cosmicweb["Nmax"] = params["CosmicWeb"]["Nmax"]
                self.cosmicweb["logsmooth"] = params["CosmicWeb"]["logsmooth"]
                self.cosmicweb["virdens"] = params["CosmicWeb"]["virdens"]
                self.cosmicweb["Output"] = params["CosmicWeb"]["Output"]

                self.MPI.mpi_print_zero(" - R0\t\t\t=", self.cosmicweb["R0"])
                self.MPI.mpi_print_zero(" - Nmax\t\t\t=", self.cosmicweb["Nmax"])
                self.MPI.mpi_print_zero(" - logsmooth\t\t=", self.cosmicweb["logsmooth"])
                self.MPI.mpi_print_zero(" - virdens\t\t=", self.cosmicweb["virdens"])
                self.MPI.mpi_print_zero(" - Output\t\t=", self.cosmicweb["Output"])

        #
        # if params["Image"] is not None:
        #     self.MPI.mpi_print_zero()
        #     self.MPI.mpi_print_zero(" Image:")
        #
        #     self.MPI.mpi_print_zero()
        #     self.MPI.mpi_print_zero(" - Density_Prefix\t=", self.cosmicweb["Density_Prefix"])
        #


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
        self.ERROR = inout.check_dens(self.cosmicweb["Density_Prefix"], 0, self.siminfo["Ngrid"], self.siminfo["Boxsize"], self.MPI)
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



    def calculate_cosmicweb(self):
        self.MPI.mpi_print_zero()
        self.MPI.mpi_print_zero(" CosmicWeb")
        self.MPI.mpi_print_zero(" ---------")
        self.MPI.mpi_print_zero()
        self.MPI.wait()
        self.time["CosmicWeb_Start"] = time.time()
        self.MPI.mpi_print_zero(" - Loading density")
        self._load_dens()

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
