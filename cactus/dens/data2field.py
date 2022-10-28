import numpy as np


class Data2Field:


    def __init__(self):
        self.usempi = False


    def setup(self, files, readfunc, boxsize, MPI=None):
        self.MPI = MPI
        self.files = files
        self.readfunc = readfunc
        self.boxsize = boxsize


    def find_ranges(self, files, readfunc):
        if self.MPI is not None:
            files_array = self.MPI.split_array(files)
        else:
            files_array = files
        xmin, xmax, ymin, ymax, zmin, zmax = [], [], [], [], [], []
        for _file in files_array:
            x, y, z = readfunc(_file, pos=True, vel=False)
            xmin.append(np.min(x))
            ymin.append(np.min(y))
            zmin.append(np.min(z))
            xmax.append(np.max(x))
            ymax.append(np.max(y))
            zmax.append(np.max(z))
        xmin = np.array(xmin)
        ymin = np.array(ymin)
        zmin = np.array(zmin)
        xmax = np.array(xmax)
        ymax = np.array(ymax)
        zmax = np.array(zmax)
        if self.MPI is not None:
            xmin = self.MPI.collect(xmin)
            xmax = self.MPI.collect(xmax)
            ymin = self.MPI.collect(ymin)
            ymax = self.MPI.collect(ymax)
            zmin = self.MPI.collect(zmin)
            zmax = self.MPI.collect(zmax)
        self.file_ranges = [xmin, xmax, ymin, ymax, zmin, zmax]
