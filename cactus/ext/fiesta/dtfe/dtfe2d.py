import numpy as np
from scipy.spatial import Delaunay as scDelaunay

from .. import boundary
from .. import coords
from .. import src


class Delaunay2D:


    def __init__(self):
        """Initialises Delaunay2D class"""
        self.points = None
        self.points_dens = None
        self.npart = None
        self.delaunay = None
        self.delaunay_simplices = None
        self.delaunay_area = None
        self.x0 = None
        self.y0 = None
        self.f0 = None
        self.delf0 = None
        self.extent = None
        self.boxsize = None
        self.buffer_length = None
        self.usebuffer = False
        self.nbuffer = None
        self.ispart = None
        self.useperiodic = False
        self.nperiodic = None


    def _extent(self):
        """"Calculates and stores minimum and maximum of
        input points in the variable extent."""
        if self.extent is None:
            xmin = np.min(self.points[:self.npart, 0])
            xmax = np.max(self.points[:self.npart, 0])
            ymin = np.min(self.points[:self.npart, 1])
            ymax = np.max(self.points[:self.npart, 1])
            self.extent = [xmin, xmax, ymin, ymax]


    def set_points(self, x, y, f, mass=None):
        """Sets the points for voronoi cells.

        Parameters
        ----------
        x, y : array
            X and Y-coordinates.
        f : array
            Field values
        mass : array, optional
            For calculating density fields.
        """
        if mass is None:
            self.points = coords.coord2points([x, y, f])
        else:
            self.points = coords.coord2points([x, y, f, mass])
        self.npart = len(self.points)
        self.ntotal = self.npart


    def set_buffer(self, boxsize, buffer_length, buffer_val=0., buffer_mass=None):
        """Defines buffer particles to be placed around a given box.

        Parameters
        ----------
        boxsize : float
            Size of the box, assumed particles lie in the range [0., boxsize]
        buffer_length : float
            Length of the buffer region.
        buffer_val : float, optional
            Buffer value.
        buffer_mass : float, optional
            Must be provided if mass is provided.
        """
        # check boxsize is consistent with particles.
        self._extent()
        assert self.extent[0] >= 0. and self.extent[1] <= boxsize, "X coordinates exceed the range of the box, check or redefine boxsize."
        assert self.extent[2] >= 0. and self.extent[3] <= boxsize, "Y coordinates exceed the range of the box, check or redefine boxsize."
        self.boxsize = boxsize
        self.buffer_length = buffer_length
        self.usebuffer = True
        self.useperiodic = False
        x_buffer, y_buffer = boundary.buffer_random_2D(self.npart, self.boxsize, self.buffer_length)
        if len(self.points[0]) == 3:
            points_buffer = coords.coord2points([x_buffer, y_buffer, buffer_val*np.ones(len(x_buffer))])
        else:
            points_buffer = coords.coord2points([x_buffer, y_buffer, buffer_val*np.ones(len(x_buffer)), buffer_mass*np.ones(len(x_buffer))])
        self.nbuffer = len(x_buffer)
        self.ntotal += self.nbuffer
        # redefine points to include buffer points and also define mask
        self.ispart = np.ones(self.npart + self.nbuffer)
        self.ispart[self.npart:] = 0.
        # concatenate points and buffer
        self.points = np.vstack([self.points, points_buffer])


    def set_periodic(self, boxsize, buffer_length):
        """Defines periodic particles to be placed around a given box.

        Parameters
        ----------
        boxsize : float
            Size of the box, assumed particles lie in the range [0., boxsize]
        buffer_length : float
            Length of the buffer region.
        """
        # check boxsize is consistent with particles.
        self._extent()
        assert self.extent[0] >= 0. and self.extent[1] <= boxsize, "X coordinates exceed the range of the box, check or redefine boxsize."
        assert self.extent[2] >= 0. and self.extent[3] <= boxsize, "Y coordinates exceed the range of the box, check or redefine boxsize."
        self.boxsize = boxsize
        self.buffer_length = buffer_length
        points_periodic = boundary.buffer_periodic_2D(self.points, self.boxsize, self.buffer_length)
        self.nperiodic = len(points_periodic)
        self.ntotal += self.nperiodic
        self.usebuffer = False
        self.useperiodic = True
        # redefine points to include periodic points and also define mask
        self.ispart = np.ones(self.npart + self.nperiodic)
        self.ispart[self.npart:] = 0.
        # concatenate points and buffer
        self.points = np.vstack([self.points, points_periodic])


    def construct(self):
        """Constructs Delaunay tesselation"""
        self.delaunay = scDelaunay(self.points[:,:2])
        self.delaunay_simplices = self.delaunay.simplices
        self.nvert = len(self.delaunay_simplices[:, 0])


    def get_area(self):
        """Calculates the area of the delaunay simplex."""
        x, y = self.points[:, 0], self.points[:, 1]
        del_vert0 = self.delaunay_simplices[:, 0]
        del_vert1 = self.delaunay_simplices[:, 1]
        del_vert2 = self.delaunay_simplices[:, 2]
        self.delaunay_area = src.delaunay_area_2d(x=x, y=y, del_vert0=del_vert0,
            del_vert1=del_vert1, del_vert2=del_vert2, npart=self.ntotal,
            nvert=self.nvert)


    def get_dens(self):
        """Calculates the density of each point in the delaunay tessellation."""
        if self.delaunay_area is None:
            self.get_area()
        del_vert0 = self.delaunay_simplices[:, 0]
        del_vert1 = self.delaunay_simplices[:, 1]
        del_vert2 = self.delaunay_simplices[:, 2]
        point_area = src.sum_delaunay4points_2d(delaunay_value=self.delaunay_area,
            del_vert0=del_vert0, del_vert1=del_vert1, del_vert2=del_vert2,
            npart=self.ntotal, nvert=self.nvert)
        if len(self.points) == 3:
            self.points_dens = 1./point_area
        else:
            self.points_dens = self.points[:,3]/point_area


    def set_field(self, f=None, bufferval=0.):
        """Sets the field values of the input points.

        Parameters
        ----------
        f : array
            Field values.
        bufferval : float, optional
            Field values to assign boundary particles.
        """
        if f is not None:
            lenf = len(f)
            assert lenf == len(self.points), "f must be equal to input points."
            self.points[:, 2] = f
        x, y, f = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        del_vert0 = self.delaunay_simplices[:, 0]
        del_vert1 = self.delaunay_simplices[:, 1]
        del_vert2 = self.delaunay_simplices[:, 2]
        self.x0 = x[del_vert0]
        self.y0 = y[del_vert0]
        self.f0 = f[del_vert0]
        self.delf0 = src.get_delf0_2d(x=x, y=y, f=f, del_vert0=del_vert0,
            del_vert1=del_vert1, del_vert2=del_vert2, npart=self.ntotal,
            nvert=self.nvert)


    def find_simplex(self, x, y):
        """Find the simplex the coordinates lie within."""
        points = coords.xy2points(x, y)
        simplices = self.delaunay.find_simplex(points)
        return simplices


    def estimate(self, x, y):
        """Estimates a field from the Delaunay tesselation.

        Parameters
        ----------
        x : array
            X-coordinate for the field estimation.
        y : array
            Y-coordinate for the field estimation.

        Returns
        -------
        f_est : array
            Estimates of the field
        """
        simplices = self.find_simplex(x, y)
        f_est = src.delaunay_estimate_2d(simplices=simplices, x=x, y=y, x0=self.x0,
            y0=self.y0, f0=self.f0, delf0=self.delf0, npart=len(x), nsimp0=len(self.x0))
        return f_est


    def clean(self):
        self.__init__()
