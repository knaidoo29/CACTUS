
magpie.grids.polargrid
^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.grids.polargrid(nphi, nr, rmin=0., rmax=1., phimin=0., phimax=2*np.pi, return1d=False)

    Constructs a polar coordinate grid.

    :Parameters:
      nphi : int
          Number of divisions along the p-axis.
      nr : int
          Number of divisions along the r-axis.
      rmin : float, optional
          Minimum radial value, default=0.
      rmax : float, optional
          Maximum radial value, default=1.
      phimin : float, optional
          Minimum phi value, default=0.
      phimax : float, optional
          Maximum phi value, default=2pi.
      return1d : bool, optional
          Returns 1d mid-points.

    :Returns:
      p2d, r2d : 2darray
          Phi and radial grid points in 2darray.
      pmid, rmid : array, optional
          The mid-point of the polar grid.
