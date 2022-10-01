
magpie.pixels.pos2pix_polar
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.pixels.pos2pix_polar(phi, r, nphi, nr, phimin=0., phimax=2*np.pi, rmin=0., rmax=1.)

    Angular and radial coordinates to polar grid.

    :Parameters:
      phi, r : float or array
          Angular and radial coordinates.
      nphi, nr : int
          Grid size along the angular and radial axes.
      phimin, phimax : float, optional
          Angular ranges.
      rmin, rmax : float, optional
          Radial ranges.

    :Returns:
      pixID : int or array
          The pixel the points are located in the polar coordinate grid.
