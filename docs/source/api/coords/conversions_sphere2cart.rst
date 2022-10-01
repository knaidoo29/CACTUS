
magpie.coords.sphere2cart
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.sphere2cart(r, phi, theta, center=[0., 0., 0.])

    Converts spherical polar coordinates into cartesian coordinates.

    :Parameters:
      r : array
          Radial distance.
      phi : array
          Longitudinal coordinates (radians = [0, 2pi]).
      theta : array
          Latitude coordinates (radians = [0, pi]).
      center : list
          Center point of spherical polar coordinate grid.

    :Returns:
      x, y, z : array
          Euclidean coordinates.
