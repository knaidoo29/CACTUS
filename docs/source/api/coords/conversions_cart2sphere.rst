
magpie.coords.cart2sphere
^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.cart2sphere(x, y, z, center=[0., 0., 0.])

    Return polar coordinates for a given set of cartesian coordinates.

    :Parameters:
      x : array
          x coordinate.
      y : array
          y coordinate.
      center : list
          Center point of polar coordinate grid.

    :Returns:
      r : array
          Radial coordinates.
      phi : array
          Phi coordinates [0, 2pi].
      theta : array
          Theta coordinates [0, pi].
