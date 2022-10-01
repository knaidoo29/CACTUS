
magpie.coords.lonlat2sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.lonlat2sphere(latitude)

    Converts from latitude to spherical coordinate convention.

    :Parameters:
      Latitude : array
          Latitude given in the range [-pi/2, pi/2].

    :Returns:
      theta : array
          Latitude given in the range [0., pi] where theta = 0 at the north pole.
