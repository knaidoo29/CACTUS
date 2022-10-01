
magpie.coords.sphere2lonlat
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.sphere2lonlat(theta)

    Converts the spherical coordinates theta to the longitude and latitude
    convention (where theta lies [-pi/2., pi/2.].

    :Parameters:
      theta : array
          Latitude given in the range [0., pi] where theta = 0 at the north pole.

    :Returns:
      Latitude : array
          Latitude given in the range [-pi/2, pi/2].
