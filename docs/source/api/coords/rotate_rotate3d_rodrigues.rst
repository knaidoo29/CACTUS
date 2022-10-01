
magpie.coords.rotate3d_Rodrigues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.coords.rotate3d_Rodrigues(x, y, z, k, dphi)

    Rotates points in 3D cartesian coordinates around an axis defined by the
    unit vector k.

    :Parameters:
      x, y, z : float or array
          Cartesian coordinates.
      k : array
          k is a unit vector around which points will be rotated by an angle
          dphi.
      dphi : float
          Rodrigues rotation angle around the unit vector k.

    :Returns:
      xrot, yrot, zrot : array or float
          Rotated x, y and z coordinates.
