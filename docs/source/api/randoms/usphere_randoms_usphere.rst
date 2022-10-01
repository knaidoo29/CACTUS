
magpie.randoms.randoms_usphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.randoms.randoms_usphere(size, phi_min=0., phi_max=2.*np.pi, theta_min=0., theta_max=np.pi)

    Random points on the unit sphere or more generally across the surface of a sphere. The
    default will give randoms on the full unit sphere.

    Coordinate convention:
      * phi lies in the range [0, 2pi]
      * theta lies in the rang [0, pi].

    :Parameters:
      size : int
          Number of randoms to generate.
      phi_min : float
          Minimum longitude in radians.
      phi_max : float
          Maximum longitude in radians.
      theta_min : float
          Minimum latitude in radians.
      theta_max : float
          Maximum longitude in radians.

    :Returns:
      phi : array
          Random phi.
      theta : array
          Random theta.
