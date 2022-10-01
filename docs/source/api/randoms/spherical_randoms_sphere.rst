
magpie.randoms.randoms_sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.randoms.randoms_sphere(size, r_min=0., r_max=1., phi_min=0., phi_max=2*np.pi, theta_min=0., theta_max=np.pi)

    Random points inside a sphere (default will give randoms within a unit sphere).
    You can specify the inner and outer radii to get randoms in a shell and the region
    on the sky.

    Coordinate convention:
      * phi lies in the range [0, 2pi]
      * theta lies in the rang [0, pi].

    :Parameters:
      size : int
          Number of randoms to generate.
      r_min : float
          Minimum radius.
      r_max : float
          Maximum radius.
      phi_min : float
          Minimum longitude in radians.
      phi_max : float
          Maximum longitude in radians.
      theta_min : float
          Minimum latitude in radians.
      theta_max : float
          Maximum longitude in radians.

    :Returns:
      r : array
          Random r.
      phi : array
          Random phi.
      theta : array
          Random theta.
