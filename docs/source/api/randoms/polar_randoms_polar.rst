
magpie.randoms.randoms_polar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.randoms.randoms_polar(size, r_min=0., r_max=1., phi_min=0., phi_max=2.*np.pi)

    Generates randoms for polar coordinates. Default will produce randoms within
    a unit circle. This can be specified to a ring segment, i.e. with inner radius
    r_min and outer radius r_max and specifying the angle of the ring segment.

    :Parameters:
      size : int
          Number of randoms.
      r_min : float
          Minimum r.
      r_max : float
          Maximum r.
      phi_min : float
          Minimum phi.
      phi_max : float
          Maximum phi.

    :Returns:
      r : array
          Random radial coordinates.
      phi : array
          Random phi coordinates.
