
magpie.randoms.randoms_healpix_pixel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.randoms.randoms_healpix_pixel(size, pix, nside)

    Returns roughly `size` number of randoms inside a HEALPix pixel.

    :Parameters:
      size : int
          Average number of randoms per pixel.
      p : int
          Pixel identifier for healpix map.
      nside : int
          Nside of the healpix map.
    
    :Returns:
      phi : array
          Random phi (latitude angle) in radians.
      theta : array
          Random theta (longitude angle) in radians.
