
magpie.pixels.bin_pix
^^^^^^^^^^^^^^^^^^^^^


.. function:: magpie.pixels.bin_pix(pixID, pixlen, weights=None)

    Bin weights by pixel index.

    :Parameters:
      pixID : array
          Pixel indices.
      pixlen : int
          Size of the pixel grid.
      weights : array, optional
          Pixel index weights, if None is given this is assumed to be one.

    :Returns:
      pix : array
          Binned pixel values.
