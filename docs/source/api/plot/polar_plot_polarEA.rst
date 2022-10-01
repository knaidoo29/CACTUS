
magpie.plot.plot_polarEA
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: magpie.plot.plot_polarEA(pix, nr=None, rmax=1., base_nphi=3, pixID=None, proj='cart', center=[0., 0.], steps=12, ax=None, vmin=None, vmax=None, cmap=plt.cm.viridis)

    Plots pixels from a polar equal area grid.

    :Parameters:
      pix : array
          Pixel values.
      nr : int, optional
          Enter the number of radial bins, if None this will be calculated from
          the size of pix and base_nphi.
      rmax : float, optional
          Maximum radius, default=1.
      base_nphi: int, optional
          Number of pixels in the first ring, default=3.
      pixID : int, optional
          Enter the pixel indices of pix if pix is a subset of the pixel grid.
          Useful if you wish to plot only a subset of the pixel grid.
      proj : str, optional
          Output coordinate projection. Either 'cart' for cartesian or 'polar'
          for polar projection.
      center : list, optional
          Center point of the polar grid, only relevant for cartesian
          projection, default at the origin.
      steps : int, optional
          Number of steps in the polygon, default=12.
      ax : obj, optional
          Matplotlib axis, if None this is plotted onto the last figure.
      vmin, vmax : float, optional
          Ranges for pixel color.
      cmap : obj, optional
          Enter matplotlib colormap object, default=plt.cm.viridis.
