=========================================
CaCTus: Cosmic web Classification Toolkit
=========================================

.. .. image:: https://badge.fury.io/py/magpie-pkg.svg
..    :target: https://badge.fury.io/py/magpie-pkg

.. .. image:: https://anaconda.org/knaidoo29/magpie-pkg/badges/version.svg
..    :target: https://anaconda.org/knaidoo29/magpie-pkg

.. image:: https://readthedocs.org/projects/cactus-doc/badge/?version=latest
  :target: https://cactus-doc.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. .. image:: https://circleci.com/gh/knaidoo29/magpie/tree/master.svg?style=svg
..    :target: https://circleci.com/gh/knaidoo29/magpie/tree/master

.. .. image:: https://codecov.io/gh/knaidoo29/magpie/branch/development/graph/badge.svg?token=P7H8FAJT43
..    :target: https://codecov.io/gh/knaidoo29/magpie

.. .. image:: https://anaconda.org/knaidoo29/magpie-pkg/badges/license.svg
..    :target: https://anaconda.org/knaidoo29/magpie-pkg


+---------------+-----------------------------------------+
| Author        | Krishna Naidoo                          |
+---------------+-----------------------------------------+
| Version       | 0.0.0                                   |
+---------------+-----------------------------------------+
| Repository    | https://github.com/knaidoo29/cactus     |
+---------------+-----------------------------------------+
| Documentation | https://cactus-doc.readthedocs.io/      |
+---------------+-----------------------------------------+

.. warning::
  CaCTus is currently in development. Functions and classes may change. Use with caution.

Contents
========

* `Introduction`_
* `Tutorials and API`_
* `Dependencies`_
* `Installation`_
* `Support`_
* `Version History`_

Introduction
============

Tutorials and API
=================

.. toctree::
  :maxdepth: 2

  api


Dependencies
============

CaCTus is being developed in Python 3.9 but should work on all versions >=3.4. CaCTus
is written mostly in python.

The following Python modules are required:

* `numpy <http://www.numpy.org/>`_
* `scipy <http://scipy.org/>`_
* `matplotlib <ttps://matplotlib.org/>`_
* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_
* `mpi4py-fft <https://mpi4py-fft.readthedocs.io/en/latest/>`_
* `FIESTA <https://github.com/knaidoo29/FIESTA>`_ (install developer branch)
* `MAGPIE <https://github.com/knaidoo29/MAGPIE>`_ (install developer branch)
* `SHIFT <https://github.com/knaidoo29/SHIFT>`_

For testing you will require `nose <https://nose.readthedocs.io/en/latest/>`_ or
`pytest <http://pytest.org/en/latest/>`_ .


Installation
============

CaCTus can be installed in a variety of ways -- using ``conda``, ``pip`` or by
directly cloning the repository. If you are having trouble installing or
running MAGPIE we recommend using the conda install as this will setup the
environment.

.. #. Using ``conda``::

..    conda install -c knaidoo29 cactus

.. #. Using ``pip``::

..    pip install cactus

#. By cloning the github repository::

    git clone https://github.com/knaidoo29/cactus.git
    cd magpie
    python setup.py build
    python setup.py install

Once this is done you should be able to call CaCTus from python:

.. code-block:: python

    import cactus

Support
=======

If you have any issues with the code or want to suggest ways to improve it please
open a new issue (`here <https://github.com/knaidoo29/cactus/issues>`_) or
(if you don't have a github account) email krishna.naidoo.11@ucl.ac.uk.


Version History
===============

* **Version 0.0.0**:
