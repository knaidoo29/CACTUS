# CaCTus: Cosmic web Classification Toolkit

[![Documentation Status](https://readthedocs.org/projects/cactus-doc/badge/?version=latest)](https://cactus-doc.readthedocs.io/en/latest/?badge=latest)

|               |                                       |
|---------------|---------------------------------------|
| Author        | Krishna Naidoo                        |               
| Version       | 0.3.0                                 |
| Repository    | https://github.com/knaidoo29/cactus   |
| Documentation | https://cactus-doc.readthedocs.io/    |

> **_Warning:_** In development!

## Introduction

The **Cosmic web Classification Toolkit (CaCTus)** is an open source python package for classifying cosmic web environments from simulations. The package allows a user to reliably classify cosmic web structures and to compare different techniques. For the time being we implement the following Hessian based approaches:

* T-Web
* V-Web
* Multiscale Morphology Filter (MMF):
  * NEXUS+
  * NEXUS_den
  * NEXUS_tidal
  * NEXUS_denlog
  * NEXUS_veldiv
  * NEXUS_velshear

As well as routines for computing density and velocity fields from a given catalogue:

* Particle mesh assignments:
  * Nearest grid point (NGP)
  * Cloud in cell (CIC)
  * Triangular shaped cloud (TSC)
* Smooth particle hydrodynamics (SPH)
* Delaunay tesselation field estimation (DTFE)

## Dependencies

* `Python`
* `Fortran`
* `OpenMPI`.

Python modules:

* [numpy](http://www.numpy.org/)
* [scipy](https://scipy.org/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

## Installation

Cloning the repository and running in the `cactus` repository:
```
python setup.py build
python setup.py install
```

## Citing

## Support

If you have any issues with the code or want to suggest ways to improve it please
open a new issue ([here](https://github.com/knaidoo29/cactus/issues)) or (if you
don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.

## Version History

* **Version 1**:
  * Density computation:
    * Particle mesh assignments: NGP, CIC and TSC.
    * SPH [todo]
    * DTFE
  * Velocity computation:
    * Particle mesh assignments: NGP, CIC and TSC. [todo]
    * SPH [todo]
    * DTFE [todo]
  * Hessian based approaches:
    * T-web
    * V-web [todo]
    * MMF:
      * NEXUS+
      * NEXUS_den
      * NEXUS_tidal [todo]
      * NEXUS_denlog [todo]
      * NEXUS_veldiv [todo]
      * NEXUS_velshear [todo]
