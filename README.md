# CaCTus: Cosmic web Classification Toolkit

[![Documentation Status](https://readthedocs.org/projects/cactus-doc/badge/?version=latest)](https://cactus-doc.readthedocs.io/en/latest/?badge=latest)

|               |                                       |
|---------------|---------------------------------------|
| Author        | Krishna Naidoo                        |               
| Version       | 1.0.0                                 |
| Repository    | https://github.com/knaidoo29/cactus   |
| Documentation | https://cactus-doc.readthedocs.io/    |

> **_Warning:_** In development!

## Introduction

The **Cosmic web Classification Toolkit (CaCTus)** is an open source python package for classifying cosmic web environments from simulations. The package will allow users to reliably classify cosmic web structures and to compare different techniques. For the time being we implement the following Hessian based approaches:

* T-Web [to do]
* V-Web [to do]
* Nexus+ [to do]

Inputs can either directly take the simulation particles themselves or use a pre-calculated density (or velocity) field. Note, this implements field estimation techniques from another package called [FIESTA](https://github.com/knaidoo29/FIESTA).

## Dependencies

* [numpy](http://www.numpy.org/)
* [scipy](https://scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
* [mpi4py-fft](https://mpi4py-fft.readthedocs.io/en/latest/)
* [FIESTA](https://github.com/knaidoo29/FIESTA) (install developer branch)
* [MAGPIE](https://github.com/knaidoo29/MAGPIE) (install developer branch)
* [MPIutils](https://github.com/knaidoo29/MPIutils)
* [SHIFT](https://github.com/knaidoo29/SHIFT)

## Installation

## Citing

## Support

If you have any issues with the code or want to suggest ways to improve it please
open a new issue ([here](https://github.com/knaidoo29/cactus/issues)) or (if you
don't have a github account) email _krishna.naidoo.11@ucl.ac.uk_.

## Version History

* **Version 0.0**:
