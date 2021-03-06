particle module
=====

This module aims to calculate scattered images from a given shape by using the Multi-slice Fourier transform (MSFT) method.   

# Requirement
This module has dependencies on the following modules.

* Python (>= 3.4)
    + Probably this module is available on Python 2.7 (not tested).
* NumPy (>= 1.11.2)
* Matplotlib (>= 1.5.1)
    + mpl_toolkits.mplot3d
* datetime
* importlib

If you want to do calculation with `pyfftw` or `pycuda`, the following modules are also required.

* pyfftw (for pyfftw)
* multiprocessing (for pyfftw)
* pycuda (for pycuda)
* skcuda (for pycuda)

# Installation
Currently, installing this module requires the following steps:

1. Download this module as a zip (or tar.gz) file.
1. Unzip the file to to any directory.
1. Run `python setup.py install` at the directory.

# Hierarchical structure of modules
```
/particle
    /core: Core modules.
        ensemble.py     : Class for multi-particle system.
        mathfuntions.py : functions related to calculation.
        particle.py     : Class for one-particle system. (**deprecated from v1.2**)
        slicefft.py     : Class for MSFT.
        space.py        : Class for meshgrids in space.
    /shape: Shape classes. Refer to the section "Shapes".
        shapeslice.py   : functions related to slicing.
```

# Shapes
The following shapes are available:

* cube
* cuboctahedron
* dodecahedron (under construction)
* hailstone / hailstone_with_sphere (**deprecated from 1.2**)
  + The primary purpose of `hailstone_with_sphere` is to randomly generate a hail model with daughter particles centered on the surface of the mother particle.
  + `hailstone` generates a hail model using arguments (list of) of information on types, sizes and positions of a mother particle and daughter particles.
* icosahedron
* polyhedron
* sphere
* spheroid
* wulffpolyhedron

## How to add a shape class
1. Add a description of the required shape (`myshape`) to the `/shape` directory with reference to other classes.
2. Add the following two sentences to `__init __. Py` in the `/shape` directory.
  * `from . import myshape`
  * `from .myshape import *`
3. Add an `elif` statement to the` particle .__ MakeParticle` function so that it can be called from the `particle` class.

## History of update
* 1.0 : first commission
* 1.0.1 : fix bugs
* 1.0.2 : fix bugs
* 1.1 : update documentation, add density-weighting mode
* 1.1.1 : fix bugs