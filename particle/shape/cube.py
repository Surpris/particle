# -*- coding: utf-8 -*-

import numpy as np

from ..core import mathfunctions as mf
from .polyhedron import polyhedron, check_poly_validity

class cube(polyhedron):
    '''cube class'''
    
    _shape_name = 'cube'
    n_vert = 8
    n_edges = 12
    n_faces = 6
    def __init__(self, a, *args, **kwargs):
        """__init__(self, a, *args, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        a      : float
            length of edge
        args   : options
        kwargs : options
            center  : 3-element list or numpy.1darray
                the center of a particle
            density : float
                the density of a particle
            euler   : 3-element list or numpy.1darray
                Euler angle for rotation
            permute : 3-element list or numpy.1darray
                direction for plotting
            chamfer : float
                degree of chamferring
            rand    : bool
                flag for random-depth chamferring
        """
        self.__shape_name = 'cube'

        self.gamma = [62.1, 64.1, 67.3] # (111), (110), (100)
        _NN = mf.MillerNormalVectors_100()
        _DD = np.ones(len(_NN), dtype=float)*0.5
        _GG = np.ones(len(_NN), dtype=float)*self.gamma[2]

        _a_range = a*np.sqrt(2.)*1.1

        # Initialize polyhedron class
        kwargs["shape_name"] = self._shape_name
        kwargs["GG"] = _GG
        kwargs["a_range"] = _a_range
        polyhedron.__init__(self, a, _NN, _DD, **kwargs)

        self._kwargs = kwargs
