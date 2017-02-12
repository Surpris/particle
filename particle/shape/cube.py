# -*- coding: utf-8 -*-

import numpy as np

from .. import core
from ..core import mathfunctions as mf

from .polyhedron import *

class cube(polyhedron):
    '''
    立方体を与えるクラス。
    内部変数として一辺の長さ、オイラー回転角等を持つ。

    __init__中の引数について。
        a       : length of edge
        kwargs  : options
    '''
    n_vert = 8
    n_edges = 12
    n_faces = 6
    def __init__(self, a, *args, **kwargs):
        """
            クラスの初期化。
        """
        self.__shape_name = 'cube'

        self.gamma = [62.1, 64.1, 67.3] # (111), (110), (100)
        _NN = mf.MillerNormalVectors_100()
        _DD = np.ones(len(_NN), dtype=float)*0.5
        _GG = np.ones(len(_NN), dtype=float)*self.gamma[2]

        _a_range = a*np.sqrt(2.)*1.1

        # Initialize polyhedron class
        kwargs["shape_name"] = self.__shape_name
        kwargs["GG"] = _GG
        kwargs["a_range"] = _a_range
        polyhedron.__init__(self, a,
                            _NN, _DD, **kwargs)

        self._kwargs = kwargs
