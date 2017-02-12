# -*- coding: utf-8 -*-

import numpy as np

from .. import core
from ..core import mathfunctions as mf

from .polyhedron import *

class icosahedron(polyhedron):
    '''
    Icosahedronを与えるクラス。
    内部変数として一辺の長さ、オイラー回転角等を持つ。

    __init__中の引数について。
        a       : length of edge
        kwargs  : options
    '''
    _shape_name = 'icosahedron'
    n_vertices = 12
    n_edges = 30
    n_faces = 20
    n_shapeinfo = n_vertices + n_edges + n_faces

    def __init__(self, a, *args, **kwargs):
        """
            クラスの初期化。
        """

        self.gamma = [62.1, 64.1, 67.3] # (111), (110), (100)
        self.gamma_max = max(self.gamma)

        phi = 2*np.cos(np.pi/5)
        self.ki2 = np.array([[phi-1,0,phi],[phi-1,0,-phi],[phi,phi-1,0],[-phi,phi-1,0],[0,phi,phi-1],[0,-phi,phi-1]])/np.linalg.norm(np.array([phi-1,0,phi]))

        NN_buff = np.array([self.ki2[0,:],self.ki2[1,:],self.ki2[2,:],
                            self.ki2[3,:],self.ki2[4,:],self.ki2[5,:],
                            -self.ki2[0,:],-self.ki2[1,:],-self.ki2[2,:],
                            -self.ki2[3,:],-self.ki2[4,:],-self.ki2[5,:]])
        n_111 = mf.MillerNormalVectors_111()
        _NN = np.vstack((n_111, NN_buff))

        _DD = np.ones(len(_NN))
        _DD[0:8] = (1+phi)/2/np.sqrt(3.0)
        _DD[8:20] = phi**2/2/np.linalg.norm(np.array([phi-1,0,phi]))
        _GG = np.ones(len(_NN))

        _a_range = a*np.sqrt(1.+phi**2)/2.*1.1

        # Initialize polyhedron class
        kwargs["shape_name"] = self._shape_name
        kwargs["GG"] = _GG
        kwargs["a_range"] = _a_range
        polyhedron.__init__(self, a,
                            _NN, _DD, **kwargs)

        self._kwargs = kwargs

    def vertices(self, a):
        coef = 2
        phi = 2*np.cos(np.pi/5)
        vert = np.zeros((3, 12))
        vert[0, :] = a/coef*np.array([phi,0,-phi,0,1,-1,-1,1,phi,0,-phi,0])
        vert[1, :] = a/coef*np.array([0,1,0,-1,phi,phi,-phi,-phi,0,1,0,-1])
        vert[2, :] = a/coef*np.array([1,phi,1,phi,0,0,0,0,-1,-phi,-1,-phi])
        return vert

    def midpoints(self, a):
        # Parameters.
        edges = 30
        out = np.zeros((3, edges))

        # Vertices of cuoctahedron.
        vert = self.vertices(a)

        # Separate the vertices into three part : top, medium and bottom.
        x_top = vert[0, 0:4]
        x_med = vert[0, 4:8]
        x_bot = vert[0, 8:]
        y_top = vert[1, 0:4]
        y_med = vert[1, 4:8]
        y_bot = vert[1, 8:]
        z_top = vert[2, 0:4]
        z_med = vert[2, 4:8]
        z_bot = vert[2, 8:]
        vrep = 3

        # Calculate the edges.
        count = 0
        # Top.
        for ii in range(vrep):
            out[:, ii] = np.array([(x_top[ii] + x_top[ii+1])/2, (y_top[ii] + y_top[ii+1])/2, (z_top[ii] + z_top[ii+1])/2])
            count = count + 1

        out[:, count] = np.array([(x_top[3] + x_top[0])/2, (y_top[3] + y_top[0])/2, (z_top[3] + z_top[0])/2])
        count = count + 1
        out[:, count] = np.array([(x_top[3] + x_top[1])/2, (y_top[3] + y_top[1])/2, (z_top[3] + z_top[1])/2])
        count = count + 1

        # Between top and medium.
        for ii in range(vrep):
            out[:, count] = np.array([(x_top[ii] + x_med[ii])/2, (y_top[ii] + y_med[ii])/2, (z_top[ii] + z_med[ii])/2])
            out[:, count+1] = np.array([(x_top[ii+1] + x_med[ii])/2, (y_top[ii+1] + y_med[ii])/2, (z_top[ii+1] + z_med[ii])/2])
            count = count + 2

        out[:, count] = np.array([(x_top[3] + x_med[3])/2, (y_top[3] + y_med[3])/2, (z_top[3] + z_med[3])/2])
        out[:, count+1] = np.array([(x_top[0] + x_med[3])/2, (y_top[0] + y_med[3])/2, (z_top[0] + z_med[3])/2])
        count = count + 2

        # Medium.
        for ii in range(2):
            out[:, count] = np.array([(x_med[2*ii] + x_med[2*ii+1])/2, (y_med[2*ii] + y_med[2*ii+1])/2, (z_med[2*ii] + z_med[2*ii+1])/2])
            count = count + 1


        # Between medium and bottom.
        for ii in range(vrep):
            out[:, count] = np.array([(x_bot[ii] + x_med[ii])/2, (y_bot[ii] + y_med[ii])/2, (z_bot[ii] + z_med[ii])/2])
            out[:, count+1] = np.array([(x_bot[ii+1] + x_med[ii])/2, (y_bot[ii+1] + y_med[ii])/2, (z_bot[ii+1] + z_med[ii])/2])
            count = count + 2

        out[:, count] = np.array([(x_bot[3] + x_med[3])/2, (y_bot[3] + y_med[3])/2, (z_bot[3] + z_med[3])/2])
        out[:, count+1] = np.array([(x_bot[0] + x_med[3])/2, (y_bot[0] + y_med[3])/2, (z_bot[0] + z_med[3])/2])
        count = count + 2

        # Between top and bottom.
        for ii in np.arange(0,4,2):
            out[:, count] = np.array([(x_bot[ii] + x_top[ii])/2, (y_bot[ii] + y_top[ii])/2, (z_bot[ii] + z_top[ii])/2])
            count = count + 1


        # Bottom.
        for ii in range(vrep):
            out[:, count] = np.array([(x_bot[ii] + x_bot[ii+1])/2, (y_bot[ii] + y_bot[ii+1])/2, (z_bot[ii] + z_bot[ii+1])/2])
            count = count + 1

        out[:, count] = np.array([(x_bot[3] + x_bot[0])/2, (y_bot[3] + y_bot[0])/2, (z_bot[3] + z_bot[0])/2])
        out[:, count+1] = np.array([(x_bot[3] + x_bot[1])/2, (y_bot[3] + y_bot[1])/2, (z_bot[3] + z_bot[1])/2])

        return out

    def info(self):
        """
            particleshapeメソッドで生成するための情報を返す
        """
        return dict(shape_name=self._shape_name, a=self.a, NN=self._NN, DD=self.DD, kwargs=self._kwargs)
