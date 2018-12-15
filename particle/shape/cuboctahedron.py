# -*- coding: utf-8 -*-

import numpy as np

from ..core import mathfunctions as mf
from .polyhedron import polyhedron, check_poly_validity

class cuboctahedron(polyhedron):
    '''cuboctahedron class'''

    _shape_name = 'cuboctahedron'
    n_vertices = 12
    n_edges = 24
    n_faces = 14
    n_shapeinfo = n_vertices + n_edges + n_faces

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

        self.gamma = [62.1, 64.1, 67.3] # (111), (110), (100)
        self.gamma_max = max(self.gamma)

        n_111 = mf.MillerNormalVectors_111()
        n_100 = mf.MillerNormalVectors_100()
        _NN = np.vstack((n_111, n_100))

        _DD = np.zeros(len(_NN), dtype=float)
        _DD[0:8] = np.sqrt(2)/np.sqrt(3.0)
        _DD[8:14] = 1/np.sqrt(2)

        _GG = np.zeros(len(_NN), dtype=float)
        _GG[0:8] = self.gamma[0]
        _GG[8:14] = self.gamma[2]

        _a_range = a*1.1

        # Initialize polyhedron class
        kwargs["shape_name"] = self._shape_name
        kwargs["GG"] = _GG
        kwargs["a_range"] = _a_range
        polyhedron.__init__(self, a,
                            _NN, _DD, **kwargs)
        self._kwargs = kwargs

    def vertices(self):
        """vertices(self) -> numpy.2darray
        return the vertices.
        """
        a = self.a
        out = np.zeros((3, 12))
        out[0, :] = a/np.sqrt(2)*np.array([1,0,-1,0,1,-1,-1,1,1,0,-1,0])
        out[1, :] = a/np.sqrt(2)*np.array([0,1,0,-1,1,1,-1,-1,0,1,0,-1])
        out[2, :] = a/np.sqrt(2)*np.array([1,1,1,1,0,0,0,0,-1,-1,-1,-1])
        return out

    def midpoints(self):
        """midpoints(self) -> numpy.2darray
        return the midpoints of edges.
        """
        # Parameters.
        a = self.a
        edges = 24
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

        # Between top and medium.
        for ii in range(vrep):
            out[:, count] = np.array([(x_top[ii] + x_med[ii])/2, (y_top[ii] + y_med[ii])/2, (z_top[ii] + z_med[ii])/2])
            out[:, count+1] = np.array([(x_top[ii+1] + x_med[ii])/2, (y_top[ii+1] + y_med[ii])/2, (z_top[ii+1] + z_med[ii])/2])
            count = count + 2

        out[:, count] = np.array([(x_top[3] + x_med[3])/2, (y_top[3] + y_med[3])/2, (z_top[3] + z_med[3])/2])
        out[:, count+1] = np.array([(x_top[0] + x_med[3])/2, (y_top[0] + y_med[3])/2, (z_top[0] + z_med[3])/2])
        count = count + 2

        # Between medium and bottom.
        for ii in range(vrep):
            out[:, count] = np.array([(x_bot[ii] + x_med[ii])/2, (y_bot[ii] + y_med[ii])/2, (z_bot[ii] + z_med[ii])/2])
            out[:, count+1] = np.array([(x_bot[ii+1] + x_med[ii])/2, (y_bot[ii+1] + y_med[ii])/2, (z_bot[ii+1] + z_med[ii])/2])
            count = count + 2

        out[:, count] = np.array([(x_bot[3] + x_med[3])/2, (y_bot[3] + y_med[3])/2, (z_bot[3] + z_med[3])/2])
        out[:, count+1] = np.array([(x_bot[0] + x_med[3])/2, (y_bot[0] + y_med[3])/2, (z_bot[0] + z_med[3])/2])
        count = count + 2

        # Bottom.
        for ii in range(vrep):
            out[:, count] = np.array([(x_bot[ii] + x_bot[ii+1])/2, (y_bot[ii] + y_bot[ii+1])/2, (z_bot[ii] + z_bot[ii+1])/2])
            count = count + 1

        out[:, count] = np.array([(x_bot[3] + x_bot[0])/2, (y_bot[3] + y_bot[0])/2, (z_bot[3] + z_bot[0])/2])

        return out

    def info(self):
        """info(self) -> dict
        return the information of this class.
        """
        return dict(shape_name=self._shape_name, a=self.a, NN=self._NN, DD=self.DD, kwargs=self._kwargs)
