# -*- coding: utf-8 -*-

import numpy as np

from ..core import mathfunctions as mf
from .shapeslice import shapeslice


class dodecahedron(shapeslice):
    '''dodecahedron class (under construction)'''

    def __init__(self, a, *args, **kwargs):
        """__init__(self, a, *args, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        a      : float
            length of edge
        args   : options
        kwargs : options
            euler   : 3-element list or numpy.1darray
                Euler angle for rotation
            permute : 3-element list or numpy.1darray
                direction for plotting
            chamfer : float
                degree of chamferring
            rand    : bool
                flag for random-depth chamferring
        """
        self.__shape_name = 'dodecahedron'
        # Get information from kwargs
        euler = kwargs.get('euler')
        permute = kwargs.get('permute')
        if kwargs.get('chamfer') is not None:
            chamfer = kwargs.get('chamfer')
        elif kwargs.get('mid') is not None:
            chamfer = kwargs.get('mid')
        else:
            chamfer = 1.0
        rand = False if kwargs.get('rand') is None else kwargs.get('rand')

        self.gamma = [62.1, 64.1, 67.3]  # (111), (110), (100)
        self.gamma_max = max(self.gamma)

        phi = 2*np.cos(np.pi/5)
        self.ki2 = np.array([
            [phi, 1, 0], [0, phi, 1], [1, 0, phi],
            [-phi, 1, 0], [0, -phi, 1], [1, 0, -phi]
        ])/np.linalg.norm(np.array([phi, 1, 0]))

        self.__NN = np.array([
            self.ki2[0, :], self.ki2[1, :], self.ki2[2, :],
            self.ki2[3, :], self.ki2[4, :], self.ki2[5, :],
            -self.ki2[0, :], -self.ki2[1, :], -self.ki2[2, :],
            -self.ki2[3, :], -self.ki2[4, :], -self.ki2[5, :]
        ])

        self.DD = 0.5*a / \
            np.linalg.norm(np.array([phi, 1, 0]))*np.ones(len(self.__NN))

        self.center = [0., 0., 0.] if kwargs.get(
            'center') is None else kwargs.get('center')

        self.chamfer = chamfer
        self.a = a + 0.0
        self.a_range = self.a*2*1.1

        # Random chamferring
        rrr = np.random.rand(200)
        sigma = 0.1
        med = 0.5
        e_rand = np.exp(-(rrr-med)**2/2./sigma**2)
        self.rand = np.zeros(
            30, dtype=float) if rand is False else e_rand[0:30]

        # Permutation of normal vectors
        self.perm = [0, 1, 2] if permute is None else permute

        # Initialize shapeslice class.
        self.NN = self.__NN + 0.0
        shapeslice.__init__(self, self.__shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm)

    def EulerRot(self, euler):
        self.NN = mf.EulerRotation(self.__NN, euler, 1)
        self.UpdSlice()

    def UpdSlice(self):
        shapeslice.__init__(self, self.__shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm)

    def shape_name(self):
        return self.__shape_name + ""
