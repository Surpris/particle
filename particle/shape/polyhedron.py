# -*- coding: utf-8 -*-

import numpy as np

from ..core import mathfunctions as mf
from ..core.space import space
from .shapeslice import shapeslice

class polyhedron(shapeslice):
    """polyhedron class.
    This class has basic functions of each polyhedron.
    Based on the list of the distance "DD" from the center defined as an arbitrary surface vector "NN",
    a region enclosed by them is given as the polyhedron.
    The surface energy "GG" of the surface may be adjusted if required.
    """
    def __init__(self, a, NN, DD, *args, **kwargs):
        """__init__(self, a, NN, DD, *args, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        a      : float
            length of edge
        NN     : list or numpy.2darray
            normal vectors
        DD     : list or numpy.2darray
            discante from the center of a polyhedron.
            "DD" is supposd to be given by the ratios to "a".
        args   : options
        kwargs : options
            center : 3-element list or numpy.1darray
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
        self._shape_name = "polyhedron" if kwargs.get("shape_name") is None else kwargs.get("shape_name")
        # Get information from kwargs
        euler = kwargs.get('euler')
        if euler is None:
            euler = np.array([0, 0, 0])
        permute = kwargs.get('permute')

        if type(NN) == list:
            self._NN = np.array(NN)
        else:
            self._NN = NN.copy()
        if type(DD) == list:
            self.DD = np.array(DD)
        else:
            self.DD = DD.copy()

        if kwargs.get("distance_rate") is not None:
            _distance_rate = kwargs.get("distance_rate")
            if type(_distance_rate) == list:
                _distance_rate = np.array(_distance_rate)
            if type(_distance_rate) in [float, np.float32, np.float64]:
                self.DD = (1. - _distance_rate) * self.DD
            else:
                if type(_distance_rate) != np.ndarray:
                    raise TypeError("Invalid type for `distance_rate`.")
                elif len(_distance_rate) != len(self.DD):
                    raise ValueError("The length of `distance_rate` is not equal to `DD`.")
                self.DD = (1. - _distance_rate) * self.DD

        GG = kwargs.get("GG")
        if GG is None:
            self.GG = np.zeros(len(self.DD), dtype=float)
        elif type(GG) == list:
            self.GG = np.array(GG)
        elif type(GG) == np.ndarray and len(GG.shape) == 1:
            self.GG = GG.copy()
        else:
            raise TypeError("`GG` must be an 1-D array.")
        
        self.gamma_max = self.GG.max()
        self.center = [0., 0., 0.] if kwargs.get('center') is None else kwargs.get('center')

        # Check validity of surface
        self._check_poly_validity()

        self.a = a
        self.a_range = self.a*1.5 if kwargs.get("a_range") is None else kwargs.get("a_range")

        # Permutation of normal vectors
        self.perm = [0, 1, 2] if permute is None else permute

        # Chamfer edges (option)
        _chamfer_edge_rate = kwargs.get("chamfer_edge_rate")
        _chamfer_vertex_rate = kwargs.get("chamfer_vertex_rate")
        if _chamfer_edge_rate is not None or _chamfer_vertex_rate is not None:
            if _chamfer_edge_rate is not None:
                self.chamfer_edge_rate = _chamfer_edge_rate
                self.chamferring("e", _chamfer_edge_rate)
            if _chamfer_vertex_rate is not None:
                self.chamfer_vertex_rate = _chamfer_vertex_rate
                self.chamferring("v", _chamfer_vertex_rate)
        else:
            """ Old format """
            if kwargs.get('chamfer') is not None:
                chamfer = kwargs.get('chamfer')
            elif kwargs.get('mid') is not None:
                chamfer = kwargs.get('mid')
            else:
                chamfer = 1.0
            self.chamfer = chamfer
            rand = False if kwargs.get('rand') is None else kwargs.get('rand')
            if self.chamfer < 1.0:
                _mid = self.midpoints(1.).transpose()
                _dd = []
                _gg = []

                # Random chamferring
                rrr = np.random.rand(200)
                sigma = 0.1
                med = 0.5
                e_rand = np.exp(-(rrr-med)**2/2./sigma**2)
                self.rand = np.ones(_mid.shape[0], dtype=float) if rand is False else e_rand[0:_mid.shape[0]]

                for ii in range(_mid.shape[0]):
                    _leng = (1.-(1.-self.chamfer)*self.rand[ii])*np.linalg.norm(_mid[ii,:])
                    _mid[ii,:] /= np.linalg.norm(_mid[ii,:])
                    _dd.append(_leng)
                    _gg.append(self.gamma_max*_leng)
                self._NN = np.vstack((self._NN, _mid))
                self.DD = np.hstack((self.DD, np.array(_dd)))
                self.GG = np.hstack((self.GG, np.array(_gg)))

        # Initialize shapeslice class.
        self.NN = self._NN.copy()
        shapeslice.__init__(self, self._shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm, center=self.center, 
                            density=kwargs.get("density"))

        # Euler rotation.
        self.EulerRot(euler)

        self._kwargs = kwargs

    def EulerRot(self, euler):
        """EulerRot(self, euler) -> None
        Rotate this object according to the Euler angle `euler`.

        Parameters
        ----------
        euler : 3-element list or numpy.1darray
            Euler angle (alpha, beta, gamma)
        """
        self.NN = mf.EulerRotation(self._NN, euler, 1)
        self.UpdSlice()

    def UpdSlice(self):
        """UpdSlice(self) -> None
        update this object.
        """
        shapeslice.__init__(self, self._shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm, center=self.center, 
                            density=self.density)

    def shape_name(self):
        return self._shape_name + ""

    def midpoints(self, *args, **kwargs):
        """midpoints(self, *args, **kwargs) -> None
        get the midpoints of each edge.
        This function is an abstract function.
        Implementation is done with child classes that inherit this class.
        """
        return None

    def vertices(self, *args, **kwargs):
        """vertices(self, *args, **kwargs) -> None
        get vertices of polyhedron.
        This function is an abstract function.
        Implementation is done with child classes that inherit this class.
        """
        return None

    def chamferring(self, v_or_e, chamfer_rate):
        """chamferring(self, v_or_e, chamfer_rate) -> None
        chamfer the edges and vertices.

        Parameters
        ----------
        v_or_e       : str
            vertices ("v") or edges ("e")
        chamfer_rate : numpy.1darray or list
            depth of chamferring.
            Actual distance of the new facet(s) from the origin is given by 
            `1.0 - chamfer_rate` times the original distance.
        """
        if v_or_e == "v":
            _chamf = self.vertices(1.)
        elif v_or_e == "e":
            _chamf = self.midpoints(1.)
        else:
            raise ValueError("`v_or_e` must be in ['v', 'e'].")
        if _chamf is None:
            raise NotImplementedError("Not implemented: `chamering`")
        _chamf = _chamf.transpose()
        _dd = []
        _gg = []

        if type(chamfer_rate) == list:
            chamfer_rate = np.array(chamfer_rate)
        elif type(chamfer_rate) in [float, np.float32, np.float64]:
            chamfer_rate = chamfer_rate * np.ones(_chamf.shape[0])

        if len(chamfer_rate) != _chamf.shape[0]:
            _ = "vertice" if v_or_e == "v" else "edges"
            raise ValueError("The length of `chamfer_rate` is not equal to  the number of `{0}`.".format(_))

        for ii in range(_chamf.shape[0]):
            _leng = (1. - chamfer_rate[ii])*np.linalg.norm(_chamf[ii,:])
            _chamf[ii,:] /= np.linalg.norm(_chamf[ii,:])
            _dd.append(_leng)
            _gg.append(self.gamma_max*_leng)

        self._NN = np.vstack((self._NN, _chamf))
        self.DD = np.hstack((self.DD, np.array(_dd)))
        self.GG = np.hstack((self.GG, np.array(_gg)))

    def info(self):
        """info(self) -> dict
        get information to make one object by `particleshape`.
        """
        return dict(shape_name=self._shape_name, a=self.a, NN=self._NN, DD=self.DD, kwargs=self._kwargs)

    def _check_poly_validity(self):
        """_check_poly_validity(self) -> None
        check the validity of a given face.
        
        Determination condition on whether to construct a polyhedron
        ------------------------------------------------------------
        A cube having a size three times larger than max (DD) is prepared.
        If at least one of the points belonging to that side is inside at least one surface, it is out.
        That is, it is OK if any side is outside of any side.
        """
        # Prepare the outer space.
        _xmax = max(self.DD)*3.
        _N = 10
        sp1 = space(_N, _xmax, ymax=_xmax, Ny=_N, zmax=_xmax, Nz=_N)
        dx = sp1.dx()[0]
        x1, y1, z1 = sp1.range_space()
        xx, yy, zz = np.meshgrid(x1, y1, z1)
        xx = np.reshape(xx, (1, np.size(xx)))[0]
        yy = np.reshape(yy, (1, np.size(yy)))[0]
        zz = np.reshape(zz, (1, np.size(zz)))[0]

        coor = np.vstack((np.vstack((xx, yy)), zz)).transpose()

        # Remove the inner space.
        ind = []
        ind_inside = []
        for ii, row in enumerate(coor):
            if row[(row == -_xmax) | (row == _xmax-dx)].size != 0:
                ind.append(ii)
            else:
                ind_inside.append(ii)

        coor_outside = coor[ind, :]
        coor_outside[:, 0] = coor_outside[:, 0] - self.center[0]
        coor_outside[:, 1] = coor_outside[:, 1] - self.center[1]
        coor_outside[:, 2] = coor_outside[:, 2] - self.center[2]

        # Judge
        ind = np.ones(coor_outside.shape[0], dtype=bool)
        for jj in range(len(self._NN)):
            ind = ind & (coor_outside[:,0]*self._NN[jj,0]
                        +coor_outside[:,1]*self._NN[jj,1]
                        +coor_outside[:,2]*self._NN[jj,2]
                        <= self.DD[jj])
        if sum(ind) != 0:
            raise ValueError("Invalid parameters for constructing a polyhedron.")

""" --- Definition for standalone use with other functions --- """
def check_poly_validity(NN, DD, center=[0., 0., 0.]):
    """check_poly_validity(NN, DD, center=[0., 0., 0.]) -> bool
    check the validity of a given face.
    
    Parameters
    ----------
    NN     : list or numpy.2darray
            normal vectors
    DD     : list or numpy.2darray
        discante from the center of a polyhedron
    center : list or numpy.1darray
        the center of a polyhedron
    
    Returns
    -------
    True / False
    
    Determination condition on whether to construct a polyhedron
    ------------------------------------------------------------
    A cube having a size three times larger than max (DD) is prepared.
    If at least one of the points belonging to that side is inside at least one surface, it is out.
    That is, it is OK if any side is outside of any side.
    """
    # Prepare the outer space.
    _xmax = max(DD)*3.
    _N = 10
    sp1 = space(_N, _xmax, ymax=_xmax, Ny=_N, zmax=_xmax, Nz=_N)
    dx = sp1.dx()[0]
    x1, y1, z1 = sp1.range_space()
    xx, yy, zz = np.meshgrid(x1, y1, z1)
    xx = np.reshape(xx, (1, np.size(xx)))[0]
    yy = np.reshape(yy, (1, np.size(yy)))[0]
    zz = np.reshape(zz, (1, np.size(zz)))[0]

    coor = np.vstack((np.vstack((xx, yy)), zz)).transpose()

    # Remove the inner space.
    ind = []
    ind_inside = []
    for ii, row in enumerate(coor):
        if row[(row == -_xmax) | (row == _xmax-dx)].size != 0:
            ind.append(ii)
        else:
            ind_inside.append(ii)

    coor_outside = coor[ind, :]
    coor_outside[:, 0] = coor_outside[:, 0] - center[0]
    coor_outside[:, 1] = coor_outside[:, 1] - center[1]
    coor_outside[:, 2] = coor_outside[:, 2] - center[2]

    # Judge
    ind = np.ones(coor_outside.shape[0], dtype=bool)
    for jj in range(len(NN)):
        ind = ind & (coor_outside[:,0]*NN[jj,0]
                    +coor_outside[:,1]*NN[jj,1]
                    +coor_outside[:,2]*NN[jj,2]
                    <= DD[jj])
    return sum(ind) == 0
