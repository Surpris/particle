# -*- coding: utf-8 -*-

# System modules
import os
import copy
import pickle
import datetime
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# User modules
from . import mathfunctions as mf
from .slicefft import slicefft
from ..shape import *

class ensemble_system(slicefft):
    __doc__ = ''' Class which deal with an ensemble of particles.
    Both `kwargs` for `slicefft` and `ensemble` are necessary for initialization.

    kwargs for slicefft
    -------------------
    kw_slicefft : dict object including the following arguments.
        N      : int
            # of points in x-axis direction
        xmax   : float
            spatial size in x-axis direction
        kwargs : dict
            option (dict object). See `space` and `slicefft` classes.

    kwargs for ensemble
    -------------------
    kw_shapes   : dict object or list of dict objests including the following arguments.
        shape_name : str
            name of shape
        a          : float
            characteristic length of the shape
        kwargs     : dict
            option (dict object). See the classes in `shape` directory.
    
    Other parameters
    ----------------
    kwargs : option
        weighting_density : bool
            if True, then each index of space where particles exist
            is weighned and overwritten by following the order of particles in kw_shapes

    Example of using ensemble_system
    --------------------------------
    import numpy as np
    import sys
    from particle import ensemble_system
    import time

    ### Setting of MSFT
    N = 1024; xmax =2048.0; # Real-space size
    delta = 1e-4; beta=1e-6; refr = 1 - delta - 1j*beta; # Refractivity
    params = dict(Ny=N, zmax=256., Nz=256, ymax=xmax, kmax=0.1212*np.pi, refr=refr)
    kw_slicefft = dict(N=N, xmax=xmax, kwargs=params)

    ### Setting of target
    Rs = np.array([100., 30.])
    coors = np.array([[0., 0., 0.], [100., 0., 0.]])

    # kwargs
    infos = [None] * len(Rs)
    for ii, _a , _center in zip(range(len(Rs)), Rs, coors):
        infos[ii] = dict(shape_name="sphere", a=_a, kwargs=dict(center=_center))

    ### ensemble_system
    st = time.time()
    ens1 = ensemble_system(kw_slicefft, infos)
    # Execute MSFT
    ens1.MSFT(qmode=False, atte=False)
    # Plot images
    ens1.PlotRhoF(**dict(qcscale=1/25.))
    # plt.savefig("../images/image.png", bbox_inches="tight", pad_inches=0.1)
    print("Elapsed time: {0:.2f} sec.".format(time.time()-st))
    # Save the model
    ens1.save("../data/data.ens")

    #--- End of example ---
    '''

    def doc():
        """ doc() -> str
        print the documentation of this class
        """
        for line in ensemble_system.__doc__.split("\n"):
            print(line)

    def __init__(self, *args, **kwargs):
        """__init__(self, *args, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        args   : options
        kwargs : options

        Examples of the way to initialize this class
        --------------------------------------------
        """
        if len(args) <= 1: # if 0 then kwargs is used
            if len(args) == 1: # file path or a dict object which self.save() outputs
                if type(args[0]) == str:
                    with open(args[0], "rb") as ff:
                        kwargs_ = pickle.load(ff)
                elif type(args[0]) == dict:
                    kwargs_ = args[0]
        elif len(args) == 2:
            if type(args[0]) == str:
                with open(args[0], "rb") as f:
                    kwargs_ = pickle.load(f)
            elif type(args[0]) == dict:
                kwargs_ = dict(kw_slicefft=args[0], kw_shapes=args[1])
        else: # if len(args) > 3 then raise Exception
            raise Exception('Failure.')
        
        weighting_density = kwargs.get("weighting_density")

        # Separate parameters
        kw_slicefft = kwargs_.get("kw_slicefft")
        if kw_slicefft is None or type(kw_slicefft) != dict:
            raise ValueError("Invalid value for `kw_slicefft`.")

        kw_shapes = kwargs_.get("kw_shapes")
        if kw_shapes is None:
            raise KeyError("kw_shapes")

        # Initialize slicefft
        N = kw_slicefft.get('N')
        xmax = kw_slicefft.get('xmax')
        options_slicefft = kw_slicefft.get('kwargs')
        slicefft.__init__(self, N, xmax, **options_slicefft)

        # Initialize attributes for calculation of coordinates
        self.__InitCoorInfo()

        # Generate an ensemble according to the input arguments
        self._shape = ensemble(kw_shapes, weighting=weighting_density)
        self._kwargs = copy.deepcopy(kwargs)

    def __InitCoorInfo(self):
        """__InitCoorInfo(self) -> None
        initialize attributes for coordinates
        """
        self.__coor_types = ['body', 'surf']
        self.__coor = None
        self.__coor_surf = None
        self.__euler = [0,0,0]

    def save(self, fpath=None):
        """save(self, fpath=None) -> None
        save this object as a dict object

        Parameters
        ----------
        fpath : float (optional, dafault : './ensemble.pickle')
            path of the file to save this class into.
        """
        if fpath is None:
            fpath = "./ensemble.pickle"
            print("`fpath` is not assigned. Save this object to {}.".format(fpath))
        _kw_slicefft = self._kwargs.get("kw_slicefft")
        _kw_slicefft.update(dict(kwargs=self.fftInfo()))
        _kw = dict(kw_slicefft=_kw_slicefft, kw_shapes=self._shape.info())
        with open(fpath, "wb") as f:
            pickle.dump(_kw, f)
    
    def shapeinfo(self):
        """shapeinfo(self) -> list
        return a list of information on the shapes in this class
        """
        return self._shape.info()[:]

    def a(self):
        """a(self) -> float
        return the characteristic length of this instance
        """
        return self._shape.a + 0.0

    def a_range(self):
        """a_range(self) -> float
        return the characteristic scale of this instance
        """
        return self._shape.a_range + 0.0

    def a_max(self):
        """a_max(self) -> float
        return the maximum of the characteristic lengths of the shapes in this instance
        """
        return self._shape._as[1:].max()

    def shape_name(self):
        """shape_name(self) -> str
        return the name of shape
        """
        return self._shape.shape_name()

    def n_shapes(self):
        """n_shapes(self) -> int
        return the number of shapes in this instance
        """
        return self._shape._n_shapes*1

    def Coor(self, coor_type='body', *args, **kwargs):
        """Coor(self, coor_type='body', *args, **kwargs) -> None
        calculate the coordinates characterizing the ensemble.

        Parameters
        ----------
        coor_type : str (default : 'body')
            type of the coordinates.
            'body' = the whole coordinates
            'surf' = the coordinates of the surface
        args      : options
        kwargs    : options
        """
        if self._shape is None:
            raise ValueError("No information on the shape.")
        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        if coor_type is self.__coor_types[0]:
            _slice = self._shape.Slice
        elif coor_type is self.__coor_types[1]:
            try:
                _slice = self._shape.SliceSurface
            except:
                print('Failure in setting the calculation method for surface coordinates.')
                print('The method for body coordinates will be used.')
                _slice = self._shape.Slice

        self.__is_trunc = False if kwargs.get('is_trunc') is None else kwargs.get('is_trunc')

        _sprange_x, _sprange_y, _sprange_z = \
            self.range_space(self._shape.a_range,self._shape.a_range,self._shape.a_range)
        _xx, _yy = np.meshgrid(_sprange_x, _sprange_y)
        dx = self.dx()[0]
        buff = np.zeros((1,3), dtype=float)
        for zz in _sprange_z:
            coor1 = _slice(_xx, _yy, zz, dx, is_ind=False, is_trunc=self.__is_trunc)
            buff = np.concatenate((buff, coor1))

        if coor_type is self.__coor_types[0]:
            if buff.shape[0] is 1:
                self.__coor = None
            else:
                self.__coor = buff[1:buff.shape[0],:]
        elif coor_type is self.__coor_types[1]:
            if buff.shape[0] is 1:
                self.__coor_surf = None
            else:
                self.__coor_surf = buff[1:buff.shape[0],:]

    def GetCoor(self, coor_type='body', **kwargs):
        """GetCoor(self, coor_type='body', **kwargs) -> numpy.2darray
        return the coordinates characterizing the ensemble.
        If the coordinates has been already calculated, then return them.

        Parameters
        ----------
        coor_type : str (default : 'body')
            type of the coordinates.
            'body' = the whole coordinates
            'surf' = the coordinates of the surface
        args      : options
        kwargs    : options

        Returns
        -------
        coor : numpy.2darray
            the coordinates
        """
        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        if coor_type is self.__coor_types[0]:
            if self.__coor is None:
                self.Coor(coor_type, **kwargs)
            coor = self.__coor
        elif coor_type is self.__coor_types[1]:
            if self.__coor_surf is None:
                self.Coor(coor_type, **kwargs)
            coor = self.__coor_surf
        if coor is None:
            raise ValueError("No information on the coordinates in the surface.")
        return coor

    def PlotCoor(self, mode="solid", coor_type='body', color='#00fa9a', alpha=0.5):
        """PlotCoor(self, mode="solid", coor_type='body', color='#00fa9a', alpha=0.5) -> pyplot.figure, pyplot.axes
        plot the coordinates characterizing the ensemble

        Parameters
        ----------
        mode      : str
            mode of plotting.
            'solid' = plot all the coordinates
            'surface' = plot the surface of shapes (this is valid for the spheroidal shapes)
        coor_type : str
            type of the coordinates.
            'body' = the whole coordinates
            'surf' = the coordinates of the surface
        color     : str
            color code
        alpha     : float
            alpha parameter of color
        
        Returns
        -------
        fig : pyplot.figure
        ax  : pyplot.axes
        """
        if mode == "surface":
            fig = plt.figure(50, figsize=(6,5), dpi=100)
            plt.clf()
            ax = Axes3D(fig)
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            _infos = self._shape.info()
            for info in _infos:
                _center = info.get("kwargs").get("center")
                _a = info.get("a")
                ax.plot_surface(_a*x + _center[0], _a*y + _center[1], _a*z + _center[2], rstride=1, cstride=1, color='b', linewidth=0, cmap=cm.ocean)

        else:
            if coor_type not in self.__coor_types:
                raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
            coor = self.GetCoor(coor_type)
            fig = plt.figure(50, figsize=(6,5), dpi=100)
            plt.clf()
            ax = Axes3D(fig)
            ax.scatter(coor[:,0], coor[:,1], coor[:,2], c=color, alpha=alpha)
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.set_zlabel('z [nm]')
        return fig, ax

class ensemble(object):
    """ensemble class
    This class represents an ensemble of particles.
    This class is supposed to be used by `ensemble_system`.
    """

    _shape_name = "ensemble"

    def __init__(self, shapes, weighting=None, *args, **kwargs):
        ''' __init__(self, shapes, weighting=None, *args, **kwargs) -> None
        initialize this class

        Parameters
        ----------
        shapes    : kwargs or list of kwargs
            information on particles
        weighting : bool
            True = weighting with particles' densities
        args      : option
        kwargs    : option
        '''
        # Keep kwargs of each particle
        if type(shapes) != list:
            self._shapes_kwarg = [shapes]
        else:
            self._shapes_kwarg = shapes
        
        # Set weighting mode
        if weighting is None:
            self._weighting = False
        elif not isinstance(weighting, bool):
            raise TypeError("'weighting' must be boolean.")
        else:
            self._weighting = weighting

        # Calculate the number of particles
        self._n_shapes = len(self._shapes_kwarg)

        ### Keep information and generate particles
        self._shapes = [None] * self._n_shapes
        self._shapes_name = [None] * self._n_shapes
        self._centers = np.zeros((self._n_shapes, 3), dtype=float)
        self._centers_d = np.zeros(self._n_shapes, dtype=float)
        self._as = np.zeros(self._n_shapes, dtype=float)
        self._as_range = np.zeros(self._n_shapes, dtype=float)
        for ii, _ in enumerate(self._shapes_kwarg):
            self._shapes[ii] = particleshape(**_)
            self._shapes_name[ii] = self._shapes[ii].shape_name()
            self._centers[ii] = self._shapes[ii].center
            self._centers_d[ii] = np.linalg.norm(self._centers[ii])
            self._as[ii] = self._shapes[ii].a
            self._as_range[ii] = self._shapes[ii].a_range

        # Calculate extension of ensemble
        self.a = self._as[0]
        for _a, _d in zip(self._as, self._centers_d):
            self.a = max([self.a, _d + _a])
        self.a_range = self._as_range[0]
        for _a_range, _d in zip(self._as_range, self._centers_d):
            self.a_range = max([self.a_range, _d + _a_range])

        # THe center of ensemble is always the center of space
        self.center = [0., 0., 0.]

    def info(self):
        """info(self) -> list of dict
        return the list of information on the shapes in this instence
        """
        kw_list = [None] * self._n_shapes
        for ii, _ in enumerate(self._shapes):
            kw_list[ii] = _.info()
        return kw_list

    def shape_name(self):
        """shape_name(self) -> str
        return the name of shape (`ensemble` in case of this class)
        """
        return self._shape_name

    def shapes_name(self):
        """shapes_name(self) -> list
        return list of names of shapes
        """
        return self._shapes_name[:]

    def centers(self, polar=False):
        """centers(self, polar=False) -> numpy.2darray
        return the center coordinates of each particle.

        Parameters
        ----------
        polar : bool
            if True then the polar coordinates are returned.
        
        Returns
        -------
        centers (polar=False) : numpy.2darray
        r, theta, phi (polar=True) : numpy.1darray
        """
        centers = self._centers.copy()
        if polar is False:
            return centers
        else:
            r = np.linalg.norm(centers, axis=1)
            theta = np.arctan2(np.sqrt(centers[:,0]**2+centers[:,1]**2), centers[:,2])
            phi = np.arctan2(centers[:,1], centers[:,0]) + np.pi
            return r, theta, phi

    def EulerRot(self, euler):
        """EulerRot(self, euler)
        TODO: implementation
        """
        # self.__shape_mother.EulerRot(euler)
        # self.center_daughters = mf.EulerRotation(self.center_daughters, euler, 1).copy()
        pass

    def Slice(self, xx, yy, z, *args, **kwargs):
        """Slice(self, xx, yy, z, *args, **kwargs) -> numpy.2darray
        get the positions or indices of the section of polyhedron at `z`.

        Parameters
        ----------
        xx     : numpy.2darray
        yy     : numpy.2darray
        z      : float
        args   : options
        kwargs : options
            is_ind : bool
                if True, then return the indices of the section.
                if False, then return the coordinates composing the section.
        
        Returns
        -------
        ind (is_ind=True)  : numpy.2darray
            the indices of the section
        out (is_ind=False) : numpy.2darray
            the coordinates composing the section
        """
        # if `is_ind` == True then return the indices
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')

        # Calculate the slices of each particle
        ind_hit = np.where(abs(z - self._centers[:,2]) <= self._as_range)[0]
        if len(ind_hit) <= 0:
            ind = np.zeros(xx.shape, dtype=bool)
        else:
            ind = None
            for jj in ind_hit:
                if ind is None:
                    ind = self._shapes[jj].Slice(xx, yy, z)
                    if is_ind is True and self._weighting is True:
                        density_ = 1.0 if self._shapes[jj].density is None else self._shapes[jj].density
                        ind = ind * density_
                else:
                    ind2 = self._shapes[jj].Slice(xx, yy, z)
                    if is_ind is True and self._weighting is True:
                        density_ = 1.0 if self._shapes[jj].density is None else self._shapes[jj].density
                        ind[ind2] = density_
                    else:
                        ind |= ind2
                # print(ind.sum())

        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out
        pass

    def SliceSurface(self, xx, yy, z, width, *args, **kwargs):
        """SliceSurface(self, xx, yy, z, width, *args, **kwargs) -> numpy.2darray
        return the surface cross-section.
        Currently implementation is not considered.
        
        Parameters
        ----------
        xx     : numpy.2darray
        yy     : numpy.2darray
        z      : float
        width  : float
        args   : options
        kwargs : options
            is_ind : bool
                if True, then return the indices of the section.
                if False, then return the coordinates composing the section.
        
        Returns
        -------
        ind (is_ind=True)  : numpy.2darray
            the indices of the section
        out (is_ind=False) : numpy.2darray
            the coordinates composing the profile of section
        """
        return self.Slice(xx, yy, z, *args, **kwargs)