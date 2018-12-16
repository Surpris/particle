# -*- coding: utf-8 -*-
import numpy as np

class space(object):
    '''space class
    This class deals with meshgrids in 3D space.
    Meshgrids in x- and y- directions are calculated.
    This class has basic functions for the above purpose and supposed to be inherited by one's class.
    '''

    def __init__(self, Nx, xmax, **kwargs):
        """__init__(self, Nx, xmax, **kwargs) -> None
        initialize this class

        Parameters
        ----------
        Nx     : the number of grid points in the x direction
        xmax   : the maximum of space in the x direction
        kwargs : options
            Ny   : int
                the number of grid points in the y direction
            Nz   : int
                the number of grid points in the z direction
            ymax : float
                the maximum of space in the y direction
            zmax : float
                the maximum of space in the z direction
        """
        if type(Nx) not in [int, np.int32, np.int64]:
            raise ValueError("Nx must be integer and be larger than 1.")
        if Nx <= 1:
            raise ValueError("Nx must be larger than 1.")
        if xmax <= 0:
            raise ValueError("xmax must be positive.")

        # The number of points in each direction
        self._Nx = Nx
        self._Ny = Nx if kwargs.get('Ny') is None else kwargs.get('Ny')
        self._Nz = Nx if kwargs.get('Nz') is None else kwargs.get('Nz')

        # The 1st root point of 1st spherical Bessel function.
        self.__x0 = 3.8315

        # Max. of each range
        self._xmax = xmax
        self._ymax = xmax if kwargs.get('ymax') is None else kwargs.get('ymax')
        self._zmax = xmax if kwargs.get('zmax') is None else kwargs.get('zmax')

        """ --- Real space --- """
        # Spatial resolution
        self._dx = 2*self._xmax/self._Nx
        self._dy = 2*self._ymax/self._Ny
        self._dz = 2*self._zmax/self._Nz

        # Range
        self._sprange_x = np.arange(-self._xmax, self._xmax, self._dx)
        self._sprange_y = np.arange(-self._ymax, self._ymax, self._dy)
        self._sprange_z = np.arange(-self._zmax, self._zmax, self._dz)

        """ --- Frequency space --- """
        # Spatial resolution of the Frequency space
        self._dfx = 0.5/self._xmax
        self._dfy = 0.5/self._ymax
        self._dfz = 0.5/self._zmax

        # Frequency
        self._freqx = self._dfx*np.arange(-0.5*self._Nx, 0.5*self._Nx, 1.0)
        self._freqy = self._dfy*np.arange(-0.5*self._Ny, 0.5*self._Ny, 1.0)
        self._freqz = self._dfz*np.arange(-0.5*self._Nz, 0.5*self._Nz, 1.0)

        # Angular frequency
        self._dqx = 2*np.pi*self._dfx
        self._dqy = 2*np.pi*self._dfy
        self._dqz = 2*np.pi*self._dfz

        self._qx = self._freqx*2*np.pi
        self._qy = self._freqy*2*np.pi
        self._qz = self._freqz*2*np.pi

    def N(self):
        """N(self) -> numpy.1darray
        return the number of grid points in each direction
        """
        return np.array([self._Nx, self._Ny, self._Nz])

    def dx(self):
        """dx(self) -> numpy.1darray
        return the spatial resolution in each direction
        """
        return np.array([self._dx, self._dy, self._dz])

    def df(self):
        """df(self) -> numpy.1darray
        return the frequency resolution in each direction
        """
        return np.array([self._dfx, self._dfy, self._dfz])

    def dq(self):
        """dq(self) -> numpy.1darray
        return the anglefrequency resolution (=2\pi*freqneucy)
        """
        return np.array([self._dqx, self._dqy, self._dqz])

    def range_space(self, x_threshold=None, y_threshold=None, z_threshold=None):
        """range_space(self, x_threshold=None, y_threshold=None, z_threshold=None)
            -> numpy.1darray, numpy.1darray, numpy.1darray
        return ranges in each direction

        Parameters
        ----------
        X_threshold : threshold to each range to return (X=x, y, z)
        """
        if x_threshold is None: _sprange_x = self._sprange_x*1
        else:
            _ind = np.where(np.abs(self._sprange_x) <= x_threshold)[0]
            _sprange_x = self._sprange_x[_ind]

        if y_threshold is None: _sprange_y = self._sprange_y*1
        else:
            _ind = np.where(np.abs(self._sprange_y) <= y_threshold)[0]
            _sprange_y = self._sprange_y[_ind]

        if z_threshold is None: _sprange_z = self._sprange_z*1
        else:
            _ind = np.where(np.abs(self._sprange_z) <= z_threshold)[0]
            _sprange_z = self._sprange_z[_ind]

        return _sprange_x, _sprange_y, _sprange_z

    def mesh_space(self, x_threshold=None, y_threshold=None, absolute=False):
        """mesh_space(self, x_threshold=None, y_threshold=None, absolute=False)
            -> numpy.2darray, numpy.2darray
        return meshgrids in x- and y- direction

        Parameters
        ----------
        X_threshold : float
            threshold to each range to return (X=x, y, z)
        absolute    : bool (default : False)
            if True, then return sqrt(x^2 + y^2)
        """
        _sprange_x, _sprange_y, _sprange_z = self.range_space(x_threshold, y_threshold)
        _xx, _yy = np.meshgrid(_sprange_x, _sprange_y)
        if absolute is True: 
            return np.sqrt(_xx**2 + _yy**2)
        else: 
            return _xx, _yy

    def range_freq(self, fx_threshold=None, fy_threshold=None, fz_threshold=None):
        """range_freq(self, fx_threshold=None, fy_threshold=None, fz_threshold=None)
            -> numpy.1darray, numpy.1darray, numpy.1darray
        return frequency ranges in each direction

        Parameters
        ----------
        fX_threshold : threshold to each range to return (X=x, y, z)
        """
        if fx_threshold is None: _fx = self._freqx*1
        else:
            _ind_f = np.where(np.abs(self._freqx) <= fx_threshold)[0]
            _fx = self._freqx[_ind_f]

        if fy_threshold is None: _fy = self._freqy*1
        else:
            _ind_f = np.where(np.abs(self._freqy) <= fy_threshold)[0]
            _fy = self._freqy[_ind_f]

        if fz_threshold is None: _fz = self._freqz*1
        else:
            _ind_f = np.where(np.abs(self._freqz) <= fz_threshold)[0]
            _fz = self._freqz[_ind_f]
        return _fx, _fy, _fz

    def mesh_freq(self, fx_threshold=None, fy_threshold=None, absolute=False):
        """mesh_freq(self, fx_threshold=None, fy_threshold=None, absolute=False)
            -> numpy.2darray, numpy.2darray
        return frequency meshgrids in fx- and fy- direction

        Parameters
        ----------
        fX_threshold : float
            threshold to each range to return (X=x, y, z)
        absolute    : bool (default : False)
            if True, then return sqrt(x^2 + y^2)
        """
        _fx, _fy, _fz = self.range_freq(fx_threshold, fy_threshold)
        _fxx, _fyy = np.meshgrid(_fx, _fy)
        if absolute is True: 
            return np.sqrt(_fxx**2 + _fyy**2)
        else: 
            return _fxx, _fyy

    def range_anglefreq(self, qx_threshold=None, qy_threshold=None, qz_threshold=None):
        """range_anglefreq(self, qx_threshold=None, qy_threshold=None, qz_threshold=None)
            -> numpy.1darray, numpy.1darray, numpy.1darray
        return anglefrequency ranges in each direction

        Parameters
        ----------
        qX_threshold : threshold to each range to return (X=x, y, z)
        """
        if qx_threshold is None: _qx = self._qx*1
        else:
            _ind_q = np.where(np.abs(self._qx) <= qx_threshold)[0]
            _qx = self._qx[_ind_q]

        if qy_threshold is None: _qy = self._qy*1
        else:
            _ind_q = np.where(np.abs(self._qy) <= qy_threshold)[0]
            _qy = self._qy[_ind_q]

        if qz_threshold is None: _qz = self._qz*1
        else:
            _ind_q = np.where(np.abs(self._qz) <= qz_threshold)[0]
            _qz = self._qz[_ind_q]
        return _qx, _qy, _qz

    def mesh_anglefreq(self, qx_threshold=None, qy_threshold=None, absolute=False):
        """mesh_anglefreq(self, qx_threshold=None, qy_threshold=None, absolute=False)
            -> numpy.2darray, numpy.2darray
        return anglefrequency meshgrids in qx- and qy- direction

        Parameters
        ----------
        qX_threshold : float
            threshold to each range to return (X=x, y, z)
        absolute    : bool (default : False)
            if True, then return sqrt(x^2 + y^2)
        """
        _qx, _qy, _qz = self.range_anglefreq(qx_threshold, qy_threshold)
        _qxx, _qyy = np.meshgrid(_qx, _qy)
        if absolute is True: 
            return np.sqrt(_qxx**2 + _qyy**2)
        else: 
            return _qxx, _qyy

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
