# -*- coding: utf-8 -*-

import numpy as np

class shapeslice(object):
    '''method class which calculates the section of a single particle'''

    def __init__(self, shape_name, *args, **kwargs):
        """__init__(self, shape_name, *args, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        shape_name : str
        args       : options
        kwargs     : options
            NN : list or numpy.2darray
                normal vectors
            DD : list or numpy.2darray
                discante from the center of a particle
            center : 3-element list or numpy.1darray
                the center of a particle
            density : float
                the density of a particle
        """
        if shape_name in ['sphere', 'spheroid']:
            self.ax = args[0]
            self.ay = args[1]
            self.az = args[2]
            self.Slice = self.__SpheroidSlice
            self.SliceSurface = self.__SpheroidSliceSurface
        else:
            if kwargs.get('NN') is None or kwargs.get('DD') is None:
                raise Exception('Invalid initalization keywords.')
            self.a = args[0]
            self.NN = kwargs.get('NN')
            self.DD = kwargs.get('DD')
            self.perm = [0,1,2] if kwargs.get('perm') is None else kwargs.get('perm')
            self.Slice = self.__PolygonSlice
            self.SliceSurface = self.__PolygonSliceSurface

        self.center = [0., 0., 0.] if kwargs.get('center') is None else kwargs.get('center')
        self.density = 1.0 if kwargs.get("density") is None else kwargs.get("density")

    def __SpheroidSlice(self, xx, yy, z, *args, **kwargs):
        '''__SpheroidSlice(self, xx, yy, z, *args, **kwargs) -> numpy.2darray
        get the positions or indices of the section of the ellipsoid at `z`.

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
        '''
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')

        x2 = (xx-self.center[0])**2/self.ax**2
        y2 = (yy-self.center[1])**2/self.ay**2
        if is_ind is True:
            return x2 + y2 <= 1. - (z-self.center[2])**2/self.az**2
        else:
            ind = x2 + y2 <= 1. - (z-self.center[2])**2/self.az**2
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

    def __SpheroidSliceSurface(self, xx, yy, z, width, **kwargs):
        """__SpheroidSliceSurface(self, xx, yy, z, width, **kwargs) -> numpy.2darray
        Get the positions or indices of the profile of the cross section of the ellipsoid at position z.

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
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')
        x2 = (xx-self.center[0])**2/self.ax**2
        y2 = (yy-self.center[1])**2/self.ay**2
        ind_slice = self.Slice(xx, yy, z)
        ind_slice_width = x2 + y2 <= (1. - width/self.ax)**2 - z**2/self.az**2
        if is_ind is True:
            return ind_slice^ind_slice_width
        else:
            ind = ind_slice^ind_slice_width
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

    def __PolygonSlice(self, xx, yy, z, *args, **kwargs):
        '''__PolygonSlice(self, xx, yy, z, *args, **kwargs) -> numpy.2darray
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
        '''
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')

        ind = np.ones((xx.shape[0], xx.shape[1]), dtype=bool)
        for jj in range(len(self.NN)):
            ind = ind & ((xx-self.center[0])*self.NN[jj,self.perm[0]]
                         +(yy-self.center[1])*self.NN[jj,self.perm[1]]
                         <= self.a*self.DD[jj]
                         -(z-self.center[2])*self.NN[jj,self.perm[2]])
        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

    def __PolygonSliceSurface(self, xx, yy, z, width, **kwargs):
        """__PolygonSliceSurface(self, xx, yy, z, width, **kwargs) -> numpy.2darray
        Get the positions or indices of the profile of the cross section of polyhedron at position z.

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
            the coordinates composing the section
        """
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')
        ind_body = self.Slice(xx, yy, z)
        ind = np.ones((xx.shape[0], xx.shape[1]), dtype=bool)
        for jj in range(0, len(self.NN)):
            ind = ind & ((xx-self.center[0])*self.NN[jj,self.perm[0]]
                         +(yy-self.center[1])*self.NN[jj,self.perm[1]]
                         <= self.a*self.DD[jj]-width
                         -(z-self.center[2])*self.NN[jj,self.perm[2]])
        ind = ind_body^ind
        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

    def EulerRot(self, *args, **kwargs):
        """EulerRot(self, *args, **kwargs)
        Euler rotation.
        This function is an abstract function.
        """
        pass

    def UpdSlice(self, *args, **kwargs):
        """EulerRot(self, *args, **kwargs)
        Update self.
        This function is an abstract function.
        """
        pass

    def shape_name(self, *args, **kwargs):
        """shape_name(self, *args, **kwargs)
        name of shape.
        This function is an abstract function.
        """
        pass
