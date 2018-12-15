# -*- coding: utf-8 -*-

from .shapeslice import shapeslice

class spheroid(shapeslice):
    '''spheroid class'''

    def __init__(self, ax, **kwargs):
        """__init__(self, ax, **kwargs) -> None
        initialize this class.

        Parameters
        ----------
        ax     : float
            x-direction length
        kwargs : options
            ay      : float
                y-directino length
            az      : float
                z-directino length
            center  : 3-element list or numpy.1darray
                the center of a particle
            density : float
                the density of a particle
        """
        self._shape_name = 'spheroid'
        self.ax = ax
        self.ay = ax if kwargs.get('ay') is None else kwargs.get('ay')
        self.az = ax if kwargs.get('az') is None else kwargs.get('az')
        if self.ax == self. ay and self.ay == self.az:
            self._shape_name = 'sphere'
        self.center = [0., 0., 0.] if kwargs.get('center') is None else kwargs.get('center')

        self.a = max(self.ax, self.ay, self.az)
        self.a_range = self.a*1.1
        shapeslice.__init__(self, self._shape_name, self.ax, self.ay, self.az,
                            center=self.center, density=kwargs.get("density"))
        self._kwargs = kwargs

    def shape_name(self):
        """shape_name(self) -> str
        return the name of this shape
        """
        return self._shape_name + ""

    def info(self):
        """info(self) -> dict
        get information to make one object by `particleshape`.
        """
        if self._shape_name == 'sphere':
            _kwargs = dict(a_range=self.a_range, center=self.center, original_kwargs=self._kwargs)
        else:
            _kwargs = dict(a_range=self.a_range, center=self.center, ay=self.ay, az=self.az, original_kwargs=self._kwargs)
        return dict(shape_name=self._shape_name, a=self.ax, kwargs=_kwargs)
