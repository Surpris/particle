# -*- coding: utf-8 -*-

from .spheroid import spheroid


class sphere(spheroid):
    '''sphere class'''

    def __init__(self, a, **kwargs):
        """__init__(self, a, **kwargs) -> None

        initialize this class.

        Parameters
        ----------
        a      : float
            radius
        kwargs : options
            center : 3-element list or numpy.1darray
                the center of a particle
            density : float
                the density of a particle
        """
        center = kwargs.get('center')
        density = kwargs.get("density")
        spheroid.__init__(self, a, center=center, density=density)
