# -*- coding: utf-8 -*-

from .spheroid import spheroid

class sphere(spheroid):
    '''
    Sphere.
    '''
    def __init__(self, a, **kwargs):
        center = kwargs.get('center')
        density = kwargs.get("density")
        spheroid.__init__(self, a, center=center, density=density)
