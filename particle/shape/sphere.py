# -*- coding: utf-8 -*-

from .shapeslice import shapeslice
from .spheroid import spheroid

class sphere(spheroid):
    '''
    Sphere.
    '''
    def __init__(self, a, **kwargs):
        center = kwargs.get('center')
        spheroid.__init__(self, a, center=center)
