# -*- coding: utf-8 -*-

from .shapeslice import shapeslice
from .spheroid import spheroid

class sphere(spheroid):
    '''
        球を与えるクラス。
        内部変数として半径と中心を持つ。
    '''
    def __init__(self, a, **kwargs):
        center = kwargs.get('center')
        spheroid.__init__(self, a, center=center)
