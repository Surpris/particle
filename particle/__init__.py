# -*- coding: utf-8 -*-

__author__ = "Toshiyuki Nishiyama"
__version__ = "1.1"

try:
    __PARTICLE_SETUP__
except NameError:
    __PARTICLE_SETUP__ = False

if __PARTICLE_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from particle source directory.\n')
    del _sys
else:
    from . import core
    from .core import ensemble_system, particle
    from . import shape
