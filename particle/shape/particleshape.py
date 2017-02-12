# -*- coding: utf-8 -*-

from .cuboctahedron import *
from .spheroid import *
from .sphere import *
from .cube import *
from .cuboctahedron import *
from .icosahedron import *
from .wulffpolyhedron import *
from .polyhedron import *

def particleshape(**info):
    """
        形状に関する情報から形状を返す
        `info`はshape_name, a, kwargsからなる。
    """
    shape_lower = info.get("shape_name")
    a = info.get("a")
    kwargs = info.get("kwargs")
    if kwargs is None:
        kwargs = {}
    if shape_lower in ['sphere']:
        return sphere(a, **kwargs)
    elif shape_lower in ['spheroid']:
        return spheroid(a, **kwargs)
    elif shape_lower in ['cube']:
        return cube(a, **kwargs)
    elif shape_lower in ['cubo', 'cuboctahedron']:
        return cuboctahedron(a, **kwargs)
    elif shape_lower in ['icosa', 'icosahedron']:
        return icosahedron(a, **kwargs)
    elif shape_lower in ['wulff', 'wulffpolyhedron']:
        return wulffpolyhedron(a, **kwargs)
    elif shape_lower == "polyhedron":
        _NN = kwargs.get("NN")
        _DD = kwargs.get("DD")
        if _NN is None:
            raise KeyError("NN")
        elif _DD is None:
            raise KeyError("DD")
        del kwargs["NN"], kwargs["DD"]
        _ = polyhedron(a, _NN, _DD, **kwargs)
        kwargs["NN"] = _NN
        kwargs["DD"] = _DD
        return _
    else:
        raise ValueError("Unknown shape: " + shape_lower)
