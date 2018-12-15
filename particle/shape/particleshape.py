# -*- coding: utf-8 -*-

from .spheroid import spheroid
from .sphere import sphere
from .cube import cube
from .cuboctahedron import cuboctahedron
from .icosahedron import icosahedron
from .wulffpolyhedron import wulffpolyhedron
from .polyhedron import polyhedron

def particleshape(**info):
    """particleshape(**info) -> (object)
    generate one object according to `info`.

    Parameters
    ----------
    info : dict
        this parameter must have `shape_name`, `a`, and `kwargs`.

    Returns
    -------
        an instance of some class in `shape` directory
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
