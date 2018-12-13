# -*- coding: utf-8 -*-

# System modules
import os
import pickle
import datetime
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import sys
# sys.path.append("./")

# User modules
# from . import core
from . import mathfunctions as mf
from . import folderfunctions as ff
# from . import slicefft
from .slicefft import slicefft
# from . import shape
from ..shape import *

class particle(slicefft):
    '''
    粒子形状を総括的に扱うクラス。
    slicefftクラスを継承することでMulti-slice FTが実行できるようにしている。
    '''
    def __init__(self, *args, **kwargs):
        """
            クラスの初期化。
        """
        if len(args) <= 1:
            if len(args) == 1:
                if type(args[0]) == str:
                    with open(args[0], "rb") as f:
                        kwargs = pickle.load(f)
                elif type(args[0]) == dict:
                    kwargs = args[0]
            N = kwargs.get('Nx'); xmax = kwargs.get('xmax')
            shape = kwargs.get('shape'); R = kwargs.get('R')
            del kwargs['Nx'], kwargs['xmax'], kwargs['shape'], kwargs['R']
            self.__kwargs = kwargs
        elif len(args) < 4: raise Exception('Failure.')
        else:
            N = args[0]; xmax = args[1]; shape = args[2]; R = args[3]
            self.__kwargs = kwargs

        self.__initiator = dict(Nx=N, xmax=xmax, shape=shape, R=R)
        self.__InitFileInfo(kwargs.get('savefldr'), kwargs.get('savecount'))
        slicefft.__init__(self, N, xmax, **kwargs)
        coorargs = dict(coor=kwargs.get('coor'), coor_surf=kwargs.get('coor_surf'),
                        euler=kwargs.get('euler'))
        self.__InitCoorInfo(**coorargs)
        self.__SetParticle(shape, R, **kwargs)

        if self.__hailstone is True:
            self.__SetHailstone(*args, **kwargs)
        self.SetParticle(self._shape)
        self._shape_name = self._shape.shape_name()
        self.EulerRot()

    def __SetParticle(self, shape, *args, **kwargs):
        self.__hailstone = False if kwargs.get('hailstone') is None else kwargs.get('hailstone')
        if type(shape) is not str:
            self._shape = shape
            return

        self._shape_name = shape.lower()
        self.__MakeParticle(*args, **kwargs)

    def __MakeParticle(self, *args, **kwargs):
        shape_lower = self._shape_name
        if shape_lower in ['sphere']:
            self._shape = sphere(*args, **kwargs)
        elif shape_lower in ['spheroid']:
            self._shape = spheroid(*args, **kwargs)
        elif shape_lower in ['cube']:
            self._shape = cube(*args, **kwargs)
        elif shape_lower in ['cubo', 'cuboctahedron']:
            self._shape = cuboctahedron(*args, **kwargs)
        elif shape_lower in ['icosa', 'icosahedron']:
            self._shape = icosahedron(*args, **kwargs)
        elif shape_lower in ['wulff', 'wulffpolyhedron']:
            self._shape = wulffpolyhedron(*args, **kwargs)
        elif shape_lower == "polyhedron":
            _a = args[0]
            _NN = kwargs.get("NN")
            _DD = kwargs.get("DD")
            del kwargs["NN"], kwargs["DD"]
            self._shape = polyhedron(_a, _NN, _DD, *args, **kwargs)
            kwargs["NN"] = _NN
            kwargs["DD"] = _DD
        elif shape_lower == "hailstone":
            self.__hailstone = False # "hailstone" クラスとのバッティングを避けるための一時的対処
            _shape_mother = kwargs.get("shape_mother")
            _shape_daughters = kwargs.get("shape_daughters")
            if _shape_mother is not None and _shape_daughters is not None:
                del kwargs['shape_mother'], kwargs['shape_daughters']
            else:
                _hailinfo = kwargs.get("hailinfo")
                if _hailinfo is None:
                    raise ValueError()
                _mother_kwargs = _hailinfo.get("mother_kwargs")
                _daughter_kwargs = _hailinfo.get("daughter_kwargs")
                _shape_mother = particleshape(**_mother_kwargs)
                _shape_daughters = []
                for _info in _daughter_kwargs:
                    _shape_daughters.append(particleshape(**_info))
            self._shape = hailstone(_shape_mother, _shape_daughters, *args, **kwargs)
            # kwargs['shape_mother'] = _shape_mother
            # kwargs['shape_daughters'] = _shape_daughters
        else:
            raise ValueError("Invalid information on crystal shape.")

    def __SetHailstone(self, *args, **kwargs):
        if self._shape_name not in ['sphere', 'spheroid']:
            self.Coor('surf')
            kwargs['coor'] = self.__coor_surf + 0.0
            self.__coor_surf = None
        self._shape = hailstone_with_sphere(self._shape, *args, **kwargs)

    def __InitFileInfo(self, savefldr=None, count=None):
        self.__d = datetime.datetime.today()
        if savefldr is None or type(savefldr) is not str:
            self.__savefldrpath = 'G:/UserData/Python_output/out_{0:%Y%m%d}/'.format(self.__d)
        else:
            self.__savefldrpath = savefldr
        try:
            ff.makefolders(self.__savefldrpath)
        except Exception as e:
            pass
            # print(e)
        self.__filename = 'particle.particle'
        self.__filepath = self.__savefldrpath + self.__filename
        if count is None or type(count) is not int:
            self.__savecount = 0
        else:
            self.__savecount = count

    def __InitCoorInfo(self, **kwargs):
        self.__coor_types = ['body', 'surf']
        self.__coor = None if kwargs.get('coor') is None else kwargs.get('coor')
        self.__coor_surf = None if kwargs.get('coor_surf') is None else kwargs.get('coor_surf')
        self.__euler = [0,0,0] if kwargs.get('euler') is None else kwargs.get('euler')

    def save(self, filepath=None, nameonly=False, overwrite=False):
        self.__kwargs.update(self.__initiator)
        self.__kwargs['shape'] = self._shape_name
        self.__kwargs['saved'] = True
        self.__kwargs['coor'] = self.__coor
        self.__kwargs['coor_surf'] = self.__coor_surf
        self.__kwargs.update(self.fftInfo())
        if self.__hailstone is True:
            self.__kwargs.update(self._shape.hailsInfo())
        if self._shape_name == "hailstone":
            try:
                self.__kwargs["hailinfo"] = self._shape.info()
                print("ok")
            except Exception as e:
                print(e)

        if self.__kwargs.get("shape_mother") is not None:
            del self.__kwargs["shape_mother"]
        if self.__kwargs.get("shape_daughters") is not None:
            del self.__kwargs["shape_daughters"]

        if filepath is None:
            _filepath = self.__filepath
        elif nameonly is False:
            ind = filepath.index('/')
            _fldrpath = filepath[:ind + 1]
            ff.makefolders(_fldrpath)
            _filepath = filepath
        else:
            _filepath = self.__savefldrpath + filepath

        buff_filepath = _filepath + ""
        while(os.path.exists(buff_filepath)):
            _ext = _filepath.split('.')[-1]
            self.__savecount += 1
            buff_filepath = _filepath.replace('.'+_ext, '_save_{0:04d}.{1}'.format(self.__savecount, _ext))
        _filepath = buff_filepath
        print('Save to the new path: {0}'.format(_filepath))

        self.__kwargs['savecount'] = self.__savecount
        self.__kwargs['filepath'] = _filepath
        with open(_filepath, "wb") as f:
            pickle.dump(self.__kwargs, f)

    def copy(self):
        """
            同じ粒子を返す。
        """
        self.__kwargs.update(self.__initiator)
        self.__kwargs['shape'] = self._shape_name
        self.__kwargs['saved'] = True
        self.__kwargs['coor'] = self.__coor
        self.__kwargs['coor_surf'] = self.__coor_surf
        self.__kwargs.update(self.fftInfo())
        if self.__hailstone is True:
            self.__kwargs.update(self._shape.hailsInfo())
        return particle(**self.__kwargs)

    def a(self):
        return self._shape.a + 0.0

    def a_range(self):
        return self._shape.a_range + 0.0

    def shape_name(self):
        return self._shape_name + ''

    def n_hail(self):
        if self.__kwargs.get('n_hail') is not None:
            return self.__kwargs.get('n_hail')
        else:
            return 0

    def a_hail_max(self):
        if self.__kwargs.get('a_hail_max') is not None:
            return self.__kwargs.get('a_hail_max')
        else:
            return 0

    def center_hails(self, angle=False):
        if self.__hailstone is True:
            return self._shape.GetCenterHails(angle)
        elif self._shape_name == "hailstone":
            return self._shape.center_daughters.copy()
        else:
            raise AttributeError('No attribute: "hailstone."')

    def center(self):
        return self._shape.center

    def EulerRot(self, euler=None):
        if euler is not None:
            self.__euler = euler
            self.__kwargs['euler'] = euler
        self._shape.EulerRot(self.__euler)
        self.SetParticle(self._shape)

    def Slice(self, xx, yy, z, *args, **kwargs):
        return self._shape.Slice(xx, yy, z, *args, **kwargs)

    def Coor(self, coor_type='body', *args, **kwargs):
        if self._shape is None:
            raise ValueError("No information on the shape.")
        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        if coor_type is self.__coor_types[0]:
            _slice = self._shape.Slice
        elif coor_type is self.__coor_types[1]:
            try:
                _slice = self._shape.SliceSurface
            except:
                print('Failure in setting the calculation method for surface coordinates.')
                print('The method for body coordinates will be used.')
                _slice = self._shape.Slice

        self.__is_trunc = False if kwargs.get('is_trunc') is None else kwargs.get('is_trunc')

        _sprange_x, _sprange_y, _sprange_z = self.range_space(self._shape.a_range,self._shape.a_range,self._shape.a_range)
        _xx, _yy = np.meshgrid(_sprange_x, _sprange_y)
        dx = self.dx()[0]
        buff = np.zeros((1,3), dtype=float)
        for zz in _sprange_z:
            coor1 = _slice(_xx, _yy, zz, dx, is_ind=False, is_trunc=self.__is_trunc)
            buff = np.concatenate((buff, coor1))

        if coor_type is self.__coor_types[0]:
            if buff.shape[0] is 1:
                self.__coor = None
            else:
                self.__coor = buff[1:buff.shape[0],:]
        elif coor_type is self.__coor_types[1]:
            if buff.shape[0] is 1:
                self.__coor_surf = None
            else:
                self.__coor_surf = buff[1:buff.shape[0],:]

    def GetCoor(self, coor_type='body', **kwargs):
        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        if coor_type is self.__coor_types[0]:
            if self.__coor is None:
                self.Coor(coor_type, **kwargs)
            coor = self.__coor
        elif coor_type is self.__coor_types[1]:
            if self.__coor_surf is None:
                self.Coor(coor_type, **kwargs)
            coor = self.__coor_surf
        if coor is None:
            raise ValueError("No information on the coordinates in the surface.")
        return coor

    def PlotCoor(self, coor_type='body', color='#00fa9a', alpha=0.5):
        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        coor = self.GetCoor(coor_type)
        fig = plt.figure(50, figsize=(6,5), dpi=100)
        plt.clf()
        ax = Axes3D(fig)
        ax.scatter(coor[:,0], coor[:,1], coor[:,2], c=color, alpha=alpha)
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.set_zlabel('z [nm]')
        return fig, ax
