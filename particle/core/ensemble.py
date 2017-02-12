# -*- coding: utf-8 -*-

# System modules
import os
import copy
import pickle
import datetime
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

class ensemble_system(slicefft):
    __doc__ = '''
        粒子の集団とslicefftをつなぐ扱うクラス。
        初期化にはslicefftに関するキーワードと粒子の集団`ensemble`に関するキーワードが必要。
        < slicefft関連 >
            kw_slicefft : slicefftに関するキーワードを含むdictオブジェクト。
                N           : x軸方向のサイズの分割数
                xmax        : x軸方向のサイズ
                kwargs      : オプション。例えばy軸方向の分割数`ymax`や、
                              計算対象とする散乱ベクトルの大きさの最大値`kmax`
        < ensemble関連 >
            kw_shapes   : 各粒子の情報（dictオブジェクト）またはそのリスト
                shape_name  : 形状の名称
                a           : 形状の特徴的長さ
                kwargs      : オプション。例えば中心`center`やEuler角`euler`
    '''

    def doc():
        for line in ensemble_system.__doc__.split("\n"):
            print(line)

    def __init__(self, *args, **kwargs):
        """
            クラスの初期化。
            slicefftに関するキーワードと粒子集団に関するキーワードを別々に扱う。
            前者はdictとして、後者はdictのリストとして扱う。
            すでに計算結果がある場合、それをslicefftに関するキーワードに含める。
            このクラスは`particle`や`Particle`と互換性がない。
            キーワードは、それぞれに必要なものとオプション（dictオブジェクト）のdictとして与えることが前提。
            どの場合も`args`のほうが優先される。
        """
        if len(args) <= 1: # 0の場合はkwargsをそのまま利用
            if len(args) == 1: # ファイルのパスか、またはself.saveで出力されたdictファイルの場合
                if type(args[0]) == str:
                    with open(args[0], "rb") as f:
                        kwargs = pickle.load(f)
                elif type(args[0]) == dict:
                    kwargs = args[0]
        elif len(args) == 2: # argsに二つ入っている場合
            if type(args[0]) == str:
                with open(args[0], "rb") as f:
                    kwargs = pickle.load(f)
            elif type(args[0]) == dict:
                kwargs = dict(kw_slicefft=args[0], kw_shapes=args[1])
        else: # argsに3つ以上ある場合は受け付けない
            raise Exception('Failure.')

        # パラメータの分離
        kw_slicefft = kwargs.get("kw_slicefft")
        if kw_slicefft is None or type(kw_slicefft) != dict:
            raise ValueError("Invalid value for `kw_slicefft`.")

        kw_shapes = kwargs.get("kw_shapes")
        if kw_shapes is None:
            raise KeyError("kw_shapes")

        # slicefftの初期化
        N = kw_slicefft.get('N'); xmax = kw_slicefft.get('xmax')
        options_slicefft = kw_slicefft.get('kwargs')
        slicefft.__init__(self, N, xmax, **options_slicefft)

        # 座標計算用メンバーの初期化
        self.__InitCoorInfo()

        # 粒子集団の生成
        self._shape = ensemble(kw_shapes)
        self._kwargs = copy.deepcopy(kwargs)

    def __InitCoorInfo(self):
        self.__coor_types = ['body', 'surf']
        self.__coor = None
        self.__coor_surf = None
        self.__euler = [0,0,0]

    def save(self, fpath=None):
        """
            ensembbleクラスを生成するための情報を保存する。
        """
        if fpath is None:
            fpath = "./ensemble.pickle"
        _kw_slicefft = self._kwargs.get("kw_slicefft")
        _kw_slicefft.update(dict(kwargs=self.fftInfo()))
        _kw = dict(kw_slicefft=_kw_slicefft, kw_shapes=self._shape.info())
        with open(fpath, "wb") as f:
            pickle.dump(_kw, f)

    def shapeinfo(self):
        return self._shape.info()[:]

    def a(self):
        return self._shape.a + 0.0

    def a_range(self):
        return self._shape.a_range + 0.0

    def a_max(self):
        return self._shape._as[1:].max()

    def shape_name(self):
        return self._shape.shape_name()

    def n_shapes(self):
        return self._shape._n_shapes*1

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

    def PlotCoor(self, mode="solid", coor_type='body', color='#00fa9a', alpha=0.5):
        if mode == "surface":
            fig = plt.figure(50, figsize=(6,5), dpi=100)
            plt.clf()
            ax = Axes3D(fig)
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            _infos = self._shape.info()
            for info in _infos:
                _center = info.get("kwargs").get("center")
                _a = info.get("a")
                ax.plot_surface(_a*x + _center[0], _a*y + _center[1], _a*z + _center[2], rstride=1, cstride=1, color='b', linewidth=0, cmap=cm.ocean)

        else:
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

class ensemble(object):
    """
        粒子の集団を表現するクラス。
        `ensemble_system`での利用を想定している。
    """

    _shape_name = "ensemble"
    def __init__(self, shapes, *args, **kwargs):
        '''
            クラスの初期化。
            < Input parameters >
                shapes           : 粒子に与えるキーワードまたはそのリスト
                *args            : オプション
                **kwargs         : オプション
        '''
        # 各形状のためのキーワードの保持
        if type(shapes) != list:
            self._shapes_kwarg = [shapes]
        else:
            self._shapes_kwarg = shapes

        # 形状の数を計算
        self._n_shapes = len(self._shapes_kwarg)

        """ 形状の生成と情報の保持 """
        self._shapes = [None] * self._n_shapes
        self._shapes_name = [None] * self._n_shapes
        self._centers = np.zeros((self._n_shapes, 3), dtype=float)
        self._centers_d = np.zeros(self._n_shapes, dtype=float)
        self._as = np.zeros(self._n_shapes, dtype=float)
        self._as_range = np.zeros(self._n_shapes, dtype=float)
        for ii, _ in enumerate(self._shapes_kwarg):
            self._shapes[ii] = particleshape(**_)
            self._shapes_name[ii] = self._shapes[ii].shape_name()
            self._centers[ii] = self._shapes[ii].center
            self._centers_d[ii] = np.linalg.norm(self._centers[ii])
            self._as[ii] = self._shapes[ii].a
            self._as_range[ii] = self._shapes[ii].a_range

        # 粒子集団の空間的広がりを計算
        self.a = self._as[0]
        for _a, _d in zip(self._as, self._centers_d):
            self.a = max([self.a, _d + _a])
        self.a_range = self._as_range[0]
        for _a_range, _d in zip(self._as_range, self._centers_d):
            self.a_range = max([self.a_range, _d + _a_range])

        # 空間のセンターは常に原点
        self.center = [0., 0., 0.,]

    def info(self):
        kw_list = [None] * self._n_shapes
        for ii, _ in enumerate(self._shapes):
            kw_list[ii] = _.info()
        return kw_list

    def shape_name(self):
        """
            形状（このクラスは集団であるが）の名称を取得する。
        """
        return self._shape_name

    def shapes_name(self):
        return self._shapes_name[:]

    def centers(self, polar=False):
        """
            各粒子の中心の座標を返す。
            polarがTrueの場合は極座標(r, theta, phi)で返す。
        """
        buff = self._centers.copy()
        if polar is False:
            return buff
        else:
            r = np.linalg.norm(buff, axis=1)
            theta = np.arctan2(np.sqrt(buff[:,0]**2+buff[:,1]**2), buff[:,2])
            phi = np.arctan2(buff[:,1], buff[:,0]) + np.pi
            return r, theta, phi


    def EulerRot(self, euler):
        """
            ToDo: 実装
        """
        # self.__shape_mother.EulerRot(euler)
        # self.center_daughters = mf.EulerRotation(self.center_daughters, euler, 1).copy()
        pass

    def Slice(self, xx, yy, z, *args, **kwargs):
        """
            断面のスライスを返す。
            < Input parameters >
                xx, yy  : 2D coordinates (N*M arrays)
                z       : z coordinate
                *args   : Option
                **kwargs: Option
        """
        # 娘クラスターのスライス。
        # "z" 周りに存在する娘クラスターのみを対象にする。
        ind_hit = np.where(abs(z - self._centers[:,2]) <= self._as_range)[0]
        if len(ind_hit) <= 0:
            ind = np.zeros(xx.shape, dtype=bool)
        else:
            ind = None
            for jj in ind_hit:
                if ind is None:
                    ind = self._shapes[jj].Slice(xx, yy, z)
                else:
                    ind2 = self._shapes[jj].Slice(xx, yy, z)
                    ind |= ind2

        # インデックスで返すかどうか。
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')
        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out
        pass

    def SliceSurface(self, xx, yy, z, width, *args, **kwargs):
        """
            表面のスライスを返す（今のところ実装は考えていない）
        """
        return self.Slice(xx, yy, z, *args, **kwargs)
