# -*- coding: utf-8 -*-

import numpy as np

from .. import core
from ..core import mathfunctions as mf

from ..core.space import space
# from ..space import space
from .shapeslice import shapeslice

class polyhedron(shapeslice):
    """
        多面体クラス。
        各多面体の基本的な機能を持つ。
        任意の面ベクトル "NN" と定めた中心からの距離 "DD" のリストをもとに、それらで囲まれる領域を与える。
        必要に応じて面の表面エネルギー "GG" を利用するよう調整してよい。
    """
    def __init__(self, a, NN, DD, *args, **kwargs):
        """
            クラスの初期化。
            "DD"は"a"に対する割合で与えるものとする。
        """
        self._shape_name = "polyhedron" if kwargs.get("shape_name") is None else kwargs.get("shape_name")
        # Get information from kwargs
        euler = kwargs.get('euler')
        permute = kwargs.get('permute')

        if type(NN) == list:
            self._NN = np.array(NN)
        else:
            self._NN = NN.copy()
        if type(DD) == list:
            self.DD = np.array(DD)
        else:
            self.DD = DD.copy()

        # 面と中心の距離の調整
        if kwargs.get("distance_rate") is not None:
            _distance_rate = kwargs.get("distance_rate")
            if type(_distance_rate) == list:
                _distance_rate = np.array(_distance_rate)
            if type(_distance_rate) in [float, np.float32]:
                self.DD = (1. - _distance_rate) * self.DD
            else:
                if type(_distance_rate) != np.ndarray:
                    raise TypeError("Invalid type for `distance_rate`.")
                elif len(_distance_rate) != len(self.DD):
                    raise ValueError("The length of `distance_rate` is not equal to `DD`.")
                self.DD = (1. - _distance_rate) * self.DD

        GG = kwargs.get("GG")
        if GG is None:
            self.GG = np.zeros(len(self.DD), dtype=float)
        elif type(GG) == list:
            self.GG = np.array(GG)
        elif type(GG) == np.ndarray and len(GG.shape) == 1:
            self.GG = GG.copy()
        else:
            raise TypeError("`GG` must be an 1-D array.")

        self.center = [0., 0., 0.] if kwargs.get('center') is None else kwargs.get('center')

        # 面の有効性のチェック
        self._check_poly_validity()

        self.a = a
        self.a_range = self.a*1.5 if kwargs.get("a_range") is None else kwargs.get("a_range")

        # Permutation of normal vectors
        self.perm = [0, 1, 2] if permute is None else permute

        # 辺の面取り
        _chamfer_edge_rate = kwargs.get("chamfer_edge_rate")
        _chamfer_vertex_rate = kwargs.get("chamfer_vertex_rate")
        if _chamfer_edge_rate is not None or _chamfer_vertex_rate is not None:
            if _chamfer_edge_rate is not None: # 先に辺から面取りを行う
                self.chamfer_edge_rate = _chamfer_edge_rate
                self.chamfering("e", _chamfer_edge_rate)
            if _chamfer_vertex_rate is not None: # 先に辺から面取りを行う
                self.chamfer_vertex_rate = _chamfer_vertex_rate
                self.chamfering("v", _chamfer_vertex_rate)
        else:
            """ 旧形式 """
            if kwargs.get('chamfer') is not None:
                chamfer = kwargs.get('chamfer')
            elif kwargs.get('mid') is not None:
                chamfer = kwargs.get('mid')
            else:
                chamfer = 1.0
            self.chamfer = chamfer
            rand = False if kwargs.get('rand') is None else kwargs.get('rand')
            if self.chamfer < 1.0:
                _mid = self.midpoints(1.).transpose()
                _dd = []
                _gg = []

                # Random chamferring
                rrr = np.random.rand(200)
                sigma = 0.1
                med = 0.5
                e_rand = np.exp(-(rrr-med)**2/2./sigma**2)
                self.rand = np.ones(_mid.shape[0], dtype=float) if rand is False else e_rand[0:_mid.shape[0]]

                for ii in range(_mid.shape[0]):
                    _leng = (1.-(1.-self.chamfer)*self.rand[ii])*np.linalg.norm(_mid[ii,:])
                    _mid[ii,:] /= np.linalg.norm(_mid[ii,:])
                    _dd.append(_leng)
                    _gg.append(self.gamma_max*_leng)
                self._NN = np.vstack((self._NN, _mid))
                self.DD = np.hstack((self.DD, np.array(_dd)))
                self.GG = np.hstack((self.GG, np.array(_gg)))

        # Initialize shapeslice class.
        self.NN = self._NN.copy()
        shapeslice.__init__(self, self._shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm, center=self.center)

    def EulerRot(self, euler):
        self.NN = mf.EulerRotation(self._NN, euler, 1)
        self.UpdSlice()

    def UpdSlice(self):
        shapeslice.__init__(self, self._shape_name, self.a,
                            NN=self.NN, DD=self.DD, perm=self.perm, center=self.center)

    def shape_name(self):
        return self._shape_name + ""

    def midpoints(self, a, *args, **kwargs):
        """
            多面体の辺の中点を取得する。
            返り値は(3, N)のnumpy.ndarrayとする。
            実装はこのクラスを継承する子クラスで行う。
            （任意の多面体の場合に利用できるようにしたいが、、優先順位は低い）
            判定用にNoneを返すようにしている。
        """
        return None

    def vertices(self, a, *args, **kwargs):
        """
            多面体の頂点を取得する。
            返り値は(3, N)のnumpy.ndarrayとする。
            実装はこのクラスを継承する子クラスで行う。
            （任意の多面体の場合に利用できるようにしたいが、、優先順位は低い）
            判定容易にNoneを返すようにしている。
        """
        return None

    def chamfering(self, v_or_e, chamfer_rate):
        """
            多面体の辺を中線に垂直な面で面取りする。
            v_or_e      : 頂点("v")か辺("e")か。
            chamfer_rate: 面取りの深さ（numpy.ndarray / list）。
                          実際の距離は 1. - chamfer_rate
        """
        if v_or_e == "v":
            _chamf = self.vertices(1.)
        elif v_or_e == "e":
            _chamf = self.midpoints(1.)
        else:
            raise ValueError("`v_or_e` must be in ['v', 'e'].")
        if _chamf is None:
            raise NotImplementedError("Not implemented: `chamering`")
        _chamf = _chamf.transpose()
        _dd = []
        _gg = []

        if type(chamfer_rate) == list:
            chamfer_rate = np.array(chamfer_rate)

        if len(chamfer_rate) != _chamf.shape[0]:
            _ = "vertice" if v_or_e == "v" else "edges"
            raise ValueError("The length of `chamfer_rate` is not equal to  the number of `{0}`.".format(_))

        for ii in range(_chamf.shape[0]):
            _leng = (1. - chamfer_rate[ii])*np.linalg.norm(_chamf[ii,:])
            _chamf[ii,:] /= np.linalg.norm(_chamf[ii,:])
            _dd.append(_leng)
            _gg.append(self.gamma_max*_leng)

        self._NN = np.vstack((self._NN, _chamf))
        self.DD = np.hstack((self.DD, np.array(_dd)))
        self.GG = np.hstack((self.GG, np.array(_gg)))

        self._kwargs = kwargs

    def info(self):
        """
            particleshapeメソッドで生成するための情報を返す
            こちらはspheroidと異なりオプションが多すぎるので、kwargsはそのまま返す
        """
        return dict(shape_name=self._shape_name, a=self.a, NN=self._NN, DD=self.DD, kwargs=self._kwargs)

    def _check_poly_validity(self):
        """
            与えられた面の有効性のチェック。
            < 多面体を構成するかどうかの判定条件 >
            max(DD)の3倍のサイズをもつ立方体を用意し、
            その側面に属する点の少なくとも一つが少なくとも一つの面より内側にある場合はアウトとする。
            つまりどの側面もどの面よりも外側にあればOKである。
        """
        # 外空間の用意
        _xmax = max(self.DD)*3.
        _N = 10
        sp1 = space(_N, _xmax, ymax=_xmax, Ny=_N, zmax=_xmax, Nz=_N)
        dx = sp1.dx()[0]
        x1, y1, z1 = sp1.range_space()
        xx, yy, zz = np.meshgrid(x1, y1, z1)
        xx = np.reshape(xx, (1, np.size(xx)))[0]
        yy = np.reshape(yy, (1, np.size(yy)))[0]
        zz = np.reshape(zz, (1, np.size(zz)))[0]

        coor = np.vstack((np.vstack((xx, yy)), zz)).transpose()

        # 内空間を除く
        ind = []
        ind_inside = []
        for ii, row in enumerate(coor):
            if row[(row == -_xmax) | (row == _xmax-dx)].size != 0:
                ind.append(ii)
            else:
                ind_inside.append(ii)

        coor_outside = coor[ind, :]
        coor_outside[:, 0] = coor_outside[:, 0] - self.center[0]
        coor_outside[:, 1] = coor_outside[:, 1] - self.center[1]
        coor_outside[:, 2] = coor_outside[:, 2] - self.center[2]
        # coor_inside = coor[ind_inside, :]

        # 判定
        ind = np.ones(coor_outside.shape[0], dtype=bool)
        for jj in range(len(self._NN)):
            ind = ind & (coor_outside[:,0]*self._NN[jj,0]
                        +coor_outside[:,1]*self._NN[jj,1]
                        +coor_outside[:,2]*self._NN[jj,2]
                        <= self.DD[jj])
        if sum(ind) != 0:
            raise ValueError("Invalid parameters for constructing a polyhedron.")

""" --- ほかの関数での単体利用のための定義 --- """
def check_poly_validity(NN, DD, center=[0., 0., 0.]):
    """
        与えられた面の有効性のチェック。
        < 多面体を構成するかどうかの判定条件 >
        max(DD)の3倍のサイズをもつ立方体を用意し、
        その側面に属する点の少なくとも一つが少なくとも一つの面より内側にある場合はアウトとする。
        つまりどの側面もどの面よりも外側にあればOKである。
    """
    # 外空間の用意
    _xmax = max(DD)*3.
    _N = 10
    sp1 = space(_N, _xmax, ymax=_xmax, Ny=_N, zmax=_xmax, Nz=_N)
    dx = sp1.dx()[0]
    x1, y1, z1 = sp1.range_space()
    xx, yy, zz = np.meshgrid(x1, y1, z1)
    xx = np.reshape(xx, (1, np.size(xx)))[0]
    yy = np.reshape(yy, (1, np.size(yy)))[0]
    zz = np.reshape(zz, (1, np.size(zz)))[0]

    coor = np.vstack((np.vstack((xx, yy)), zz)).transpose()

    # 内空間を除く
    ind = []
    ind_inside = []
    for ii, row in enumerate(coor):
        if row[(row == -_xmax) | (row == _xmax-dx)].size != 0:
            ind.append(ii)
        else:
            ind_inside.append(ii)

    coor_outside = coor[ind, :]
    coor_outside[:, 0] = coor_outside[:, 0] - center[0]
    coor_outside[:, 1] = coor_outside[:, 1] - center[1]
    coor_outside[:, 2] = coor_outside[:, 2] - center[2]
    # coor_inside = coor[ind_inside, :]

    # 判定
    ind = np.ones(coor_outside.shape[0], dtype=bool)
    for jj in range(len(NN)):
        ind = ind & (coor_outside[:,0]*NN[jj,0]
                    +coor_outside[:,1]*NN[jj,1]
                    +coor_outside[:,2]*NN[jj,2]
                    <= DD[jj])
    return sum(ind) == 0
