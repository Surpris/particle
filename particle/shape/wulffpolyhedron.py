# -*- coding: utf-8 -*-

import numpy as np

from .. import core
from ..core import mathfunctions as mf

from .polyhedron import *

class wulffpolyhedron(polyhedron):
    '''
    Wulff polyhedronを与えるクラス。
    内部変数として一辺の長さ、オイラー回転角等を持つ。

    __init__中の引数について。
        a       : length of edge
        kwargs  : options
    '''

    _shape_name = 'wulffpolyhedron'
    # 最初の時点では頂点などの数はわからないためゼロに設定しておく。
    n_vertices = 0
    n_edges = 0
    n_faces = 0
    n_shapeinfo = n_vertices + n_edges + n_faces

    # 一応、 (111), (110), (100) の普通の場合の辺などの数を提供できるようにしておく。
    n_vertices_111_110_100 = 48
    n_edges_111_110_100 = 72
    n_faces_111_110_100 = 26
    n_shapeinfo_111_110_100 = n_vertices_111_110_100 + n_edges_111_110_100 + n_faces_111_110_100

    # 同様に、 (111), (100) の普通の場合の辺などの数を提供できるようにしておく。
    n_vertices_111_100 = 24
    n_edges_111_100 = 36
    n_faces_111_100 = 14
    n_shapeinfo_111_100 = n_vertices_111_100 + n_edges_111_100 + n_faces_111_100

    def __init__(self, a, *args, **kwargs):
        """
            クラスの初期化
        """

        polytype = kwargs.get('polytype')
        gamma = [62.1, 64.1, 67.3]  if kwargs.get('gamma') is None else kwargs.get('gamma')

        self.gamma = gamma # (111), (110), (100)
        self.gamma_max = max(self.gamma)
        n_111 = mf.MillerNormalVectors_111()
        n_110 = mf.MillerNormalVectors_110()
        n_100 = mf.MillerNormalVectors_100()

        self.polytype='111_110_100' if polytype is None else polytype
        if self.polytype.find('111_100') != -1: # for (111), (100)
            _NN = np.vstack((n_111, n_100))

            _DD = np.zeros(len(_NN), dtype=float)
            _DD[0:8] = self.gamma[0]/self.gamma_max
            _DD[8:14] = self.gamma[2]/self.gamma_max

            _GG = np.zeros(len(_NN), dtype=float)
            _GG[0:8] = self.gamma[0]
            _GG[8:14] = self.gamma[2]

            self.n_vertices = 24
            self.n_edges = 36
            self.n_faces = 14

        else: # for (111), (110), (100)
            _NN = np.vstack((n_111, n_110, n_100))

            _DD = np.zeros(len(_NN), dtype=float)
            _DD[0:8] = self.gamma[0]/self.gamma_max
            _DD[8:20] = self.gamma[1]/self.gamma_max
            _DD[20:26] = self.gamma[2]/self.gamma_max

            _GG = np.zeros(len(_NN), dtype=float)
            _GG[0:8] = self.gamma[0]
            _GG[8:20] = self.gamma[1]
            _GG[20:26] = self.gamma[2]

            self.n_vertices = 48
            self.n_edges = 72
            self.n_faces = 26

        _a_range = a*1.2 # = a*1.0879*1.1

        # Initialize polyhedron class
        kwargs["shape_name"] = self._shape_name
        kwargs["GG"] = _GG
        kwargs["a_range"] = _a_range
        polyhedron.__init__(self, a,
                            _NN, _DD, **kwargs)

        self._kwargs = kwargs

    def vertices(self, a, gamma=[62.1, 64.1, 67.3]):
        """
            2016/09/28 (Wed.) 現在、 `111_110_100` のタイプの場合しか実装されていない。
        """
        gamma_max = max(gamma)

        # Coefficient matices of first quadrant.
        CoefMat = []
        CoefMat.append(np.array([[1,1,1],[0,1,1],[0,0,1]]))   # (111), (011), (001)
        CoefMat.append(np.array([[1,1,1],[1,0,1],[0,0,1]]))   # (111), (101), (001)
        CoefMat.append(np.array([[1,1,1],[1,0,1],[1,0,0]]))   # (111), (101), (100)
        CoefMat.append(np.array([[1,1,1],[1,1,0],[1,0,0]]))   # (111), (110), (100)
        CoefMat.append(np.array([[1,1,1],[1,1,0],[0,1,0]]))   # (111), (110), (010)
        CoefMat.append(np.array([[1,1,1],[0,1,1],[0,1,0]]))   # (111), (011), (010)

        # Determinant vector for (111), (110) and (100).
        T = a/gamma_max*np.array([np.sqrt(3)*gamma[0],np.sqrt(2)*gamma[1],gamma[2]])

        # Calculation vertices in first quadrant.
        out1 = []
        for ii  in range(6):
            out1.append(np.linalg.solve(CoefMat[ii],T))

        # Symmetrical displacement.
        out = np.zeros((3,48))
        for ii in range(6):
            out[:,ii] = np.array([out1[ii][0], out1[ii][1], out1[ii][2]])
            out[:,ii+6] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], out1[ii][2]]),[0,0,90],0); # yz-plane
            out[:,ii+12] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], out1[ii][2]]),[0,0,180],0);  # z-axis
            out[:,ii+18] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], out1[ii][2]]),[0,0,270],0);  # zx-plane
            out[:,ii+24] = np.array([out1[ii][0], out1[ii][1], -out1[ii][2]])  # xy-plane
            out[:,ii+30] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], -out1[ii][2]]),[0,0,90],0);   # x-axis
            out[:,ii+36] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], -out1[ii][2]]),[0,0,180],0); # origin
            out[:,ii+42] = mf.EulerRotation(np.array([out1[ii][0], out1[ii][1], -out1[ii][2]]),[0,0,270],0); # y-axis

        return out


    def midpoints(self, a, gamma=[62.1, 64.1, 67.3]):
        """
            2016/09/28 (Wed.) 現在、 `111_110_100` のタイプの場合しか実装されていない。
        """
        vert = self.vertices(a, gamma)

        out1 = np.zeros((3,10))
        out1[:,0] = (vert[:,0]+vert[:,1])/2
        out1[:,1] = (vert[:,2]+vert[:,3])/2
        out1[:,2] = (vert[:,4]+vert[:,5])/2
        out1[:,3] = (vert[:,1]+vert[:,2])/2
        out1[:,4] = (vert[:,3]+vert[:,4])/2
        out1[:,5] = (vert[:,5]+vert[:,0])/2
        out1[:,6] = (vert[:,0]+vert[:,7])/2
        out1[:,7] = (vert[:,5]+vert[:,8])/2
        out1[:,8] = (vert[:,3]+vert[:,27])/2
        out1[:,9] = (vert[:,4]+vert[:,28])/2

        out = np.zeros((3,72))
        for ii in range(8):
            out[:,ii] = out1[:,ii]
            out[:,ii+8] = mf.EulerRotation(out1[:,ii],[0,0,90],0) # yz-plane
            out[:,ii+16] = mf.EulerRotation(out1[:,ii],[0,0,180],0)  # z-axis
            out[:,ii+24] = mf.EulerRotation(out1[:,ii],[0,0,270],0)  # zx-plane
            out[:,ii+32] = np.array([out1[0,ii], out1[1,ii], -out1[2,ii]])  # xy-plane
            out[:,ii+40] = mf.EulerRotation(np.array([out1[0,ii], out1[1,ii], -out1[2,ii]]),[0,0,90],0)   # x-axis
            out[:,ii+48] = mf.EulerRotation(np.array([out1[0,ii], out1[1,ii], -out1[2,ii]]),[0,0,180],0) # origin
            out[:,ii+56] = mf.EulerRotation(np.array([out1[0,ii], out1[1,ii], -out1[2,ii]]),[0,0,270],0) # y-axis

        out[:,64] = out1[:,8]
        out[:,65] = mf.EulerRotation(out1[:,8],[0,0,90],0); # yz-plane
        out[:,66] = mf.EulerRotation(out1[:,8],[0,0,180],0);  # z-axis
        out[:,67] = mf.EulerRotation(out1[:,8],[0,0,270],0);  # zx-plane
        out[:,68] = out1[:,9]
        out[:,69] = mf.EulerRotation(out1[:,9],[0,0,90],0); # yz-plane
        out[:,70] = mf.EulerRotation(out1[:,9],[0,0,180],0);  # z-axis
        out[:,71] = mf.EulerRotation(out1[:,9],[0,0,270],0);  # zx-plane

        return out

    def info(self):
        """
            particleshapeメソッドで生成するための情報を返す
        """
        return dict(shape_name=self._shape_name, a=self.a, NN=self._NN, DD=self.DD, kwargs=self._kwargs)