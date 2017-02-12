# -*- coding: utf-8 -*-

import numpy as np

from .. import core
from ..core import mathfunctions as mf

class hailstone_with_sphere(object):
    '''
        形状１の表面にいくらかの球がめり込んだような雹状の形状を与えるクラス。
        形状１の表面上の点を引数として球を構築する。
    '''
    def __init__(self, shape, *args, **kwargs):
        '''
            初期化。このとき親形状１は形状クラスを渡す。
        '''
        self.__shape_name = shape.shape_name()
        self.__shape = shape
        self.center = shape.center
        self.a = self.__shape.a

        _saved = False if kwargs.get('saved') is None else kwargs.get('saved')
        if _saved is True:
            self.SetFromDict(**kwargs)
            return

        # Flag for superposition mode
        self.__superpos = False if kwargs.get('superpos') is None else kwargs.get('superpos')

        if self.__superpos is False:
            # Get information from kwargs
            n_hail = 0 if kwargs.get('n_hail') is None else kwargs.get('n_hail')
            if type(n_hail) not in [int, np.int32]:
                raise ValueError("n_hail must be positive integer.")
            if n_hail <= 0:
                raise ValueError("n_hail must be positive integer.")
            self.__n_hails = n_hail

            a_hail_max = kwargs.get('a_hail_max')
            self.a_hail_max = self.__shape.a/10. if a_hail_max is None else a_hail_max
            self.a_range = (self.__shape.a + self.a_hail_max)*1.1

            if shape.shape_name() in ['sphere', 'spheroid']:
                # Create spheroidal branches of a hailstone
                self.a_hails = self.a_hail_max*np.random.rand(n_hail)
                self.theta = np.pi*np.random.rand(n_hail)
                self.phi = 2.*np.pi*np.random.rand(n_hail)

                self.__center_hails = np.zeros((n_hail, 3), dtype=float)
                self.__center_hails[:,0] = self.__shape.ax*np.sin(self.theta)*np.sin(self.phi) + self.__shape.center[0]
                self.__center_hails[:,1] = self.__shape.ay*np.sin(self.theta)*np.cos(self.phi) + self.__shape.center[1]
                self.__center_hails[:,2] = self.__shape.az*np.cos(self.theta) + self.__shape.center[2]
                self.center_hails = self.__center_hails.copy()

            else:
                coor1 = kwargs.get('coor')
                if coor1 is None:
                    raise ValueError("No information on the surface.")

                inds = range(0, coor1.shape[0])
                inds = np.random.choice(inds, n_hail, replace=False)
                coor = coor1[inds,:]
                self.a_hails = self.a_hail_max*np.random.rand(n_hail)
                self.__center_hails = coor
                self.center_hails = coor.copy()
        else: # Superposition mode
            self.__n_hails = 1
            a_hail_max = kwargs.get('a_hail_max')
            self.a_hail_max = self.__shape.a/10. if a_hail_max is None else a_hail_max
            self.a_hails = np.array([self.a_hail_max])
            self.a_range = max([self.__shape.a, self.a_hail_max])*1.1
            self.__center_hails = np.zeros((1,3), dtype=float)
            self.center_hails = self.__center_hails.copy()

    def hailsInfo(self):
        _hailsInfo=dict(n_hail=self.__n_hails, a_hail_max=self.a_hail_max,
                        a_hails=self.a_hails, center_hails=self.__center_hails)
        return _hailsInfo

    def SetFromDict(self, **kwargs):
        n_hail = kwargs.get('n_hail')
        if type(n_hail) not in [int, np.int32]:
            raise ValueError("h_hail must be positive integer.")
        if n_hail <= 0:
            raise ValueError("h_hail must be positive integer.")
        self.__n_hails = n_hail

        a_hail_max = kwargs.get('a_hail_max')
        self.a_hail_max = self.__shape.a/10. if a_hail_max is None else a_hail_max
        self.a_range = (self.__shape.a + self.a_hail_max)*1.1

        if kwargs.get('a_hails') is None or kwargs.get('center_hails') is None:
            raise ValueError('No information on a hailstone.')
        self.a_hails = kwargs.get('a_hails')
        self.__center_hails = kwargs.get('center_hails')
        self.center_hails = self.__center_hails + 0.0

    def GetCenterHails(self, angle=False):
        if angle is False:
            return 1.*self.__center_hails
        else:
            buff = 1.*self.__center_hails
            phi = np.arctan2(buff[:,1], buff[:,0])+np.pi
            theta = np.arctan2(np.sqrt(buff[:,0]**2+buff[:,1]**2), buff[:,2])
            return theta, phi

    def EulerRot(self, euler):
        self.__shape.EulerRot(euler)
        self.center_hails = mf.EulerRotation(self.__center_hails, euler, 1) + 0.0

    def shape_name(self):
        return self.__shape_name

    def Slice(self, xx, yy, z, *args, **kwargs):
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')
        is_trunc = False if kwargs.get('is_trunc') is None else kwargs.get('is_trunc')
        ind = self.__shape.Slice(xx, yy, z)
        if is_trunc is False:
            ind_b = np.where(abs(z - self.center_hails[:,2]) <= self.a_hails)[0]
            for jj in ind_b:
                ind = ind | ((xx - self.center_hails[jj, 0])**2
                             + (yy - self.center_hails[jj, 1])**2
                             <= self.a_hails[jj]**2 - (z - self.center_hails[jj, 2])**2)
        else:
            ind = ind & ((xx - self.center_hails[0, 0])**2
                         + (yy - self.center_hails[0, 1])**2
                         <= self.a_hails[0]**2 - (z - self.center_hails[0, 2])**2)
        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

    def SliceSurface(self, xx, yy, z, width, *args, **kwargs):
        is_ind = True if kwargs.get('is_ind') is None else kwargs.get('is_ind')
        is_trunc = False if kwargs.get('is_trunc') is None else kwargs.get('is_trunc')
        ind = self.__shape.Slice(xx, yy, z)
        ind_b = np.where(abs(z-self.center_hails[:,2]) <= self.a_hails)[0]
        if is_trunc is False:
            for jj in ind_b:
                ind = ind | ((xx-self.center_hails[jj, 0])**2
                             + (yy-self.center_hails[jj, 1])**2
                             <= self.a_hails[jj]**2 - (z-self.center_hails[jj, 2])**2)
        else:
            ind = ind & ((xx-self.center_hails[0, 0])**2
                         + (yy-self.center_hails[0, 1])**2
                         <= self.a_hails[0]**2 - (z-self.center_hails[0, 2])**2)
        if is_ind is True:
            return ind
        else:
            ind = ind.reshape(xx.shape[0]*xx.shape[1], 1)[:,0]
            out = np.zeros((np.sum(ind), 3), dtype=float)
            out[:,0] = xx.reshape(xx.shape[0]*xx.shape[1], 1)[ind,0]
            out[:,1] = yy.reshape(yy.shape[0]*yy.shape[1], 1)[ind,0]
            out[:,2] = z
            return out

class hailstone(object):
    '''
        雹形クラスターを生成するクラス。
        母クラスターと娘クラスターの種類、サイズ、位置を指定して生成できるようにする。
        娘クラスターのサイズと位置をランダムに与えて生成する機能はオプションとする。
        つまり外部で初期位置などを与えるようにする。
        娘クラスターは母クラスターから離れていないことを想定しているが、今のところエラー処理はしない。
            （娘クラスターの半径） + （母クラスターの半径） >= （中心間の距離）
        また、娘クラスターの中心は母クラスターの中心からの相対位置とする。

        スライスは "(mother) | (dauguters)" でのみ与えるものとして、ほかの余計なオプションは与えない。
        （例えば"superpos" modeなどは別のクラスで与える）
    '''
    def __init__(self, shape_mother, shape_daughters, *args, **kwargs):
        '''
            クラスの初期化。
            < Input parameters >
                shape_mother     : 母クラスターオブジェクト
                shape_daughters  : 娘クラスターオブジェクトまたはそのリスト
                *args            : オプション
                **kwargs         : オプション
        '''
        self._shape_name = "hailstone"

        self.__shape_mother = shape_mother
        self.__shape_daughters = shape_daughters

        # 娘クラスターの数を計算
        if type(shape_daughters) != list:
            self.__n_daughters = 1
        else:
            self.__n_daughters = len(self.__shape_daughters)

        # 各クラスターの種類を保持する
        self.__shape_name_mother = shape_mother.shape_name()
        self.__shape_name_daughters = []
        if type(shape_daughters) != list:
            self.__shape_name_daughters.append(shape_daughters.shape_name())
        else:
            for daughter in shape_daughters:
                self.__shape_name_daughters.append(daughter.shape_name())

        # 各クラスターの中心の位置情報を保持する
        self.center = shape_mother.center
        self.center_daughters = np.zeros((self.__n_daughters, 3), dtype=float)
        if type(shape_daughters) != list:
            self.center_daughters[0,:] = shape_daughters.center[:]
        else:
            for ii, daughter in enumerate(shape_daughters):
                self.center_daughters[ii,:] = daughter.center[:]

        # Euler回転前の情報を保持
        self.__center_daughters = self.center_daughters.copy()

        # 各クラスターの中心と原点の間の距離を保持する
        self.d_center = np.linalg.norm(self.center)
        self.d_center_daughters = []
        for _ in self.center_daughters:
            self.d_center_daughters.append(np.linalg.norm(_))

        # 各クラスターの特徴的長さの情報を保持する
        self.a_mother = shape_mother.a
        self.a_daughters = []
        if type(shape_daughters) != list:
            self.a_daughters.append(shape_daughters.a)
        else:
            for daughter in shape_daughters:
                self.a_daughters.append(daughter.a)

        # 各クラスターの空間的広がり "a_ratio" を保持する
        self.a_range_mother = shape_mother.a_range
        self.a_range_daughters = []
        if type(shape_daughters) != list:
            self.a_range_daughters.append(shape_daughters.a)
        else:
            for daughter in shape_daughters:
                self.a_range_daughters.append(daughter.a_range)

        # このオブジェクトの特徴的長さと空間的広がりを計算
        self.a = self.a_mother
        for _a, _d in zip(self.a_daughters, self.d_center_daughters):
            self.a = max([self.a, _d + _a])

        self.a_range = self.a_range_mother
        for _a_range, _d in zip(self.a_range_daughters, self.d_center_daughters):
            self.a_range = max([self.a_range, _d + _a_range])

    def shape_name(self):
        """
            クラスターの種類を取得する。
        """
        return self._shape_name

    def shape_name_mother(self):
        """
            母クラスターの種類を取得する。
        """
        return self.__shape_name_mother + ""

    def shape_name_daughters(self):
        """
            娘クラスターの種類を取得する。
        """
        return self.__shape_name_daughters[:]

    def n_daughters(self):
        """
            娘クラスターの数を取得する。
        """
        return self.__n_daughters

    def info(self):
        _mother_kwargs = self.__shape_mother.info()
        _daughter_kwargs = []
        for _daughter in self.__shape_daughters:
            _daughter_kwargs.append(_daughter.info())

        _ = dict(shape_name=self._shape_name, mother_kwargs=_mother_kwargs, daughter_kwargs=_daughter_kwargs)
        return _

    def SetFromDict(self, **kwargs):
        pass

    def GetCenterOfDaughters(self, angle=False, original=False):
        """
            娘クラスターの中心の位置を取得する。
            < Input parameters >
                angle : True = return daughters' positions as polar coordinates (r, theta, phi)
                original : True = return daughters' original positions.
        """
        if original is True:
            buff = self.__center_daughters.copy()
        else:
            buff = self.center_daughters.copy()
        if angle is False:
            return buff
        else:
            r = np.linalg.norm(buff, axis=1)
            theta = np.arctan2(np.sqrt(buff[:,0]**2+buff[:,1]**2), buff[:,2])
            phi = np.arctan2(buff[:,1], buff[:,0]) + np.pi
            return r, theta, phi

    def EulerRot(self, euler):
        """
            ToDo: 次の実装
                1. 母クラスターを回転（実装済み）
                2. 娘クラスターの中心位置を回転（実装済み）
                3. 娘クラスターを回転
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

        # 母クラスターのスライス。
        ind = self.__shape_mother.Slice(xx, yy, z)

        # 娘クラスターのスライス。
        # "z" 周りに存在する娘クラスターのみを対象にする。
        ind_hit = np.where(abs(z - self.center_daughters[:,2]) <= self.a_range_daughters)[0]
        for jj in ind_hit:
            ind_daughter = self.__shape_daughters[jj].Slice(xx, yy, z)
            ind |= ind_daughter

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

    def SliceSurface(self, xx, yy, z, width, *args, **kwargs):
        """
            表面のスライスを返す。（未実装、今のところ実装は考えていない）
        """
        return self.Slice(xx, yy, z, *args, **kwargs)
