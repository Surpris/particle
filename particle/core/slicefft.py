# -*- coding: utf-8 -*-

# System modules
import numpy as np
from numpy.fft import *
from matplotlib import pyplot as plt
import importlib
# import sys
# sys.path.append("./")

# Use modules
from .space import space

# Check availability of "pyfftw" module
spam_spec = importlib.util.find_spec("pyfftw")
found = spam_spec is not None
if found is True:
    import pyfftw
spam_spec = importlib.util.find_spec("multiprocessing")
found = spam_spec is not None
if found is True:
    import multiprocessing

# Check availability of "pycuda" and "skcuda" modules
spam_spec = importlib.util.find_spec("pycuda")
found_pycuda = spam_spec is not None
spam_spec = importlib.util.find_spec("skcuda")
found_skcuda = spam_spec is not None
if found_pycuda is True and found_skcuda is True:
    found_cufft = True
    import pycuda.gpuarray as gpuarray
    import skcuda.fft as cu_fft
    import pycuda.autoinit
else:
    found_cufft = False

def funcfftw(F, *args, **kwargs):
    if found is True and kwargs.get('fft_type') == 'fftw':
        pyfftw.forget_wisdom()
        func = pyfftw.builders.fft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
        return func()
    elif found_cufft is True and kwargs.get('fft_type') == 'cufft':
        x_gpu = gpuarray.to_gpu(F.astype(np.complex64))
        xf_gpu = gpuarray.empty(F.shape, np.complex64)
        cu_fft.fft(x_gpu, xf_gpu, args[0])
        return xf_gpu.get()
    else:
        return fft2(F)

def ifuncfftw(F, *args, **kwargs):
    if found_cufft is True and kwargs.get('fft_type') == 'cufft':
        x_gpu = gpuarray.to_gpu(F.astype(np.complex64))
        xf_gpu = gpuarray.empty(F.shape, np.complex64)
        cu_fft.ifft(x_gpu, xf_gpu, args[0], True)
        return xf_gpu.get()
    elif found is True and kwargs.get('fft_type') == 'fftw':
        pyfftw.forget_wisdom()
        ifunc = pyfftw.builders.ifft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
        return ifunc()
    else:
        return ifft2(F)

class slicefft(space):
    '''
    X線領域でのMulti-Slice Fourier Transform (MSFT) を計算するクラス。
    多重散乱を考慮しない。
    Thomson scatteringを仮定。Valence electronもすべてfree (conduction) electronとみなしている。
    2016/09/13 (Tue.) 現在、屈折率を与える形でターゲットによる光の吸収効果を導入。

    主な使い方は、初期化して"MSFT"関数を実行することである。

    実装している関数はつぎのようになっている。
        < 内部利用 >
        __init__ : 初期化

    '''
    def __init__(self, Nx, xmax, **kwargs):
        """
            クラスの初期化。
            引数について、
                Nx, xmax : spaceの初期化で利用。
                kwargs   : speceとslicefftの初期化で利用。
        """
        space.__init__(self, Nx, xmax, **kwargs)
        self.InitPhoton(**kwargs)
        self.InitMSFT(**kwargs)
        self.InitAxes(**kwargs)
        self.InitDensity(**kwargs)

        """
            fft_type の有効性のチェック。
            cufft、fftw、numpyの順に優先する。
        """
        if kwargs.get('fft_type') is None:
            self.__fft_type = 'cufft' # とりあえずcufftを最優先
        elif kwargs.get('fft_type') not in ['numpy', 'fftw', 'cufft']:
            raise ValueError('Invalid value for the keyword "fft_type."')
        else:
            self.__fft_type = kwargs.get('fft_type')

        # cufftの有効性のチェック
        if self.__fft_type == 'cufft':
            if found_cufft is True: # cudaがある場合
                buff = self.mesh_space()[0].shape
                self.__x_gpu = gpuarray.empty(buff, np.complex64)
                self.__xf_gpu = gpuarray.empty(buff, np.complex64)
                self.__plan = cu_fft.Plan(buff, np.complex64, np.complex64)
            else: # cufftがない場合はfftwに切り替える
                self.__fft_type = 'fftw'

        # fftwの有効性のチェック
        if self.__fft_type == 'fftw':
            if found is False: # fftwがない場合はnumpyに切り替える。
                self.__fft_type = 'numpy'

    def InitMSFT(self, **kwargs):
        """
            MSFT関連の初期化。
        """
        self._rho0 = None if kwargs.get('rho0') is None else kwargs.get('rho0')
        self._F = None if kwargs.get('F') is None else kwargs.get('F')
        self.__coor_types = ['body', 'surf']
        self._coor_type = 'coor' if kwargs.get('coor_type') is None else kwargs.get('coor_type')
        self.kmax = self._k / 20.
        if kwargs.get('kmax') is not None: self.kmax = kwargs.get('kmax')
        self._rrmax = None if kwargs.get('rrmax') is None else kwargs.get('rrmax')
        self._rmax = None if kwargs.get('rmax') is None else kwargs.get('rmax')
        self._qmax = None if kwargs.get('qmax') is None else kwargs.get('qmax')
        self._qmode = True if kwargs.get('qmode') is None else kwargs.get('qmode')
        self.__calc_count = 0
        self._shape = None

    def InitAxes(self, **kwargs):
        """
            プロット用に軸関連の初期化。
        """
        self._xlim = None if kwargs.get('xlim') is not None else kwargs.get('xlim')
        self._ylim = None if kwargs.get('ylim') is not None else kwargs.get('ylim')
        self._rcscale = 1. if kwargs.get('rcscale') is None else kwargs.get('rcscale')
        self._xscale = 1. if kwargs.get('xscale') is None else kwargs.get('xscale')
        self._yscale = 1. if kwargs.get('yscale') is None else kwargs.get('yscale')

        self._qxlim = None if kwargs.get('qxlim') is None else kwargs.get('qxlim')
        self._qylim = None if kwargs.get('qylim') is None else kwargs.get('qylim')
        self._qcscale = 1. if kwargs.get('qcscale') is None else kwargs.get('qcscale')
        self._qxscale = 1. if kwargs.get('qxscale') is None else kwargs.get('qxscale')
        self._qyscale = 1. if kwargs.get('qyscale') is None else kwargs.get('qyscale')

    def InitPhoton(self, **kwargs):
        """
            入射光関連の初期化。
        """
        self._lambda = 2.2e-1 if kwargs.get('wavelength') is None else kwargs.get('wavelength')
        self._k = 2*np.pi/self._lambda
        self._photons = 1e10 if kwargs.get('photons') is None else kwargs.get('photons')

    def InitDensity(self, **kwargs):
        """
            self._shape(Particleクラスを想定)の密度に関する情報の初期化。
            とりあえず屈折率の情報さえあればよいとする。
            屈折率の定義は n = 1 - delta - 1j*beta.
            もしbetaがマイナスの場合は符号を逆転させる。
        """
        self._refr = 1. if kwargs.get("refr") is None else kwargs.get("refr")
        if np.imag(self._refr) > 0.:
            self._refr = np.real(self._refr) - np.imag(self._refr)
        self._phase_factor = -1j*(self._refr-1.)*self._k*self.dx()[2] # 直接かかわるのはdeltaとbeta

    def fftInfo(self):
        _fftInfo = dict(rho0=self._rho0, F=self._F, coor_type=self._coor_type,
                       kmax=self.kmax, rrmax=self._rrmax, rmax=self._rmax,
                       qmax=self._qmax, qmode=self._qmode,
                       xlim=self._xlim, ylim=self._ylim, rcscale=self._rcscale,
                       xscale=self._xscale, yscale=self._yscale,
                       qxlim=self._qxlim, qylim=self._qylim, qcscale=self._qcscale,
                       qxscale=self._qxscale, qyscale=self._qyscale,
                       wavelength=self._lambda, photons=self._photons)
        return _fftInfo

    def SetParticle(self, shape):
        """
            Particleの設定。
            近い将来、この設定関数は削除する。
        """
        self._shape = shape

    def SetFFT(self):
        pass

    def __fft(self, F):
        """
            2D Forward Fourier Transformを実行する関数。
            引数について、
                F   : 被FFTデータ
        """
        if self.__fft_type not in ['numpy', 'fftw', 'cufft']:
            raise ValueError('Invalid parameter for the keyword "fft_type."')
        if found is True and self.__fft_type == 'fftw':
            pyfftw.forget_wisdom()
            func = pyfftw.builders.fft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
            return func()
        elif found_cufft is True and self.__fft_type == 'cufft':
            self.__x_gpu.set(F.astype(np.complex64))
            # xf_gpu = gpuarray.empty(F.shape, np.complex64)
            cu_fft.fft(self.__x_gpu, self.__xf_gpu, self.__plan)
            return self.__xf_gpu.get()
        else:
            return fft2(F)

    def __ifft(self, F):
        """
            2D Inverse Fourier Transformを実行する関数。
            引数について、
                F   : 被IFFTデータ
        """
        if self.__fft_type not in ['numpy', 'fftw', 'cufft']:
            raise ValueError('Invalid parameter for the keyword "fft_type."')
        if found is True and self.__fft_type == 'fftw':
            pyfftw.forget_wisdom()
            ifunc = pyfftw.builders.ifft2(F, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
            return ifunc()
        elif found_cufft is True and self.__fft_type == 'cufft':
            self.__x_gpu.set(F.astype(np.complex64))
            # xf_gpu = gpuarray.empty(F.shape, np.complex64)
            cu_fft.ifft(self.__x_gpu, self.__xf_gpu, self.__plan, True)
            return self.__xf_gpu.get()
        else:
            return ifft2(F)

    def MSFT(self, coor_type='body', **kwargs):
        if self._shape is None:
            raise ValueError("No information on the shape.")

        if coor_type not in self.__coor_types:
            raise ValueError("coor_type must be '{0}' or'{1}'.".format(self.__coor_types[0], self.__coor_types[1]))
        if coor_type is self.__coor_types[0]:
            _slice = self._shape.Slice
        elif coor_type is self.__coor_types[1]:
            _slice = self._shape.SliceSurface
        _qmode = True if kwargs.get('qmode') is None else kwargs.get('qmode')
        self._qmode = _qmode
        _atte = True if kwargs.get('atte') is None else kwargs.get('atte')
        self._atte = _atte
        self._coor_type = coor_type
        self._is_trunc = False if kwargs.get('is_trunc') is None else kwargs.get('is_trunc')

        # Set a global map.
        # Because of shift of the center of particle, maximum range should be expanded.
        self.a_range_expand = np.max([np.abs(self._shape.center[0]), np.abs(self._shape.center[1])])
        self._rrmax = np.array([self._shape.a_range + self.a_range_expand,
                                self._shape.a_range + self.a_range_expand,
                                self._shape.a_range + np.abs(self._shape.center[2])])
        _sprange_x, _sprange_y, _sprange_z = self.range_space(z_threshold=self._rrmax[2])
        rho0 = np.zeros((len(_sprange_x), len(_sprange_y)), dtype=float)

        # Set a spatial mesh.
        _xx, _yy = self.mesh_space(self._rrmax[0], self._rrmax[1])
        _ind_x = np.where(abs(_sprange_x) <= self._rrmax[0])[0]
        _ind_y = np.where(abs(_sprange_y) <= self._rrmax[1])[0]
        _ind_z = np.where(abs(_sprange_z) <= self._rrmax[2])[0]
        self._rmax = np.array([[_sprange_x[_ind_x.min()], _sprange_x[_ind_x.max()]],
                                [_sprange_y[_ind_y.min()], _sprange_y[_ind_y.max()]],
                                [_sprange_z[_ind_z.min()], _sprange_z[_ind_z.max()]]])

        # Set a frequency mesh.
        if kwargs.get('kmax') is not None: self.kmax = kwargs.get('kmax')

        _qx, _qy, _qz = self.range_anglefreq()
        _qzz = self.mesh_anglefreq(self.kmax, self.kmax, True)
        _qzz = np.sqrt(self._k**2-_qzz**2) - self._k
        _ind_qx = np.where(abs(_qx) <= self.kmax)[0]
        _ind_qy = np.where(abs(_qy) <= self.kmax)[0]
        _ind_qz = np.where(abs(_qz) <= self.kmax)[0]
        self._qmax = np.array([[_qx[_ind_qx.min()],_qx[_ind_qx.max()]],
                                [_qy[_ind_qy.min()],_qx[_ind_qy.max()]],
                                [_qz[_ind_qz.min()],_qz[_ind_qz.max()]]])

        is_rho = False if kwargs.get('is_rho') is None else kwargs.get('is_rho')

        """ --- Calculation --- """
        self._rho0 = np.zeros((_xx.shape[0], _xx.shape[1]), dtype = float)
        self._F = np.zeros((_qzz.shape[0], _qzz.shape[1]), dtype = complex)
        self.__calc_count = 0
        if is_rho is True:
            for ii, zz in enumerate(_sprange_z):
                rho1 = _slice(_xx, _yy, zz, self._dx, is_trunc=self._is_trunc)
                if sum(sum(rho1)) == 0:
                    continue
                if _atte is True:
                    rho1 *= np.exp(self._phase_factor*self.__calc_count)
                self._rho0 += rho1
                self.__calc_count += 1
            rho0[_ind_x.min():_ind_x.max()+1, _ind_y.min():_ind_y.max()+1] = self._rho0
            self.rho0 = rho0
        else:
            if _qmode is True:
                for ii, zz in enumerate(_sprange_z):
                    rho1 = _slice(_xx, _yy, zz, self._dx, is_trunc=self._is_trunc)
                    if sum(sum(rho1)) == 0:
                        continue
                    self._rho0 += rho1
                    rho0[_ind_x.min():_ind_x.max()+1, _ind_y.min():_ind_y.max()+1] = rho1.copy()
                    F1 = fftshift(self.__fft(rho0))[_ind_qx.min():_ind_qx.max()+1,
                                                    _ind_qy.min():_ind_qy.max()+1]
                    if _atte is True:
                        F1 *= np.exp(self._phase_factor*self.__calc_count)
                    self._F += F1*np.exp(-1j*zz*_qzz)
                    rho0[_ind_x.min():_ind_x.max()+1, _ind_y.min():_ind_y.max()+1] = 0
                    self.__calc_count += 1
                # rho0[_ind_x.min():_ind_x.max()+1, _ind_y.min():_ind_y.max()+1] = self._rho0
                self.rho0 = rho0
            else:
                if _atte is True:
                    self._rho0 = np.zeros((_xx.shape[0], _xx.shape[1]), dtype = complex)
                    rho0 = np.zeros((len(_sprange_x), len(_sprange_y)), dtype=complex)
                for ii, zz in enumerate(_sprange_z):
                    rho1 = _slice(_xx, _yy, zz, self._dx, is_trunc=self._is_trunc)
                    if sum(sum(rho1)) == 0:
                        continue
                    if _atte is True:
                        rho1 = np.exp(self._phase_factor*self.__calc_count)*rho1
                    self._rho0 += rho1
                    self.__calc_count += 1
                rho0[_ind_x.min():_ind_x.max()+1, _ind_y.min():_ind_y.max()+1] = self._rho0
                # F = fftshift(fft2(rho0))
                F = fftshift(self.__fft(rho0))[_ind_qx.min():_ind_qx.max()+1,
                                               _ind_qx.min():_ind_qx.max()+1]
                self._F += F
                self.rho0 = np.abs(rho0)

    def calc_count(self):
        """
            MSFTでの計算回数を取得する。
        """
        return self.__calc_count

    def MakeMaskRegion(self, crop=False):
        """
            kmaxの範囲内の円形マスクを生成。
            crop: kmaxで切った配列を返すかどうか
        """
        if crop is True:
            _qrr = self.mesh_anglefreq(self.kmax, self.kmax, True)
        else:
            _qrr = self.mesh_anglefreq(absolute=True)
        return _qrr <= self.kmax

    def qindex(self):
        """
            もとのイメージ配列内で、実際に和をとる領域のインデックスを返す。
            計算時にはkmaxの範囲内で和をとるようにしており、その領域が元の配列のどの場所に位置するかを与えるものである。
        """
        _qx, _qy, _qz = self.range_anglefreq()
        _ind_qx = np.where(abs(_qx) <= self.kmax)[0]
        _ind_qy = np.where(abs(_qy) <= self.kmax)[0]
        _ind_qz = np.where(abs(_qz) <= self.kmax)[0]
        return [_ind_qx.min(), _ind_qx.max()+1], [_ind_qy.min(), _ind_qy.max()+1], [_ind_qz.min(), _ind_qz.max()+1]

    def SetRho(self, rho0):
        self._rho0 = rho0

    def SetF(self, F):
        self._F = F

    def SetRhoF(self, rho0, F):
        self._rho0 = rho0
        self._F = F

    def GetRho(self, full=False):
        if full is False: return 1.*self._rho0
        else: return 1.*self.rho0

    def GetF(self, shift=True):
        if shift is True: return 1.*self._F
        else: return 1.*fftshift(self._F)

    def GetRhoF(self, shift=True):
        return self._rho0, self.GetF(shift)

    def PlotRhoF(self, **kwargs):
        plt.figure(100, figsize=(12, 5), dpi=100)
        plt.subplot(1,2,1)
        self.PlotRho(**kwargs)
        plt.subplot(1,2,2)
        self.PlotF(**kwargs)

    def PlotRhoI(self, **kwargs):
        plt.figure(100, figsize=(12, 5), dpi=100)
        plt.subplot(1,2,1)
        self.PlotRho(**kwargs)
        plt.subplot(1,2,2)
        self.PlotI(**kwargs)

    def PlotRho(self, **kwargs):
        self.InitAxes(**kwargs)
        rangex = self._rmax[0,:]*self._xscale
        rangey = self._rmax[1,:]*self._yscale
        buff = np.abs(self._rho0)
        plt.imshow(buff, origin="normal", extent=[rangex[0], rangex[1], rangey[0], rangey[1]])
        if self._xlim is not None: plt.xlim(self._xlim[0], self._xlim[1])
        if self._ylim is not None: plt.ylim(self._ylim[0], self._ylim[1])
        plt.clim(0, np.max(buff)*self._rcscale)
        plt.xlabel('x [nm]', fontsize=15)
        plt.ylabel('y [nm]', fontsize=15)
        plt.title('Projection of density map', fontsize=15)

    def PlotF(self, **kwargs):
        self.InitAxes(**kwargs)
        rangeqx = self._qmax[0,:]*self._qxscale
        rangeqy = self._qmax[1,:]*self._qyscale

        plt.imshow(np.abs(self._F), origin="normal",
                   extent=[rangeqx[0], rangeqx[1], rangeqy[0], rangeqy[1]])
        if self._qxlim is not None: plt.xlim(self._qxlim[0], self._qxlim[1])
        if self._qylim is not None: plt.ylim(self._qylim[0], self._qylim[1])
        plt.clim(0, np.max(np.abs(self._F))*self._qcscale)
        plt.xlabel('wave number [1/nm]', fontsize=15)
        plt.ylabel('wave number [1/nm]', fontsize=15)
        plt.title('FFT image by MSFT', fontsize=15)

    def PlotI(self, **kwargs):
        self.InitAxes(**kwargs)
        rangeqx = self._qmax[0,:]*self._qxscale
        rangeqy = self._qmax[1,:]*self._qyscale

        plt.imshow(np.abs(self._F)**2, aspect="auto", origin="lower",
                   extent=[rangeqx[0], rangeqx[1], rangeqy[0], rangeqy[1]])
        if self._qxlim is not None: plt.xlim(self._qxlim[0], self._qxlim[1])
        if self._qylim is not None: plt.ylim(self._qylim[0], self._qylim[1])
        plt.clim(0, np.max(np.abs(self._F)**2)*self._qcscale)
        plt.xlabel('wave number [1/nm]', fontsize=15)
        plt.ylabel('wave number [1/nm]', fontsize=15)
        plt.title('FFT image by MSFT', fontsize=15)
