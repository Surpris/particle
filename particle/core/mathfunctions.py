# -*- coding: utf-8 -*-

import sys
import numpy as np
from numpy.fft import fft, fft2, ifft, ifft2, fftshift
import scipy.special as spec

def make_lattice_points(a=1.0, lattice_type="sc", ind_min=-10, ind_max=10, CAR=None):
    """
        格子点の座標を与える
        <Input>
            a: lattice constant
            lattice_type: type of lattice
            ind_min: Min of Miller's index
            ind_max: Max of Miller's index
            car: C-to-A ratio for hcp

        <Output>
            coordinates of lattice points
    """

    # 単位格子ベクトルの生成
    if lattice_type == "sc":
        a1 = np.array([1., 0., 0.])
        a2 = np.array([0., 1., 0.])
        a3 = np.array([0., 0., 1.])
    elif lattice_type == "fcc":
        a1 = np.array([0.5, 0.5, 0.])
        a2 = np.array([0., 0.5, 0.5])
        a3 = np.array([0.5, 0., 0.5])
    elif lattice_type == "bcc":
        a1 = np.array([0.5, 0.5, -0.5])
        a2 = np.array([-0.5, 0.5, 0.5])
        a3 = np.array([0.5, -0.5, 0.5])
    elif lattice_type == "hcp":
        if CAR is None:
            CAR = 2.0 * np.sqrt(2.0 / 3.0) # c-to-a ratio = 2.0 * np.sqrt(2.0 / 3.0)
        a1 = np.array([1.0, 0., 0.])
        a2 = np.array([-0.5, 0.5*np.sqrt(3.0), 0.])
        a3 = np.array([0.0, 0.0, CAR])
    else:
        raise ValueError("Invalid lattice_type: ", lattice_type)

    # 係数列の生成
    ind_list = np.arange(ind_min, ind_max + 1, 1)
    h, k, l = np.meshgrid(ind_list, ind_list, ind_list)

    h = np.reshape(h, (h.size, 1))
    k = np.reshape(k, (k.size, 1))
    l = np.reshape(l, (l.size, 1))

    hkl = np.hstack((np.hstack((h, k)),l))

    # 格子の各点の座標の生成
    if lattice_type != "hcp":
        A = np.vstack((np.vstack((a1, a2)), a3)) # lattice
        return a * np.dot(hkl, A)
    else:
        A = np.vstack((np.vstack((a1, a2)), a3)) # lattice
        A_coor = a * np.dot(hkl, A)
        B_coor = np.zeros((2*len(A_coor), 3))
        B_coor[::2] = A_coor
        B = 2./3.*a1 +  1./3.*a2 + 0.5*a3 # the other atom in the basis
        B_coor[1::2] = A_coor + a * np.tile(B[None, :], (len(A_coor), 1))
        return B_coor.copy()

def elapsed(elapse):
    print('Elapsed time: {0:.2f} sec.'.format(elapse))

def PlaneNormalVector(h, k, l):
    vec = np.array([h,k,l])
    return vec/np.linalg.norm(vec)

def MillerNormalVectors_100():
    k_100 = np.array([[1,0,0], [0,1,0], [0,0,1]])
    return np.array([k_100[0,:],k_100[1,:],k_100[2,:],
                     -k_100[0,:],-k_100[1,:],-k_100[2,:]])

def MillerNormalVectors_110():
    k_110 = np.array([[1,1,0], [0,1,1], [1,0,1],
                      [1,-1,0], [0,1,-1],[-1,0,1]])/np.sqrt(2.0)
    return np.array([k_110[0,:],k_110[1,:],k_110[2,:],
                     k_110[3,:],k_110[4,:],k_110[5,:],
                     -k_110[0,:],-k_110[1,:],-k_110[2,:],
                     -k_110[3,:],-k_110[4,:],-k_110[5,:]])

def MillerNormalVectors_111():
    k_111 = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])/np.sqrt(3.0)
    return np.array([k_111[0,:],k_111[1,:],k_111[2,:],k_111[3,:],
                     -k_111[0,:],-k_111[1,:],-k_111[2,:],-k_111[3,:]])

def EulerRotation(Coor, EulerAngle=[0,0,0], mode=None):
    # Check the validity of coordinate.
    try: h = Coor.shape[0]
    except: h = 1
    try: w = Coor.shape[1]
    except: w = 1
    if w != 3 and h != 3:
        raise Exception('Coordinate must be 3-dimensional.')

    l = len(EulerAngle)
    if l is not 3:
        raise Exception('Coordinate must be 3-dimensional.')
    if EulerAngle[0] == 0 and EulerAngle[1] == 0 and EulerAngle[2] == 0:
        return Coor


    # Substitute Euler angles.
    alpha = EulerAngle[0]*np.pi/180
    beta = EulerAngle[1]*np.pi/180
    gamma = EulerAngle[2]*np.pi/180

    # mode 0 : normal transform, 1 : inverse transform.
    if mode is None:
        # First rotation with respect to z0 axis.
        Euler_z0_ori = np.array([[np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        # Second rotation with respect to x1 axis.
        Euler_x1_ori = np.array([[1, 0, 0],
            [0, np.cos(beta), np.sin(beta)],
            [0, -np.sin(beta), np.cos(beta)]])
        # Third rotation with respect to z2 axis.
        Euler_z2_ori = np.array([[np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z2_ori,np.dot(Euler_x1_ori,Euler_z0_ori))
    else:
        Euler_z0_inv = np.array([[np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        Euler_x1_inv = np.array([[1, 0, 0],
            [0, np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta), np.cos(beta)]])
        Euler_z2_inv = np.array([[np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z0_inv,np.dot(Euler_x1_inv,Euler_z2_inv))

    # Reshaping for output : (N, 3) np.array.
    if h == 3:
        out = np.dot(Euler_all,Coor)
    else:
        out = np.dot(Euler_all,np.transpose(Coor))
        out = np.transpose(out)
    return out

def EulerRotation_ZYZ(Coor, EulerAngle=[0,0,0], mode=None):
    # Check the validity of coordinate.
    try: h = Coor.shape[0]
    except: h = 1
    try: w = Coor.shape[1]
    except: w = 1
    if w != 3 and h != 3:
        raise Exception('Coordinate must be 3-dimensional.')

    l = len(EulerAngle)
    if l is not 3:
        raise Exception('Coordinate must be 3-dimensional.')
    if EulerAngle[0] == 0 and EulerAngle[1] == 0 and EulerAngle[2] == 0:
        return Coor


    # Substitute Euler angles.
    alpha = EulerAngle[0]*np.pi/180
    beta = EulerAngle[1]*np.pi/180
    gamma = EulerAngle[2]*np.pi/180

    # mode 0 : normal transform, 1 : inverse transform.
    if mode is None:
        # First rotation with respect to z0 axis.
        Euler_z0_ori = np.array([[np.cos(alpha), np.sin(alpha), 0],
            [-np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        # Second rotation with respect to y1 axis.
        Euler_y1_inv = np.array([[np.cos(beta), 0, -np.sin(beta)],
            [0, 1, 0],
            [np.sin(beta), 0, np.cos(beta)]])
        # Third rotation with respect to z2 axis.
        Euler_z2_ori = np.array([[np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z2_ori,np.dot(Euler_y1_ori,Euler_z0_ori))
    else:
        Euler_z0_inv = np.array([[np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])
        Euler_y1_inv = np.array([[np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]])
        Euler_z2_inv = np.array([[np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        Euler_all = np.dot(Euler_z0_inv,np.dot(Euler_y1_inv,Euler_z2_inv))

    # Reshaping for output : (N, 3) np.array.
    if h == 3:
        out = np.dot(Euler_all,Coor)
    else:
        out = np.dot(Euler_all,np.transpose(Coor))
        out = np.transpose(out)
    return out

def local_minima(mat):
    """
        Return local minima of the given np.array.
        minima_ls: list of (row, column, value).
    """
    minima_ls = []
    # 行列の一番外側の行、列は調べない。
    # そのため、rangeが1, len-2となっている。
    for i in range(1, len(mat) - 2):
        for j in range(1, len(mat[i]) - 2):
            if mat[i, j] < mat[i - 1, j] and mat[i, j] < mat[i + 1, j] and mat[i, j] < mat[i, j - 1] and mat[i, j] < mat[i, j + 1]:
                minima_ls.append((i, j, mat[i, j]))
    return np.array(minima_ls)

def peakdet(v, delta, x=None):
    """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html
        Returns two np.arrays
        function [maxtab, mintab]=peakdet(v, delta, x)
        % PEAKDET Detect peaks in a vector
        % [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        % maxima and minima ("peaks") in the vector V.
        % MAXTAB and MINTAB consists of two columns. Column 1
        % contains indices in V, and column 2 the found values.
        %
        % With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        % in MAXTAB and MINTAB are replaced with the corresponding
        % X-values.
        %
        % A point is considered a maximum peak if it has the maximal
        % value, and was preceded (to the left) by a value lower by
        % DELTA.
        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
        v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

def peakdet_saxs(x, y, qrange=None, xdims=0, nums=None, is_smooth=True, window_len=10):
    if len(x) != len(y):
        sys.exit('Input vectors y and x must have same length')

    if qrange is not None:
        ind_x = np.where((x >= min(qrange))&(x <= max(qrange)))[0]
    else:
        ind_x = np.ones(len(x), dtype=bool)
    if is_smooth is True:
        smoothy = smooth(y,window_len=window_len,window='hanning')
    else:
        smoothy = y
    x4y = (smoothy*x**xdims)[ind_x]
    x4y /= max(x4y)

    delta0 = 1.

    i = 0
    if nums is None:
        nums = 0
    elif type(nums) is not int or nums < 0:
        raise ValueError("'nums' must be a positive integer")

    while i < 20:
        maxtab, mintab = peakdet(x4y, delta0)
        if maxtab.shape[0] <= nums:
            i += 1
            delta0 /=2.
        else:
            break

    ind_max = []
    ind_min = []
    for ii in range(maxtab.shape[0]):
        ind_max.append(int(maxtab[ii,0]))
    for ii in range(mintab.shape[0]):
        ind_min.append(int(mintab[ii,0]))

    minout = np.zeros((len(ind_min), 2), dtype=float)
    minout[:,0] = (x[ind_x])[ind_min]
    minout[:,1] = (y[ind_x])[ind_min]

    maxout = np.zeros((len(ind_max),2), dtype=float)
    maxout[:,0] = (x[ind_x])[ind_max]
    maxout[:,1] = (y[ind_x])[ind_max]

    return maxout, minout


def smooth(y,window_len=60,window='hanning'):
    x = np.array(y)
    if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
            w=np.ones(window_len,'d')
    else:
            w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def ccf(y, mode='full'):
    y_acf = np.correlate(y, y, mode=mode)
    if mode == 'full':
        y_acf = y_acf[y_acf.size/2:]
    y_acf /= len(y_acf)
    y2 = np.mean(y)**2
    return y_acf/y2 - 1.

def accf(data, window=True, mask=None):
    # out = accf(data, mask)
    # return cross correlation such as angular cross correlation.
    # The formula is given in P. Wochner et al., PNAS 106, 11511 (2009):
    # C(q, shift) = <I(q, phi)*I(q, phi+shift)>_{\phi}/<I(q,phi)>_{\phi}^2 - 1.
    # This function returns C(shit) from 1D array (, or data at given q).
    #--- input
    # data: data array (1D array)
    # mask: masking array (1D array)
    #--- output
    # out: cross correlation (1D array)

    # check validity of input data & masking array.
    try:
        (h, w) = data.shape
        raise ValueError('data must have the size of 1*N.')
    except:
        w = data.size
        if window is True:
            buff = smooth(data, window_len=10, window='hanning')
        else:
            buff = data
    try:
        (h, w) = mask.shape
        raise ValueError('mask must have the size of 1*N.')
    except:
        if mask is None:
            mask = np.ones(len(data), dtype=float)
        elif len(mask) != w:
            raise ValueError('data and mask must have the same size.')

    # prepare the shift matrices.
    data_shift = np.zeros((w, w))
    mask_shift = np.zeros((w, w))
    for ii in range(w):
        data_shift[ii,:] = np.roll(buff, ii)*buff
        mask_shift[ii,:] = np.roll(mask, ii)*mask
    data_shift = data_shift*mask_shift

    # calculate the square of mean of data: <I(q,phi)>_{\phi}^2.
    I_mean = buff*mask
    I_mean = sum(I_mean)/sum(mask)

    # calculate average of correlation: <I(q, phi)*I(q, phi+shift)>_{\phi}.
    data_shift = (np.sum(data_shift, axis=1)/np.sum(mask_shift, axis=1)).transpose()
    return data_shift/I_mean**2 - 1.0

def legendre_coef(y, leg_n):
    """
        Return coefficients of Fourier-Legendre expansion with the maximum order of 'leg_n'.
    """
    if len(y.shape) != 1:
        raise ValueError('y must be an 1-D array or a list.')

    dx = 2./len(y)
    x = np.arange(-1, 1, dx)
    legendre = []
    for xx in x:
        legendre.append(spec.lpn(leg_n, xx)[0])
    legendre = np.array(legendre)

    # Calculate the coefficients of Legendre polynomials
    coef_leg = np.zeros(leg_n+1)
    for nn in range(leg_n+1):
        coef_leg[nn] = 0.5*(2.*nn+1)*dx*np.sum(y*legendre[:, nn])
    return coef_leg

def rotsym_coef(z, multi=1, mode='same'):
    """
        Return x-axis values and amplitudes of each rotation axis(?).
    """
    types = ['same', 'zero']
    if len(z.shape) != 1:
        raise ValueError('y must be an 1-D array or a list.')
    if type(multi) is not int:
        raise ValueError('multi must be integer.')
    if multi <= 0:
        raise ValueError('multi must be positive.')
    if mode not in types:
        raise ValueError('mode must be "{0}" or "{1}".'.format(types[0], types[1]))

    if multi == 1:
        buff= z
    else:
        nums = len(z)
        buff = np.zeros(nums*multi, dtype=float)
        if mode == 'same':
            for mu in range(multi):
                buff[mu:mu+nums] = z
        elif mode == 'zero':
            buff[0:nums] = z

    b = fft(buff)/(nums*multi)
    fs = np.arange(0, 1., 1./(360*multi))
    rot =1./(fs+pow(2, -52))
    ind_rot = np.where(rot <= 360.)[0]
    rot = rot[ind_rot]
    b = b[ind_rot]
    return rot, np.abs(b)

def polarmap(data, x=None, y=None, delta=None, r_points=None, theta_points=None, order=1):
    """
        polar mapを(x, y)から中心を探索して計算。
        返り値は
            polar_map
            r_polar
            theta_polar
    """
    if x is None and y is None:
        ny, nx = data.shape
        ind_zero = [nx//2, ny//2]
    elif y is None:
        ind_zero = [np.where(np.abs(x)<1e-10)[0][0],
                    np.where(np.abs(x)<1e-10)[0][0]]
    else:
        ind_zero = [np.where(np.abs(x)<1e-10)[0][0],
                    np.where(np.abs(y)<1e-10)[0][0]]
    if delta is not None:
        ind_zero = [ind_zero[0] - delta[0], ind_zero[1] - delta[1]]
    return polfun.reproject_image_into_polar(data, ind_zero, r_points=r_points, theta_points=theta_points, order=order)

def integral(poldata, mask=None, epsilon=0):
    """
        polar mapからradial profileを計算。
        返り値は
            radial_profile
    """
    if mask is not None:
        data = poldata*mask
    else:
        data = poldata
    data_int = []
    wid = poldata.shape[0]
    for row in data:
        ind_nonzero = np.where(np.abs(row) >= epsilon)[0]
        nonzero = len(ind_nonzero)
        if nonzero == 0:
            data_int.append(0)
        else:
            data_int.append(sum(row[ind_nonzero])/len(row))
    return np.array(data_int)
