# -*- coding: utf-8 -*-

import numpy as np

def make_lattice_points(a=1.0, lattice_type="sc", ind_min=-10, ind_max=10, CAR=None):
    """make_lattice_points(a, lattice_type, ind_min, ind_max, CAR) -> numpy.2darray
    calculate coordinates of a lattice points
    
    Parameters
    ----------
    a : float
        lattice constant
    lattice_type : str
        type of lattice
    ind_min : int
        min of Miller's index
    ind_max : int
        max of Miller's index
    CAR : float
        ratio of c-axis base length to a-axis base length for hcp

    Returns
    -------
    coordinates of lattice points
    """

    # generate basis
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
            CAR = 2.0 * np.sqrt(2.0 / 3.0) # default c-to-a ratio = 2.0 * np.sqrt(2.0 / 3.0)
        a1 = np.array([1.0, 0., 0.])
        a2 = np.array([-0.5, 0.5*np.sqrt(3.0), 0.])
        a3 = np.array([0.0, 0.0, CAR])
    else:
        raise ValueError("Invalid lattice_type: ", lattice_type)

    # generate indices
    ind_list = np.arange(ind_min, ind_max + 1, 1)
    h, k, l = np.meshgrid(ind_list, ind_list, ind_list)

    h = np.reshape(h, (h.size, 1))
    k = np.reshape(k, (k.size, 1))
    l = np.reshape(l, (l.size, 1))

    hkl = np.hstack((np.hstack((h, k)),l))

    # calculate coordinates
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

def PlaneNormalVector(h, k, l):
    """PlaneNormalVector(h, k, l) -> numpy.1darray
    return the normal vector of the plane identified by the Miller indices

    Parameters
    ----------
    h, k, l : int
        the Miller indices
    
    Returns
    -------
    the normal vector : numpy.1darray
    """
    vec = np.array([h,k,l])
    return vec/np.linalg.norm(vec)

def MillerNormalVectors_100():
    """MillerNormalVectors_100() -> numpy.1darray
    return the normal vector of the (100) plane
    """
    k_100 = np.array([[1,0,0], [0,1,0], [0,0,1]])
    return np.array([k_100[0,:],k_100[1,:],k_100[2,:],
                     -k_100[0,:],-k_100[1,:],-k_100[2,:]])

def MillerNormalVectors_110():
    """MillerNormalVectors_110() -> numpy.1darray
    return the normal vector of the (110) plane
    """
    k_110 = np.array([[1,1,0], [0,1,1], [1,0,1],
                      [1,-1,0], [0,1,-1],[-1,0,1]])/np.sqrt(2.0)
    return np.array([k_110[0,:],k_110[1,:],k_110[2,:],
                     k_110[3,:],k_110[4,:],k_110[5,:],
                     -k_110[0,:],-k_110[1,:],-k_110[2,:],
                     -k_110[3,:],-k_110[4,:],-k_110[5,:]])

def MillerNormalVectors_111():
    """MillerNormalVectors_111() -> numpy.1darray
    return the normal vector of the (111) plane
    """
    k_111 = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])/np.sqrt(3.0)
    return np.array([k_111[0,:],k_111[1,:],k_111[2,:],k_111[3,:],
                     -k_111[0,:],-k_111[1,:],-k_111[2,:],-k_111[3,:]])

def EulerRotation(Coor, EulerAngle=[0,0,0], mode=None):
    """EulerRotation(Coor, EulerAngle=[0,0,0], mode=None) -> numpy.2darray
    return the rotated coordinates

    Parameters
    ----------
    Coor        : numpy.2darray
    EulerAngle= : 3-element list or numpy.1darray (default : [0,0,0])
    mode        : int (default : None)
        0 : normal transform
        1 : inverse transform.

    Returns
    -------
    the rotated coordinates (numpy.2darray)
    """
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
