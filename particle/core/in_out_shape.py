# -*- coding: utf-8 -*-

import numpy as np

def isInsideSphereIndex(coor, r):
    """
        半径`r`の球内にある座標のインデックスを返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r: radius of sphere

        <Output>
            The indices of coordinates inside the sphere with the radius of `r`
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return buff[0]**2 + buff[1]**2 + buff[2]**2 <= r**2


def isInsideSphere(coor, r):
    """
        半径`r`の球内にある座標を返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r: radius of sphere

        <Output>
            The coordinates inside the sphere with the radius of `r`
    """

    index = isInsideSphereIndex(coor, r)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()


def isOutsideSphereIndex(coor, r):
    """
        半径`r`の球外にある座標のインデックスを返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r: radius of sphere

        <Output>
            The indices of coordinates outside the sphere with the radius of `r`
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return buff[0]**2 + buff[1]**2 + buff[2]**2 >= r**2


def isOutsideSphere(coor, r):
    """
        半径`r`の球外にある座標を返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r: radius of sphere

        <Output>
            The coordinates outside the sphere with the radius of `r`
    """

    index = isOutsideSphereIndex(coor, r)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

def isInsideSphereShellIndex(coor, r_inner, r_outer):
    """
        `R-dr~R`の球殻内にある座標のインデックスを返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r_inner: inner radius of spherical shell
            r_outer: outer radius of spherical shell

        <Output>
            The indices of coordinates inside the spherical shell defined by (R-dr, R]
    """
    ind_inside = isInsideSphereIndex(coor, r_outer)
    ind_outside = isOutsideSphereIndex(coor, r_inner)
    return ind_inside & ind_outside


def isInsideSphereShell(coor, r_inner, r_outer):
    """
        `R-dr~R`の球殻内にある座標を返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            r_inner: inner radius of spherical shell
            r_outer: outer radius of spherical shell

        <Output>
            The coordinates outside the sphere with the radius of `r`
    """

    index = isInsideSphereShellIndex(coor, r_inner, r_outer)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()

def isInsideCubeIndex(coor, a):
    """
        一辺が`a`の立方体内にある座標のインデックスを返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            a: edge length of cube

        <Output>
            The indices of coordinates inside the cube with the edge length of `a`
    """
    if coor.shape[1] == 3:
        buff = coor.transpose()
    else:
        buff = coor.copy()
    return (np.abs(buff[0]) <= 0.5*a) & (np.abs(buff[1]) <= 0.5*a) & (np.abs(buff[2]) <= 0.5*a)


def isInsideCube(coor, a):
    """
        一辺が`a`の立方体内にある座標を返す

        <Input>
            coor: coordinates (N*3 or 3*N shape)
            a: edge length of cube

        <Output>
            The coordinates inside the cube with the edge length of `a`
    """

    index = isInsideCubeIndex(coor, a)
    if coor.shape[1] == 3:
        return coor[index].copy()
    else:
        return coor[:, index].copy()
