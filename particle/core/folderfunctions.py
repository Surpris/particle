# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:27:35 2015

@author: Surpris
"""

import os, shutil, re

def ordinal(n):
    suffix = ['th', 'st', 'nd', 'th']
    if type(n) is not int:
        raise TypeError('"n" must be integer')
    if n < 0:
        raise ValueError('"n" must be positive.')
    n1 = n%100
    if n1 <= 3:
        return '{0}{1}'.format(n, suffix[n1])
    elif 4 <= n1 <= 20:
        return '{0}{1}'.format(n, suffix[0])
    else:
        n2 = n1%10
        if n2 <= 3:
            return '{0}{1}'.format(n, suffix[n2])
        else:
            return '{0}{1}'.format(n, suffix[0])

def addends(str0, adstr):
    if str0.endswith(adstr) is True:
        return str0
    else:
        return str0 + adstr

def makefolders(fldrpath, childpath=""):
    """
        Make all folders in "fldrpath" if they don't exist.
        "childpath" is not needed in the first step.
    """
    pp = r'^\w+:/$'
    if childpath is "":
        if not fldrpath.endswith('/'):
            fldrpath += '/'
        ind = fldrpath.index('/')
        _fldrpath = fldrpath[:ind + 1]
        _childpath = fldrpath[ind + 1:]
    else:
        ind = childpath.index('/')
        _fldrpath = fldrpath + childpath[:ind + 1]
        _childpath = childpath[ind + 1:]

    if re.match(pp, _fldrpath):
        if not os.path.exists(_fldrpath):
            raise Exception("Root folder \"" + _fldrpath + "\" is not found.")
        makefolders(_fldrpath, _childpath)
    else:
        if not os.path.exists(_fldrpath):
            os.mkdir(_fldrpath)
##            else:
##                print(_fldrpath + " exists.")
        if _childpath is "":
            return
        makefolders(_fldrpath, _childpath)

def makefilelist(fldrpath, subdir=False, includes="", rel_or_abs="abs"):
    """
        Make a list of all the file paths in "fldrpath" and in its subfolders.
        Only file paths including "includes" are listed.
    """
    filelist = []
    if rel_or_abs == "abs":
        for root, dirs, files in os.walk(fldrpath):
            for filename in files:
                #if filename.endswith(includes) is True:
                if filename.find(includes) > -1:
                    fpath = os.path.join(root, filename)
                    filelist.append(fpath.replace('\\', '/'))
    else:
        if rel_or_abs == "rel":
            for root, dirs, files in os.walk(fldrpath):
                for filename in files:
                    #if filename.endswith(includes) is True:
                    if filename.find(includes) > -1:
                        fpath = os.path.join(root, filename)
                        fpath = fpath.replace(fldrpath, "")
                        filelist.append(fpath.replace('\\', '/'))
        else:
            raise('The key "rel_or_abs" must be "rel" or "abs".')
    return filelist

def missinglist(srcfldr, dstfldr, includes="", rel_or_abs="rel"):
    """
        Make a list of files which are in srcfldr but not in dstfldr.
    """
    srcflist = makefilelist(srcfldr, includes, rel_or_abs)
    srcflist.sort()
    dstflist = makefilelist(dstfldr, includes, rel_or_abs)
    dstflist.sort()
    common = list(set(srcflist) & set(dstflist))
    updated = list(set(dstflist) ^ set(common))
    updated.sort()
    return updated

def copyupdates(srcfldr, dstfldr, includes="", rel_or_abs="rel"):
    """
        Copy files made newly in srcfldr to dstfldr.
    """
    updated = missinglist(srcfldr, dstfldr, includes, rel_or_abs)
    if updated == []:
        return
    else:
        if rel_or_abs == "rel":
            for fpath in updated:
                srcfpath = addends(srcfldr, '/') + fpath
                dstfpath = addends(dstfldr, '/') + fpath
                try:
                    shutil.copyfile(srcfpath, dstfpath)
                except:
                    makefolders(dstfpath.rstrip(fpath.split('/')[-1]))
                    shutil.copyfile(srcfpath, dstfpath)
        else:
            for fpath in updated:
                dstfpath = addends(dstfldr, '/') + fpath.split('/')[-1]
                shutil.copyfile(fpath, dstfpath)

def copyupdateswithlist(flist, dstfldr):
    for fpath in flist:
        dstfpath = addends(dstfldr, '/') + fpath.split('/')[-1]
        shutil.copyfile(fpath, dstfpath)

def movefiles(srcfldr, dstfldr, includes="", rel_or_abs="rel"):
    """
        Move files in srcfldr to dstfldr without overwriting.
    """
    updated = missinglist(srcfldr, dstfldr, includes, rel_or_abs)
    if updated == []:
        return
    else:
        for fpath in updated:
            dstfpath = addends(dstfldr, '/') + fpath.split('/')[-1]
            print("Moved file: " + fpath.split('/')[-1])
            shutil.move(fpath, dstfpath)

def movefilesoverwrite(srcfldr, dstfldr, includes="", rel_or_abs="abs"):
    """
        Move files in srcfldr to dstfldr without overwriting.
    """
    srcflist = makefilelist(srcfldr, includes, rel_or_abs)
    srcflist.sort()
    if srcflist == []:
        return
    else:
        for fpath in srcflist:
            dstfpath = addends(dstfldr, '/') + fpath.split('/')[-1]
            shutil.move(fpath, dstfpath)

def movefileswithlist(flist, dstfldr):
    errpath = []
    for fpath in flist:
        dstfpath = addends(dstfldr, '/') + fpath.split('/')[-1]
        try:
            shutil.move(fpath, dstfpath)
        except:
            errpath.append(fpath)
    if len(errpath) !=0:
        print('Error path:')
        print(errpath)

def file2list(fpath, delimiter=None):
    f = open(fpath, "r").read()
    lis = []
    for line in f.split('\n'):
        if delimiter is not None:
            items = np.zeros()
            for item in line.split(delimiter):
                items.append()
        else:
            lis.append(line)
    try:
        lis.remove('')
    except:
        lis = lis
    return lis

def list2file(fpath, lis, permission="w"):
    if lis != []:
        f = open(fpath, permission)
        for line in lis:
            f.writelines(line + '\n')
        f.close()
