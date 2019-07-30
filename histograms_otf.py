#!/bin/python
# -*- coding: utf-8 -*-

import ctypes
import sys, os
import platform
from numpy.ctypeslib import ndpointer
from numpy import zeros, fromstring, arange, log2
from numpy import int8, uint8, int16, uint16, single, double
from ctypes import c_uint8, c_int8, c_uint16, c_int16, c_float, c_double, c_int, c_uint16, c_uint32, c_uint64, c_uint
import operator as op
from functools import reduce

# Debuging
from IPython import get_ipython
ipython = get_ipython()


libpath = os.path.abspath(os.path.dirname(__file__))
if not libpath in os.environ['PATH']:
    os.environ['PATH'] = libpath+os.path.pathsep+os.environ['PATH']

plat_info = dict(plat=platform.system())
if plat_info['plat'] == 'Windows':
    plat_info['lib'] = os.path.join(libpath, 'histograms.dll')
    plat_info['com'] = 'make histograms.dll'
    # Adding cygwin libs path for windows
    libspath = 'C:\\cygwin64\\usr\\x86_64-w64-mingw32\\sys-root\\mingw\\bin'
    if libspath not in os.environ['PATH']:
        os.environ['PATH'] = libspath+os.path.pathsep+os.environ['PATH']  
else:
    plat_info['lib'] = os.path.join(libpath, 'histograms.so')
    plat_info['com'] = 'make histograms.so'


if not os.path.isfile(plat_info['lib']):
    raise IOError("{lib} is missing. To compile on {plat}:\n{com}\n".format(**plat_info))

lib = ctypes.cdll[plat_info['lib']]

# OpenMP stuff
if plat_info["plat"] == "Windows":
    omp = ctypes.CDLL('libgomp-1')
else:
    try:
        omp = ctypes.CDLL("/usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so")
    except:
        omp = ctypes.CDLL("libgomp.so")
set_num_threads = omp.omp_set_num_threads
set_num_threads.argtypes=(c_int,)
set_num_threads.restype=None
get_num_threads = omp.omp_get_max_threads
get_num_threads.restype=c_int
get_num_threads.argtypes=None


def hist1dNbits(x, n=8, ihist=None, max = 1):
    """
    *ihist* is for using a previously filled histogram and keep filling it.
    """
    float32 = True if (x.dtype in [single]) else False
    float64 = True if (x.dtype in [double]) else False
    signed = True if (x.dtype in [int8, int16]) else False
    container8 = x.dtype in [int8, uint8]
    assert n in range(8,16+1), "Supported bit depths are from 8 to 16"

    k = 2**n 
    #fct = lib['histogram{:d}'.format(n)]
    #if n==8:
    if float32|float64:
        fct = lib['histogram_single' if float32 else 'histogram_double']
        fct.argtypes = (
            ndpointer(
                dtype=c_float if float32 else c_double,
                shape=(len(x),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,)),
            c_uint8,
            c_float if float32 else c_double
        )
    elif container8:
        fct = lib['histogram8_signed' if signed else 'histogram8_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=c_int8 if signed else c_uint8,
                shape=(len(x),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,))
        )
    else:
        fct = lib['histogram16_signed' if signed else 'histogram16_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=c_int16 if signed else c_uint16,
                shape=(len(x),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,)),
            c_int
        )
        
    if ihist is None:
        hist = zeros(k, dtype=c_uint64)
    else:
        assert ihist.size == k, "*ihist* has wrong size"
        if signed:
            swap_histogram = lib['swap_histogram'] 
            swap_histogram.argtypes = (ndpointer(dtype=c_uint64, shape=(k,)), c_int)
            swap_histogram(ihist, 8 if container8 else n)
        hist = ihist

    if float32|float64:
        fct(x, len(x), hist, n, max)
    elif container8 :
        fct(x, len(x), hist)
    else: 
        fct(x, len(x), hist, n)

    return hist


def hist2dNbits(x, y, n=8, force_n=False, option="", ihist=None, max=1):
    """
    No atomic chosen heuristically as a sweet spot for:
        - somewhat correlated data
        - 10 bit histogram
    Full atomic performance is highly dependent on data and bit depth.
    
    It requires testing but it can drastically improve performance for large
    bitdepth histograms and/or uncorrelated data.
    """
    if not force_n: # To avoid filling the ram instantly
        assert 8<=n<=12, "8<=n<=12 is required, set kwarg *force_n* to True to override"
    assert len(x)==len(y), "len(x)==len(y) is required"
    assert x.dtype == y.dtype, "x and y should be of the same type"
    float32 = True if (x.dtype in [single]) else False
    float64 = True if (x.dtype in [double]) else False
    signed = True if (x.dtype in [int8, int16]) else False
    container8 = x.dtype in [int8, uint8]
    
    if container8|signed :
        if (option == "atomic"):
            OPT = 1 ;
        else:
            OPT = 0 ;
    elif (option == "serial_uint8") :
        OPT = 1 ;
    elif (option == "atomic"):
        OPT = 2 ;
    elif (option == "par_red"):
        OPT = 3 ;
    elif (option == "atomic_uint8"):
        OPT = 4 ;
    elif (option == "par_rred"):
        OPT = 5 ;
    elif (option == "serial_contiguous"):
        OPT = 6;
    else :
        OPT = 0 ;
        
    k = 2**n 
      
    if float32|float64:
        fct = lib['histogram2D_single' if float32 else 'histogram2D_double']
        fct.argtypes = (
            ndpointer(
                dtype=c_float if float32 else c_double,
                shape=(len(x),)
            ),
            ndpointer(
                dtype=c_float if float32 else c_double,
                shape=(len(y),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,k)),
            c_uint8,
            c_float if float32 else c_double,
            c_uint
        )
    elif container8:
        assert n==8, "Only 8bit histograms are supported for 8bit containers"
        fct = lib['histogram2d8_signed' if signed else 'histogram2d8_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=c_int8 if signed else c_uint8,
                shape=(len(x),)
            ),
            ndpointer(
                dtype=c_int8 if signed else c_uint8,
                shape=(len(y),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,k))
        )
    else:
        fct = lib['histogram2d16_signed' if signed else 'histogram2d16_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=c_int16 if signed else c_uint16,
                shape=(len(x),)
            ),
            ndpointer(
                dtype=c_int16 if signed else c_uint16,
                shape=(len(y),)
            ),
            c_uint64,
            ndpointer(dtype=c_uint64, shape=(k,k)),
            c_uint64,
            c_uint
        )
  
    if ihist is None:
        hist = zeros((k,k), dtype=c_uint64)
    else:
        assert ihist.size == k**2, "*ihist* has wrong size"
        if signed:
            swap_histogram2d = lib['swap_histogram2d'] 
            swap_histogram2d.argtypes = (ndpointer(dtype=c_uint64, shape=(k,k)), c_int)
            swap_histogram2d(ihist, 8 if container8 else n)
        hist = ihist
        
    if float32|float64:
        fct(x, y, len(x), hist, n, max, OPT)
    elif container8 :
        fct(x, y, len(x), hist)
    else: 
        fct(x, y, len(x), hist, n, OPT)
    
    return hist


# Extra stuff: Computing moments and cumulants on histograms

#Python implementation
# n choose r: n!/(r!(n-r)!)
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

# kth moment of h, centered by default
def moment_py(h, k, centered=True):
    if centered:
        bshift = double(moment_py(h, 1, centered=False))
    else: 
        bshift = 0
    b = double(arange(h.size))-bshift
    n = double(h.sum())
    return (h*b**k).sum()/n

def cumulant_py(h, k, centered=True):
    hh = double(h)
    ret = moment_py(hh,k,False)
    ret -= sum([ncr(k-1,m-1)*cumulant_py(hh,m)*moment_py(hh,k-m,False) for m in range(1,k)])
    return ret

# C implementation
def moment(h, k, centered=True):
    b = int(log2(len(h))) # Assumes h is a b-bit histogram
    c = int(centered)
    fct = lib['moment']
    fct.argtypes = (
            ndpointer(
                dtype=c_uint64,
                shape=(len(h),)
            ),
            c_int,
            c_int,
            c_int
        )
    fct.restype = c_double
    return fct(h, b, k, c)

def cumulant(h, k):
    b = int(log2(len(h))) # Assumes h is a b-bit histogram
    fct = lib['cumulant']
    fct.argtypes = (
            ndpointer(
                dtype=c_uint64,
                shape=(len(h),)
            ),
            c_int,
            c_int
        )
    fct.restype = c_double
    return fct(h, b, k)
