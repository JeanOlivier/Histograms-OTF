#!/bin/python
# -*- coding: utf-8 -*-

import ctypes
import sys, os
import platform
from numpy.ctypeslib import ndpointer
from numpy import zeros, fromstring
from numpy import int8, uint8, int16, uint16
from ctypes import c_uint8, c_int8, c_uint16, c_int16, c_double, c_int, c_uint64

libpath = os.path.abspath(os.path.dirname(__file__))
if not libpath in os.environ['PATH']:
    os.environ['PATH'] = libpath+os.environ['PATH']

plat_info = dict(plat=platform.system())
if plat_info['plat'] == 'Windows':
    plat_info['lib'] = os.path.join(libpath, 'histograms.dll')
    plat_info['com'] = 'make histograms.dll'
    # Adding cygwin libs path for windows
    libspath = r'C:\cygwin64\usr\x86_64-w64-mingw32\sys-root\mingw\bin\;'
    if libpath not in os.environ['PATH']:
        os.environ['PATH'] = libspath+os.environ['PATH']  
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


def hist1dNbits(x, n=8):
    """
    """
    signed = True if (x.dtype in [int8, int16]) else False
    container8 = x.dtype in [int8, uint8]
    assert n in range(8,16+1), "Supported bit depths are from 8 to 16"

    k = 2**n 
    #fct = lib['histogram{:d}'.format(n)]
    #if n==8:
    if container8:
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

    hist = zeros(k, dtype=c_uint64)
    fct(x, len(x), hist) if container8 else fct(x, len(x), hist, n)
    
    return hist


def hist2dNbits(x, y, n=8, force_n=False, atomic=False):
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
    signed = True if (x.dtype in [int8, int16]) else False
    container8 = x.dtype in [int8, uint8]
    a = 1 if atomic else 0
    
    k = 2**n 

    if container8:
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
        
    hist = zeros((k,k), dtype=c_uint64)
    fct(x, y, len(x), hist) if container8 else fct(x, y, len(x), hist, n, a)
    
    return hist
