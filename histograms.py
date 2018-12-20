#!/bin/python
# -*- coding: utf-8 -*-

import ctypes
import os
import platform
from numpy.ctypeslib import ndpointer
from numpy import zeros, fromstring

plat_info = dict(plat=platform.system())
if plat_info['plat'] == 'Windows':
    plat_info['lib'] = './histograms.dll'
    plat_info['com'] = 'make histograms.dll'
else:
    plat_info['lib'] = './histograms.so'
    plat_info['com'] = 'make histograms.so'


if not os.path.isfile(plat_info['lib']):
    raise IOError("{lib} is missing. To compile on {plat}:\n{com}\n".format(**plat_info))

lib = ctypes.cdll[plat_info['lib']]


def hist1dNbits(x, n=8):
    """
    """
    signed = True if (x.dtype in [int8, int16]) else False
    cont8 = x.dtype in [int8, uint8]
    assert n in range(8,16+1), "Supported bit depths are from 8 to 16"

    k = 2**n 
    #fct = lib['histogram{:d}'.format(n)]
    #if n==8:
    if cont8:
        fct = lib['histogram8_signed' if signed else 'histogram8_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=ctypes.c_int8 if signed else ctypes.c_uint8,
                shape=(len(x),)
            ),
            ctypes.c_uint64,
            ndpointer(dtype=ctypes.c_uint64, shape=(k,))
        )
    else:
        fct = lib['histogram16_signed' if signed else 'histogram16_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=ctypes.c_int16 if signed else ctypes.c_uint16,
                shape=(len(x),)
            ),
            ctypes.c_uint64,
            ndpointer(dtype=ctypes.c_uint64, shape=(k,)),
            ctypes.c_int32
        )

    hist = zeros(k, dtype=ctypes.c_uint64)
    fct(x, len(x), hist) if cont8 else fct(x, len(x), hist, n)
    
    return hist


def hist2dNbits(x, y, n=8, force_n=False):
    """
    """
    if not force_n:
        assert 8<=n<=12, "8<=n<=12 is required, set kwarg *force_n* to True to override"
    assert len(x)==len(y), "len(x)==len(y) is required"
    assert x.dtype == y.dtype, "x and y should be of the same type"
    signed = True if (x.dtype in [int8, int16]) else False
    cont8 = x.dtype in [int8, uint8]

    k = 2**n 

    if cont8:
        assert n==8, "Only 8bit histograms are supported for 8bit containers"
        fct = lib['histogram2d8_signed' if signed else 'histogram2d8_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=ctypes.c_int8 if signed else ctypes.c_uint8,
                shape=(len(x),)
            ),
            ndpointer(
                dtype=ctypes.c_int8 if signed else ctypes.c_uint8,
                shape=(len(y),)
            ),
            ctypes.c_uint64,
            ndpointer(dtype=ctypes.c_uint64, shape=(k,k))
        )
    else:
        fct = lib['histogram2d16_signed' if signed else 'histogram2d16_unsigned']
        fct.argtypes = (
            ndpointer(
                dtype=ctypes.c_int16 if signed else ctypes.c_uint16,
                shape=(len(x),)
            ),
            ndpointer(
                dtype=ctypes.c_int16 if signed else ctypes.c_uint16,
                shape=(len(y),)
            ),
            ctypes.c_uint64,
            ndpointer(dtype=ctypes.c_uint64, shape=(k,k)),
            ctypes.c_uint64
        )
        
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist) if cont8 else fct(x, y, len(x), hist, n)
    
    return hist


# Following function are old, they're kept for documentation.

def histogram2d8(x, y):#, n=8):
    """
    """
    n=8
    #assert n in range(8,16+1), "Supported bit depths are from 8 to 16"
    assert len(x)==len(y), "*x* and *y* should have the same length"

    k = 2**n 
    fct = lib['histogram2d8_signed']
    fct.argtypes = (
        ndpointer(dtype=ctypes.c_int8, shape=(len(x),)),
        ndpointer(dtype=ctypes.c_int8, shape=(len(y),)),
        ctypes.c_uint64,
        ndpointer(dtype=ctypes.c_uint64, shape=(k,k))
    )
    
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist) if n==8 else fct(x, len(x), hist, n)
    
    return hist


def histogram2d16(x, y, n=10):#, n=8):
    """
    """
    #assert n in range(8,16+1), "Supported bit depths are from 8 to 16"
    assert len(x)==len(y) and n<=10, "len(x)==len(y) and n<=10 are required"

    k = 2**n 
    fct = lib['histogram2d16_signed']
    fct.argtypes = (
        ndpointer(dtype=ctypes.c_int16, shape=(len(x),)),
        ndpointer(dtype=ctypes.c_int16, shape=(len(y),)),
        ctypes.c_uint64,
        ndpointer(dtype=ctypes.c_uint64, shape=(k,k)),
        ctypes.c_uint64
    )
    
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist, n)
    
    return hist

def histogram2d16_old(x, y, n=10):#, n=8):
    """
    """
    #assert n in range(8,16+1), "Supported bit depths are from 8 to 16"
    assert len(x)==len(y) and n<=10, "len(x)==len(y) and n<=10 are required"

    k = 2**n 
    fct = lib['histogram2d16_old']
    fct.argtypes = (
        ndpointer(dtype=ctypes.c_uint16, shape=(len(x),)),
        ndpointer(dtype=ctypes.c_uint16, shape=(len(y),)),
        ctypes.c_uint64,
        ndpointer(dtype=ctypes.c_uint64, shape=(k,k)),
        ctypes.c_uint64
    )
    
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist, n)
    
    return hist

def histogram2d16_bak(x, y, n=10):#, n=8):
    """
    """
    #assert n in range(8,16+1), "Supported bit depths are from 8 to 16"
    assert len(x)==len(y) and n<=10, "len(x)==len(y) and n<=10 are required"

    k = 2**n 
    fct = lib['histogram2d16_bak']
    fct.argtypes = (
        ndpointer(dtype=ctypes.c_uint16, shape=(len(x),)),
        ndpointer(dtype=ctypes.c_uint16, shape=(len(y),)),
        ctypes.c_uint64,
        ndpointer(dtype=ctypes.c_uint64, shape=(k,k)),
        ctypes.c_uint64
    )
    
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist, n)
    
    return hist

def histogram2d8_bak(x, y):#, n=8):
    """
    """
    n=8
    #assert n in range(8,16+1), "Supported bit depths are from 8 to 16"
    assert len(x)==len(y), "*x* and *y* should have the same length"

    k = 2**n 
    fct = lib['histogram2d8_bak']
    fct.argtypes = (
        ndpointer(dtype=ctypes.c_uint8, shape=(len(x),)),
        ndpointer(dtype=ctypes.c_uint8, shape=(len(y),)),
        ctypes.c_uint64,
        ndpointer(dtype=ctypes.c_uint64, shape=(k,k))
    )
    
    hist = zeros((k,k), dtype=ctypes.c_uint64)
    fct(x, y, len(x), hist) if n==8 else fct(x, len(x), hist, n)
    
    return hist

