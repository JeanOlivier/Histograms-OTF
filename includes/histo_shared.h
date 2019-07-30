/** @file histograms_shared.h
 * 
 * @brief Contains shared function of the histograms library. 
 *
 * @par Reulet Lab.       
 *
 */ 

#ifndef HISTOGRAMS_SHARED_H
#define HISTOGRAMS_SHARED_H

#if defined(__CYGWIN__) || defined(__MINGW64__)
    // see number from: sdkddkver.h
    // https://docs.microsoft.com/fr-fr/windows/desktop/WinProg/using-the-windows-headers
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <omp.h>
#include "mpfr.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <math.h>

void manage_thread_affinity();
void reduce_uint64(uint64_t** arrs, uint64_t bins, uint64_t begin, uint64_t end);
void reduce_uint8(uint8_t** arrs, uint64_t bins, uint64_t begin, uint64_t end);

#endif /* HISTOGRAMS_SHARED_H */
