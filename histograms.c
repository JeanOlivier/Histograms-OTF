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


// Windows doesn't really like systems with over 64 logical cores.
// This function assign the thread it's called from to a core, bypassing the 
// default assignation. It alternates between CPU Groups to assign a thread to
// each physical core first; then it can make use of HTT.
//
// This could be much more sophisticated, but it works well for dual identical
// cpu systems with HTT on and over 64 logical cores.
void manage_thread_affinity()
{
    #ifdef _WIN32_WINNT
        int nbgroups = GetActiveProcessorGroupCount();
        int *threads_per_groups = (int *) malloc(nbgroups*sizeof(int));
        for (int i=0; i<nbgroups; i++)
        {
            threads_per_groups[i] = GetActiveProcessorCount(i);
        }

        // Fetching thread number and assigning it to cores
        int tid = omp_get_thread_num(); // Internal omp thread number (0 -- OMP_NUM_THREADS)
        HANDLE thandle = GetCurrentThread();
        _Bool result;
        
        int set_group = tid%nbgroups; // We change group for each thread
        int nbthreads = threads_per_groups[set_group]; // Nb of threads in group for affinity mask.
        GROUP_AFFINITY group = {((uint64_t)1<<nbthreads)-1, set_group}; // nbcores amount of 1 in binary
                
        result = SetThreadGroupAffinity(thandle, &group, NULL); // Actually setting the affinity
        if(!result) fprintf(stderr, "Failed setting output for tid=%i\n", tid);
        free(threads_per_groups);
    #else
        //We let openmp and the OS manage the threads themselves
    #endif
}

// To store an i-bits value in a j-bits integer, with j being a power of 2,
// you need at least j = 2**( log(i)/log(2) + (1-(log(i)/log(2)%1)/1)%1 )


// An histogram done on int casted as uint will be swapped
// This swaps it back, b is the bitlength of the histogram
void swap_histogram(uint64_t *hist, const int b)
{
    const int halfsize = 1<<(b-1);
    uint64_t *tmp = calloc(halfsize, sizeof(uint64_t));
    int i=0;
    for (; i<halfsize; i++){  // Paralelizing those small loops is detrimental
        tmp[i] = hist[i];
        hist[i] = hist[i+halfsize];
    }
    for (; i<2*halfsize; i++){
        hist[i] = tmp[i-halfsize];
    }
    free(tmp);
}


// Computes the histogram for 8-bit samples in uint8 containers
void histogram8_unsigned(uint8_t *data, uint64_t size, uint64_t *hist)
{   
    uint64_t *data_64 = (uint64_t *) data;
    #pragma omp parallel
    {
        manage_thread_affinity(); // For 64+ logical cores on Windows
        uint64_t tmp=0;
        #pragma omp for reduction(+:hist[:1<<8])
        for (uint64_t i=0; i<size/8; i++){
            tmp = data_64[i];
            hist[tmp >>  0 & 0xFF]++;
            hist[tmp >>  8 & 0xFF]++;
            hist[tmp >> 16 & 0xFF]++;
            hist[tmp >> 24 & 0xFF]++;
            hist[tmp >> 32 & 0xFF]++;
            hist[tmp >> 40 & 0xFF]++;
            hist[tmp >> 48 & 0xFF]++;
            hist[tmp >> 56 & 0xFF]++;
        }
    }
    // The data that doesn't fit in 64bit chunks, openmp would be overkill here.
    for (uint64_t i=size-(size%8); i<size; i++){
        hist[ data[i] ]++; 
    }
}


void histogram8_signed(int8_t *data, uint64_t size, uint64_t *hist)
{
    uint8_t *data_unsigned = (uint8_t *) data;
    histogram8_unsigned(data_unsigned, size, hist);
    swap_histogram(hist, 8); // b is always 8 here
}


// Computes the histogram for (8<b<=16)-bit samples in uint16 containers
void histogram16_unsigned(uint16_t *data, uint64_t size, uint64_t *hist, const int b)
{   
    const int32_t tail = 16-b;
    uint64_t *data_64 = (uint64_t *) data;
    #pragma omp parallel
    {
        manage_thread_affinity(); // For 64+ logical cores on Windows
        uint64_t tmp=0;
        #pragma omp for reduction(+:hist[:1<<b])
        for (uint64_t i=0; i<size/4; i++){
            tmp = data_64[i]; // tail get rid of bits > b
            hist[ (tmp >>   0 & 0xFFFF) >> tail ]++;
            hist[ (tmp >>  16 & 0xFFFF) >> tail ]++;
            hist[ (tmp >>  32 & 0xFFFF) >> tail ]++;
            hist[ (tmp >>  48 & 0xFFFF) >> tail ]++;
        }
    }
    // The data that doesn't fit in 64bit chunks, openmp would be overkill here.
    for (uint64_t i=size-(size%4); i<size; i++){
        hist[ data[i] >> tail ]++; 
    }
}


void histogram16_signed(int16_t *data, uint64_t size, uint64_t *hist, const int b)
{
    uint16_t *data_unsigned = (uint16_t *) data;
    histogram16_unsigned(data_unsigned, size, hist, b);
    swap_histogram(hist, b);
}


// #Python POC implementation of the 2d swap, simple but not optimal: 
//  
// def swap(x):
//     tmp = copy(x[:len(x)/2])
//     x[:len(x)/2] = copy(x[len(x)/2:])
//     x[len(x)/2:] = copy(tmp)
// 
// def swap2d(x):
//     xx = x.flatten()
//     swap(xx)
//     l = int(sqrt(len(xx)))
//     #print len(xx), l
//     for i in range(l):
//         swap(xx[i*l:(i+1)*l])
//     return xx.reshape(x.shape)
//
// A 2d histogram done on int casted as uint will be swapped on its two axis
// This swaps it back, b is the bitlength of the histogram
void swap_histogram2d(uint64_t *hist, const int b)
{
    uint64_t rsize = 1<<b;     // Number AND Size of rows (because it's a square)
    swap_histogram(hist, 2*b); // Vertical swap
    #pragma omp parallel
    {
        manage_thread_affinity();
        #pragma omp for
        for (uint64_t i=0; i<rsize; i++){ // For each row
            swap_histogram(hist+(i*rsize), b); // Horizontal swap of each row
        }
    }
}


// Computes the 2d histogram for 8-bit samples in uint8 containers
//
// The 2d histogram is represented by a single dimension array, logically
// seperated in 256 blocks corresponding to the data1 stream, with in-block
// indices corresponding to the data2 stream.
// It appears as a 2d array in the python wrapper.
void histogram2d8_unsigned(uint8_t *data1, uint8_t *data2, uint64_t size, uint64_t *hist)
{
    uint64_t *data1_64 = (uint64_t *) data1;
    uint64_t *data2_64 = (uint64_t *) data2;
    #pragma omp parallel
    {
        manage_thread_affinity(); // For 64+ logical cores on Windows
        uint64_t tmp1=0;
        uint64_t tmp2=0;
        #pragma omp for reduction(+:hist[:1<<(8*2)])
        for (uint64_t i=0; i<size/8; i++){
            tmp1 = data1_64[i]; 
            tmp2 = data2_64[i]; 
            hist[ (tmp1 <<  8 & 0xFF00) + (tmp2 >>  0 & 0xFF) ]++;
            hist[ (tmp1 >>  0 & 0xFF00) + (tmp2 >>  8 & 0xFF) ]++;
            hist[ (tmp1 >>  8 & 0xFF00) + (tmp2 >> 16 & 0xFF) ]++;
            hist[ (tmp1 >> 16 & 0xFF00) + (tmp2 >> 24 & 0xFF) ]++;
            hist[ (tmp1 >> 24 & 0xFF00) + (tmp2 >> 32 & 0xFF) ]++;
            hist[ (tmp1 >> 32 & 0xFF00) + (tmp2 >> 40 & 0xFF) ]++;
            hist[ (tmp1 >> 40 & 0xFF00) + (tmp2 >> 48 & 0xFF) ]++;
            hist[ (tmp1 >> 48 & 0xFF00) + (tmp2 >> 56 & 0xFF) ]++;
        }
    }
    // The data that doesn't fit in 64bit chunks, openmp would be overkill here.
    for (uint64_t i=size-(size%8); i<size; i++){
        hist[ (data1[i]<<8) + data2[i] ]++;
    }
}


void histogram2d8_signed(uint8_t *data1, uint8_t *data2, uint64_t size, uint64_t *hist)
{
    uint8_t *data1_unsigned = (uint8_t *) data1;
    uint8_t *data2_unsigned = (uint8_t *) data2;
    histogram2d8_unsigned(data1_unsigned, data2_unsigned, size, hist);
    swap_histogram2d(hist, 8); // b is always 8 here
}


void reduce(uint64_t** arrs, uint64_t bins, uint64_t begin, uint64_t end)
{
    assert(begin < end);
    if (end - begin == 1) {
        return;
    }
    uint64_t pivot = (begin + end) / 2;
    /* Moving the termination condition here will avoid very short tasks,
     * but make the code less nice. */
    //#pragma omp task
    reduce(arrs, bins, begin, pivot);
    //#pragma omp task
    reduce(arrs, bins, pivot, end);
    //#pragma omp taskwait
    /* now begin and pivot contain the partial sums. */
    #pragma omp parallel 
    {
        manage_thread_affinity();
        #pragma omp for
        for (uint64_t i = 0; i < bins; i++)
            arrs[begin][i] += arrs[pivot][i];
    }
}

// Computes the 2d histogram for (8<b<=16)-bit samples in uint16 containers
//
// The 2d histogram is represented by a single dimension array, logically
// seperated in 2**b blocks corresponding to the data1 stream, with in-block
// indices corresponding to the data2 stream.
// It appears as a 2d array in the python wrapper.
// 
// atomic: 0 = no atomic; 1 = full atomic 
// Best value depends on data. 
//  - Full atomic is better for random data and large b
//  - no atomic is better for correlated data
void histogram2d16_unsigned(uint16_t *data1, uint16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic)
{
    // Precomputing the correct mask and shift values. Helps readability, doesn't  really help performance. 
    const int32_t tail0 = 16-b;
    const int32_t tail1 = tail0+16;
    const int32_t tail2 = tail1+16;
    const int32_t tail3 = tail2+16;
    const int32_t mask = (1<<b)-1; // Right amount of 0b1

    uint64_t **hs;
    int n;

    uint64_t *data1_64 = (uint64_t *) data1;
    uint64_t *data2_64 = (uint64_t *) data2;
    #pragma omp parallel
    {
        manage_thread_affinity(); // For 64+ logical cores on Windows
        n = omp_get_num_threads();
        
        uint64_t tmp1=0;
        uint64_t tmp2=0;
        // Full atomic should be faster when there's low memory collision, e.g. random data or large *b*.
        // Strickingly, for a given set of data it's typically faster for large *b*.
        if (atomic == 1){
            // No local histogram; no reduction!
            #pragma omp for //reduction(+:h[:1<<(b*2)])  
            for (uint64_t i=0; i<size/4; i++){
                tmp1 = data1_64[i]; 
                tmp2 = data2_64[i]; 
                #pragma omp atomic update
                hist[ ((tmp1 >> tail0 & mask) << b) + (tmp2 >> tail0 & mask) ]++;   
                #pragma omp atomic update
                hist[ ((tmp1 >> tail1 & mask) << b) + (tmp2 >> tail1 & mask) ]++;
                #pragma omp atomic update
                hist[ ((tmp1 >> tail2 & mask) << b) + (tmp2 >> tail2 & mask) ]++;
                #pragma omp atomic update
                hist[ ((tmp1 >> tail3 & mask) << b) + (tmp2 >> tail3 & mask) ]++;
            }
        }
        // Using local histograms that have to be reduced afterwards 
        else{
            #pragma omp single
            hs = (uint64_t **) malloc(n * sizeof(uint64_t));
            uint64_t *h = (uint64_t *) calloc(1<<(b*2), sizeof(uint64_t)); // Filled with 0s.
            hs[omp_get_thread_num()] = h;
            
            #pragma omp for nowait
            for (uint64_t i=0; i<size/4; i++){
                tmp1 = data1_64[i]; 
                tmp2 = data2_64[i]; 
                h[ ((tmp1 >> tail0 & mask) << b) + (tmp2 >> tail0 & mask) ]++;   
                h[ ((tmp1 >> tail1 & mask) << b) + (tmp2 >> tail1 & mask) ]++;
                h[ ((tmp1 >> tail2 & mask) << b) + (tmp2 >> tail2 & mask) ]++;
                h[ ((tmp1 >> tail3 & mask) << b) + (tmp2 >> tail3 & mask) ]++;
            }
            // TODO: An option to choose between *reduce* and *critical*
            //#pragma omp critical
            //for (uint64_t j=0; j<(1<<(b*2));j++){
            //    hist[j] += h[j];
            //}
            //free(h);
        }
    }
    if (atomic == 0){
        reduce(hs, 1<<(b*2), 0, n);
        #pragma omp parallel
        {
            manage_thread_affinity();
            #pragma omp for
            for (uint64_t i=0; i<1<<(b*2); i++){
                hist[i]=hs[0][i];
            }
        }
        for (int i=0; i<n; i++){
            free(hs[i]);
        }
        free(hs);
    }

    // The data that doesn't fit in 64bit chunks, openmp would be overkill here.
    for (uint64_t i=size-(size%4); i<size; i++){
        hist[ ((data1[i]>>tail0)<<b) + (data2[i]>>tail0) ]++;
    }
}


void histogram2d16_signed(int16_t *data1, int16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic)
{
    uint16_t *data1_unsigned = (uint16_t *) data1;
    uint16_t *data2_unsigned = (uint16_t *) data2;
    histogram2d16_unsigned(data1_unsigned, data2_unsigned, size, hist, b, atomic);
    swap_histogram2d(hist, b);
}


