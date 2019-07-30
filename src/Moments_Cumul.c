#include "../includes/histo_shared.h"
// Simple, but could be faster
int64_t nCk(int n, int k)
{
    if (k==0){
        return 1;
    }
    return (n*nCk(n-1, k-1))/k; // Product form, division always yields an integer
}

double moment(uint64_t *hist, const int b, const int k, const int centered)
{
    const int size = 1<<b;
    long double bshift=0;
    long double val = 0;
    uint64_t n=0;
    
    if (centered){
        bshift = moment(hist, b, 1, 0);
    }
    #pragma omp parallel
    {
        manage_thread_affinity(); // For 64+ logical cores on Windows
        if (centered){
            #pragma omp for reduction(+:val), reduction(+:n)
            for (int i=0; i<size; i++){
                val += (long double)hist[i] * powl((long double)i - (long double)bshift, k);
                n += hist[i];
            }
        }
        else{
            #pragma omp for reduction(+:val), reduction(+:n)
            for (int i=0; i<size; i++){
                val += (long double)hist[i] * powl((long double)i, k);
                n += hist[i];
            }
        }
    }
    return (double)(val/(long double)n);
}

double cumulant(uint64_t *hist, const int b, const int k){
    double ret = moment(hist, b, k, 0);
    for (int i=1; i<k; i++){
        ret -= (double)nCk(k-1, i-1)*cumulant(hist, b, i)*moment(hist, b, k-i, 0);
    }
    return ret;
}