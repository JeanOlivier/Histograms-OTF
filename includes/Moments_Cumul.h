/** @file Moments_Cumulants.h
 * 
 * @brief Calculation of moments and cumulants from 1D and 2D histrograms
 *	
 * @par Reulet Lab.       
 *
 */ 

#ifndef MOMENTS_CUMULANTS_H
#define MOMENTS_CUMULANTS_H

// N chose K
int64_t nCk(int n, int k);

double moment(uint64_t *hist, const int b, const int k, const int centered);

double cumulant(uint64_t *hist, const int b, const int k);

#endif /* MOMENTS_CUMULANTS_H */