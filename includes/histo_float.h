/** @file histograms_float.h
 * 
 * @brief Fast calculation of 1D and 2D hisograms from signed and unsigned floats inputs (single and double precision)
 *	Histogramme1D_Float V0.1
 *		- Assumes that bin are symetrically distributed around zero.
 *		- The number of bin is always even (2**n) and n is assumed to be in [8 to 16]
 *		- Makes a small error on binning related to a few float * and +
 *		- Bin defined are as folow : [-max to -max+L_bin[; [-max +L_bin to -max+2*L_bin[ ... ; [max - L_bin to max[
 * @par Reulet Lab.       
 *
 */ 

#ifndef HISTOGRAMS_FLOAT_H
#define HISTOGRAMS_FLOAT_H

void histogram_single(const float *data,  const uint64_t size, uint64_t *hist, const uint8_t n, const float max);
void histogram_double( double *data,  uint64_t size, uint64_t *hist, uint8_t n, double max);

// histogram2D tends to be limited by the amount of Cache memory available. 
// Therefore it can be more efficient to turn of hyperthreadind by using one of OpenMP's environment variable
// (set OMP_NUM_THREADS=<number of threads to use>). Setting the number of thread to a value lower than the number of
// core can also lead to faster histogram sorting. 
// Also the option variable toggles different implementations of the algorithm which can be faster depending on your workload and machine. 
void histogram2D_single(const float *data1, const float *data2,  const uint64_t size, uint64_t *hist, const uint8_t n, const float max, int atomic);
void histogram2D_double(const double *data1, const double *data2,  const uint64_t size, uint64_t *hist, const uint8_t n, const double max);

#endif /* HISTOGRAMS_FLOAT_H */
