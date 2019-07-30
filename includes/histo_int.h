/** @file histograms_int.h
 * 
 * @brief Fast calculation of 1D and 2D hisograms from signed and unsigned integer inputs
 *
 * @par Reulet Lab.       
 *
 */ 

#ifndef HISTOGRAMS_INT_H
#define HISTOGRAMS_INT_H

void swap_histogram(uint64_t *hist, const int b);
void histogram8_unsigned(uint8_t *data, uint64_t size, uint64_t *hist);
void histogram8_signed(int8_t *data, uint64_t size, uint64_t *hist);
void histogram16_unsigned(uint16_t *data, uint64_t size, uint64_t *hist, const int b);
void histogram16_signed(int16_t *data, uint64_t size, uint64_t *hist, const int b);
void swap_histogram2d(uint64_t *hist, const int b);
void histogram2d8_unsigned(uint8_t *data1, uint8_t *data2, uint64_t size, uint64_t *hist);
void histogram2d8_signed(int8_t *data1, int8_t *data2, uint64_t size, uint64_t *hist);
void histogram2d16_unsigned(uint16_t *data1, uint16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic);
void histogram2d16_signed(int16_t *data1, int16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic);


#endif /* HISTOGRAMS_INT_H */






