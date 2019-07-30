#include "../includes/histo_shared.h"
// Histogramme1D_Float V0.1
// 		- Assumes that bin are symetrically distributed around zero.
// 		- The number of bin is always even (2**n) and n is assumed to be in [8 to 16]
// 		- Makes a small error on binning related to a few float * and +
// 		- Bin defined are as folow : [-max to -max+L_bin[; [-max +L_bin to -max+2*L_bin[ ... ; [max - L_bin to max[

///// fonctions static inline
static inline void to_hist_float(const float* p_tmp1,const float p_L_bin,const float p_max,const uint8_t n ,uint64_t* hist){ 	
	
	const float tmp1 = *p_tmp1;
	const float max = p_max;
	
	if (tmp1 >= max){
		// clipping
		hist[(1<<n)-1]++; // add one to last bin 
	}
	else if (tmp1 < -(max)){
		// clipping
		hist[0]++; // add one to first bin
	}
	else{
	uint16_t tmp2 = (uint16_t)((tmp1+max)/(p_L_bin));
	hist[ tmp2 ]++;
	}	
}	

static inline void to_hist_double(const double* p_tmp1,const double p_L_bin,const double p_max,const uint8_t n,uint64_t* hist){ 	
	
	const double  tmp1 = *p_tmp1;
	const double max = p_max;
	
	if (tmp1 >= max){
		// clipping
		hist[(1<<n)-1]++; // add one to last bin 
	}
	else if (tmp1 < -(max)){
		// clipping
		hist[0]++; // add one to first bin	
	}
	else{
	uint16_t tmp2 = (uint16_t)((tmp1+max)/(p_L_bin));
	hist[ tmp2 ]++;
	}
}	
static inline void to_hist2D_uint8(uint8_t* h, uint64_t* hist,uint32_t BIN){
	if (h[BIN]==255){
		hist[BIN]+=(1<<8);
	}
	h[BIN]++;
}

static inline void bin_hist2D_float(const float* p_data1,const float* p_data2, const float L_bin,const float max,const uint8_t n ,uint64_t* hist, uint16_t* binx, uint16_t* biny){ 	
	const float data1 = *p_data1;
	const float data2 = *p_data2;
	
	if (data1 >= max){
		// clipping
		*binx = (1<<n)-1;	
	}
	else if (data1 < -(max)){
		// clipping
		// add one to first bin
		*binx = 0;
	}
	else{
	*binx = (uint16_t)((data1+max)/(L_bin));
	}
	
	if (data2 >= max){
		// clipping
		*biny = (1<<n)-1;	
	}
	else if (data2 < -(max)){
		// clipping
		// add one to first bin
		*biny = 0;
	}
	else{
	*biny = (uint16_t)((data2+max)/(L_bin));
	}
	
}

static inline void bin_hist2D_double(const double* p_data1,const double* p_data2, const double L_bin,const double max,const uint8_t n ,uint64_t* hist, uint16_t* binx, uint16_t* biny){ 	
	const float data1 = *p_data1;
	const float data2 = *p_data2;
	
	if (data1 >= max){
		// clipping
		*binx = (1<<n)-1;	
	}
	else if (data1 < -(max)){
		// clipping
		// add one to first bin
		*binx = 0;
	}
	else{
	*binx = (uint16_t)((data1+max)/(L_bin));
	}
	
	if (data2 >= max){
		// clipping
		*biny = (1<<n)-1;	
	}
	else if (data2 < -(max)){
		// clipping
		// add one to first bin
		*biny = 0;
	}
	else{
	*biny = (uint16_t)((data2+max)/(L_bin));
	}
	
}
///////

void histogram_single(const float *data,  const uint64_t size, uint64_t *hist, const uint8_t n, const float max)
{   
	const uint16_t N_bin = 1<<n ;
	const float L_bin = 2*max/N_bin;
	uint64_t *p_data_64 = (uint64_t *) data;
	
    #pragma omp parallel
    {
		manage_thread_affinity(); // For 64+ logical cores on Windows
		//uint64_t tmp=0;
		//uint64_t tp =0;
		#pragma omp for reduction(+:hist[:N_bin]) 
		for (uint64_t i=0; i<size/2; i++){
			to_hist_float((float*)(p_data_64 + i) + 0, L_bin, max, n, hist);
			to_hist_float((float*)(p_data_64 + i) + 1, L_bin, max, n, hist);
		}
    }
	// The data that doesn't fit in 64bit chunks, openmp would be overkill here.
	for (uint64_t i= 2*(size/2); i< 2*(size/2) + size%2; i++){
		to_hist_float( (data+i), L_bin, max, n, hist );	
		}
}

void histogram_double( double *data,  uint64_t size, uint64_t *hist, uint8_t n, double max)
{   
	 uint16_t N_bin = 1<<n ;
	 double L_bin = 2*max/N_bin;
	
    #pragma omp parallel
    {
		manage_thread_affinity(); // For 64+ logical cores on Windows
		#pragma omp for reduction(+:hist[:N_bin]) 
		for (uint64_t i=0; i<size; i++){
			to_hist_double( (data+i), L_bin, max,n, hist );
		}
    }
}

// histogram2D tends to be limited by the amount of Cache memory available. 
// Therefore it can be more efficient to turn of hyperthreadind by using one of OpenMP's environment variable
// (set OMP_NUM_THREADS=<number of threads to use>). Setting the number of thread to a value lower than the number of
// core can also lead to faster histogram sorting. 
void histogram2D_single(const float *data1, const float *data2,  const uint64_t size, uint64_t *hist, const uint8_t n, const float max, int option)
{	
	const uint32_t N_bin = 1<<n ;
	const float L_bin = 2*max/N_bin;
	const uint64_t *p_data1_64 = (uint64_t *) data1;
	const uint64_t *p_data2_64 = (uint64_t *) data2;
	
	//option = 6; //overwrites option for developpement only
	//printf("option =%d\n",option);
	
	// Serial_uint8 stores the histogram into a list of uint8 called h and adds h to hist only when one of its bins is going 
	// to be incremented above 2**8-1 or when the calculation is done.
	if (option == 1){
		uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
		uint16_t binx = 0;
		uint16_t biny = 0;
		uint32_t BIN;
			for (uint64_t i=0; i<size/2; i++){
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, hist, BIN );
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, hist, BIN );
			}
			// additionner le reste dans hist
			for(uint64_t i=0; i<(1<<(2*n)); i++){
				hist[i]+=h[i];
			}
		free(h);
	}
	// Atomic makes the histogram in parralel but uses only one instance on hist in memory. 
	// It's slowed when two threads try to read and write in the same location in memory.
	// Therefoe it's faster when the histogram contains more bins (less colisions).
	// For some applications it could be faster to use a larger number of bin, then to concatenate neighbor
	// bins afterward (which is very low cost).
	else if (option == 2){
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			//printf("# %d ; &binx = %p ; &biny = %p \n",omp_get_thread_num(), (void*)(&binx), (void*)(&biny));
			#pragma omp for
			for (uint64_t i=0; i<size/2; i++){
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				#pragma omp atomic update
				hist[BIN]++;
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				#pragma omp atomic update
				hist[BIN]++;
			}
		}
	}
	// Parralel_reduce makes a copie of hist called h for each threads  and calculates histograms seperatly for each thread.
	// When the calculation is done is combine the h's into a single histogram using the reduce function.
	else if(option==3){
		uint64_t **hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			n_threads = omp_get_num_threads();

			#pragma omp single // Affects only next line
			hs = (uint64_t **) malloc(n_threads* sizeof(uint64_t));
			////
			uint64_t* h = (uint64_t* ) calloc(1<<(2*n), sizeof(uint64_t));// Filled with 0s.
			hs[omp_get_thread_num()] = h ;
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			#pragma omp for
			for (uint64_t i=0; i<size/2; i++){
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				h[BIN]++;
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				h[BIN]++;
			}
		}
		// Critical reduction was very slow, this is faster.
        reduce_uint64(hs, 1<<(2*n), 0, n_threads); // hs[0] is the reduced array afterwards
        #pragma omp parallel
        {
            manage_thread_affinity();
            // Returning the result to the output array
            #pragma omp for
            for (uint64_t i=0; i<(1<<(2*n)); i++){
                hist[i]+=hs[0][i];
            }
        }
        for (int i=0; i<n_threads; i++){
            free(hs[i]);
        }
        free(hs);
	}
	// Atomic_uint8_t combines the use of (uint8_t)h and atomic but h is copie and then reduce then added to hist.
	else if (option ==4){
		uint8_t **hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			n_threads = omp_get_num_threads();

			#pragma omp single // Affects only next line
			hs = (uint8_t **) malloc(n_threads* sizeof(uint64_t));
			////
			uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
			hs[omp_get_thread_num()] = h ;
			
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			#pragma omp for
			for (uint64_t i=0; i<size/2; i++){
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				if (h[BIN]==255){
					#pragma omp atomic update
					hist[BIN]+=(1<<8);
				}
				h[BIN]++;
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				if (h[BIN]==255){
					#pragma omp atomic update
					hist[BIN]+=(1<<8);
				}
				h[BIN]++;
			}
			/* for (uint64_t i=0; i<(1<<(2*n)); i++){
				#pragma omp atomic update
                hist[i]+=h[i];
            } */
		}
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			for (uint64_t j=0; j<n_threads; j++)
			{
				#pragma omp for
				for (uint64_t i=0; i<(1<<(2*n)); i++)
				{
					hist[i]+=hs[j][i];
				}
			}
		}
        for (int i=0; i<n_threads; i++){
            free(hs[i]);
        }
        free(hs);
	}
	// Parralel_reduce_reduce
	else if (option ==5){
		uint64_t **Hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			
			#pragma omp single // Affects only next line
			n_threads = omp_get_num_threads();
			#pragma omp single 
			Hs = (uint64_t **) malloc(n_threads* sizeof(uint64_t));
			////
			uint64_t* H = (uint64_t* ) calloc(1<<(2*n), sizeof(uint64_t));// Filled with 0s.
			uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
			Hs[omp_get_thread_num()] = H ;
			
			#pragma omp for
			for (uint64_t i=0; i<size/2; i++){
				uint16_t binx = 0;
				uint16_t biny = 0;
				uint32_t  BIN;
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, H, BIN );
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, H, BIN );
			}
			for(uint64_t i=0; i<(1<<(2*n)); i++){
				H[i]+=h[i];
			}
		}
		// Critical reduction was very slow, this is faster.
        reduce_uint64(Hs, 1<<(2*n), 0, n_threads); // Hs[0] is the reduced array afterwards
		#pragma omp parallel
        {
            manage_thread_affinity();
            // Returning the result to the output array
            #pragma omp for
            for (uint64_t i=0; i<(1<<(2*n)); i++){
                hist[i]+=Hs[0][i];
            }
        }
        for (int i=0; i<n_threads; i++){
            free(Hs[i]);
        }
        free(Hs);
	}
	// Serial is the simplest implementation for the calculation of 2d histograms.
	// Depending on the load it can be faster then the other options.
	else{
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			for (uint64_t i=0; i<size/2; i++){
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 0), ((float*)(p_data2_64 + i) + 0), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n); 
				hist[BIN]++;
				bin_hist2D_float( ((float*)(p_data1_64 + i) + 1), ((float*)(p_data2_64 + i) + 1), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n); 
				hist[BIN]++;
			}
	}
}

void histogram2D_double(const double *data1, const double *data2,  const uint64_t size, uint64_t *hist, const uint8_t n, const double max, int option)
{	
	const uint32_t N_bin = 1<<n ;
	const double L_bin = 2*max/N_bin;
	
	//option = 6; //overwrites option for developpement only
	printf("Double option =%d\n",option);
	
	// Serial_uint8 stores the histogram into a list of uint8 called h and adds h to hist only when one of its bins is going 
	// to be incremented above 2**8-1 or when the calculation is done.
	if (option == 1){
		uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
		uint16_t binx = 0;
		uint16_t biny = 0;
		uint32_t BIN;
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, hist, BIN );
			}
			// additionner le reste dans hist
			for(uint64_t i=0; i<(1<<(2*n)); i++){
				hist[i]+=h[i];
			}
		free(h);
	}
	// Atomic makes the histogram in parralel but uses only one instance on hist in memory. 
	// It's slowed when two threads try to read and write in the same location in memory.
	// Therefoe it's faster when the histogram contains more bins (less colisions).
	// For some applications it could be faster to use a larger number of bin, then to concatenate neighbor
	// bins afterward (which is very low cost).
	else if (option == 2){
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			#pragma omp for
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				#pragma omp atomic update
				hist[BIN]++;
			}
		}
	}
	// Parralel_reduce makes a copie of hist called h for each threads  and calculates histograms seperatly for each thread.
	// When the calculation is done is combine the h's into a single histogram using the reduce function.
	else if(option==3){
		uint64_t **hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			n_threads = omp_get_num_threads();

			#pragma omp single // Affects only next line
			hs = (uint64_t **) malloc(n_threads* sizeof(uint64_t));
			////
			uint64_t* h = (uint64_t* ) calloc(1<<(2*n), sizeof(uint64_t));// Filled with 0s.
			hs[omp_get_thread_num()] = h ;
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			#pragma omp for
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				h[BIN]++;
			}
		}
		// Critical reduction was very slow, this is faster.
        reduce_uint64(hs, 1<<(2*n), 0, n_threads); // hs[0] is the reduced array afterwards
        #pragma omp parallel
        {
            manage_thread_affinity();
            // Returning the result to the output array
            #pragma omp for
            for (uint64_t i=0; i<(1<<(2*n)); i++){
                hist[i]+=hs[0][i];
            }
        }
        for (int i=0; i<n_threads; i++){
            free(hs[i]);
        }
        free(hs);
	}
	// Atomic_uint8_t combines the use of (uint8_t)h and atomic but h is copie and then reduce then added to hist.
	else if (option ==4){
		uint8_t **hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			n_threads = omp_get_num_threads();

			#pragma omp single // Affects only next line
			hs = (uint8_t **) malloc(n_threads* sizeof(uint64_t));
			////
			uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
			hs[omp_get_thread_num()] = h ;
			
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t  BIN;
			#pragma omp for
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				if (h[BIN]==255){
					#pragma omp atomic update
					hist[BIN]+=(1<<8);
				}
				h[BIN]++;
			}
			/* for (uint64_t i=0; i<(1<<(2*n)); i++){
				#pragma omp atomic update
                hist[i]+=h[i];
            } */
		}
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			for (uint64_t j=0; j<n_threads; j++)
			{
				#pragma omp for
				for (uint64_t i=0; i<(1<<(2*n)); i++)
				{
					hist[i]+=hs[j][i];
				}
			}
		}
        for (int i=0; i<n_threads; i++){
            free(hs[i]);
        }
        free(hs);
	}
	// Parralel_reduce_reduce
	else if (option ==5){
		uint64_t **Hs;
		int n_threads;
		#pragma omp parallel
		{
			manage_thread_affinity(); // For 64+ logical cores on Windows
			#pragma omp single // Affects only next line
			n_threads = omp_get_num_threads();
			#pragma omp single 
			Hs = (uint64_t **) malloc(n_threads* sizeof(uint64_t));
			/////
			uint64_t* H = (uint64_t* ) calloc(1<<(2*n), sizeof(uint64_t));// Filled with 0s.
			uint8_t* h = (uint8_t* ) calloc(1<<(2*n), sizeof(uint8_t));// Filled with 0s.
			Hs[omp_get_thread_num()] = H ;
			
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t BIN;
			#pragma omp for
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				to_hist2D_uint8( h, H, BIN );
			}
			for(uint64_t i=0; i<(1<<(2*n)); i++){
				H[i]+=h[i];
			}	
		}
		// Critical reduction was very slow, this is faster.
        reduce_uint64(Hs, 1<<(2*n), 0, n_threads); // Hs[0] is the reduced array afterwards
        #pragma omp parallel
        {
            manage_thread_affinity();
            // Returning the result to the output array
            #pragma omp for
            for (uint64_t i=0; i<(1<<(2*n)); i++){
                hist[i]+=Hs[0][i];
            }
        }
        for (int i=0; i<n_threads; i++){
            free(Hs[i]);
        }
        free(Hs);
	}
	// Serial is the simplest implementation for the calculation of 2d histograms.
	// Depending on the load it can be faster then the other options.
	else{
			uint16_t binx = 0;
			uint16_t biny = 0;
			uint32_t BIN;
			for (uint64_t i=0; i<size; i++){
				bin_hist2D_double( (double*)(data1 + i), (double*)(data2 + i), L_bin, max, n, hist, &binx, &biny);  
				BIN = binx + (biny<<n);
				hist[BIN]++;
			}
	}
}

//////