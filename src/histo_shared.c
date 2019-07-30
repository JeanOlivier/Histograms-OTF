#include "../includes/histo_shared.h"
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



void reduce_uint64(uint64_t** arrs, uint64_t bins, uint64_t begin, uint64_t end)
{
    assert(begin < end);
    if (end - begin == 1) {
        return;
    }
    uint64_t pivot = (begin + end) / 2;
    /* Moving the termination condition here will avoid very short tasks,
     * but make the code less nice. */
    //#pragma omp task
    reduce_uint64(arrs, bins, begin, pivot);
    //#pragma omp task
    reduce_uint64(arrs, bins, pivot, end);
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

void reduce_uint8(uint8_t** arrs, uint64_t bins, uint64_t begin, uint64_t end)
{
    assert(begin < end);
    if (end - begin == 1) {
        return;
    }
    uint64_t pivot = (begin + end) / 2;
    /* Moving the termination condition here will avoid very short tasks,
     * but make the code less nice. */
    //#pragma omp task
    reduce_uint8(arrs, bins, begin, pivot);
    //#pragma omp task
    reduce_uint8(arrs, bins, pivot, end);
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