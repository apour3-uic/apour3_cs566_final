#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H

#include <mpi.h>

typedef struct {
    int *data;       /* elements (unsorted during Phases 1-3, sorted after Phase 4) */
    int size;        /* current number of elements n(i) */
    int capacity;    /* allocated memory */
    double comp_load;/* CL(A): estimated insertion sort computational load */
    int rank;        /* owning processor rank */
} SubArray;

typedef struct {
    double pivot_time;
    double sort_time;
    double lb_time;
    double io_time;
    double total_time;
    double sequential_baseline;
    double parallel_time;  /* this run's time: no_lb or with_lb */
} TimingBreakdown;

/*
 * Phase 1: Proc 0 reads input array and pivot vector, partitions array
 * into P buckets using pivots, sends bucket[i] to proc i.
 *
 * After return, each proc's local_array is populated with its subarray.
 * PSOR is guaranteed: all elements in bucket i <= all elements in bucket i+1.
 */
void distribute_by_pivots(int rank, int p,
                          const char *input_file, const char *pivot_file,
                          SubArray *local_array, TimingBreakdown *timing);

/*
 * Phase 4: Sort local subarray using insertion sort. Records sort_time.
 */
void sort_local(int rank, SubArray *local_array, TimingBreakdown *timing);

/*
 * Free SubArray data.
 */
void subarray_free(SubArray *sa);

#endif /* PARALLEL_SORT_H */
