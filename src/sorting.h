#ifndef SORTING_H
#define SORTING_H

/*
 * Sort array in-place using insertion sort.
 * Complexity: Theta(n) best case (sorted), Theta(n^2) worst case (reverse).
 */
void insertion_sort(int *arr, int n);

/*
 * Partition arr[0..n-1] into P buckets using P pivot boundaries.
 * Bucket i gets elements in (pivots[i-1], pivots[i]] for i>0,
 * and elements <= pivots[0] for bucket 0.
 *
 * pivots must be sorted in ascending order and pivots[P-1] >= max(arr).
 *
 * Output:
 *   buckets[i]      = allocated array of elements for bucket i
 *   bucket_sizes[i] = number of elements in bucket i
 *
 * Caller must free each buckets[i].
 */
void partition_by_pivots(int *arr, int n, int *pivots, int num_pivots,
                         int **buckets, int *bucket_sizes);

#endif /* SORTING_H */
