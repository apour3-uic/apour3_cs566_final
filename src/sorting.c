#include "sorting.h"
#include <stdlib.h>
#include <string.h>

void insertion_sort(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/*
 * Partitions arr into (num_pivots + 1) buckets using num_pivots partition points.
 * Bucket b contains elements where pivots[b-1] < x <= pivots[b].
 * Bucket 0: x <= pivots[0]
 * Bucket num_pivots: x > pivots[num_pivots-1]  (catch-all)
 *
 * buckets and bucket_sizes must have room for (num_pivots + 1) entries.
 */
void partition_by_pivots(int *arr, int n, int *pivots, int num_pivots,
                         int **buckets, int *bucket_sizes) {
    int num_buckets = num_pivots + 1;

    /* First pass: count elements per bucket */
    memset(bucket_sizes, 0, num_buckets * sizeof(int));
    for (int i = 0; i < n; i++) {
        int b;
        for (b = 0; b < num_pivots; b++) {
            if (arr[i] <= pivots[b]) break;
        }
        /* b == num_pivots means element goes into last (catch-all) bucket */
        bucket_sizes[b]++;
    }

    /* Allocate buckets */
    for (int b = 0; b < num_buckets; b++) {
        buckets[b] = (int *)malloc((bucket_sizes[b] ? bucket_sizes[b] : 1) * sizeof(int));
    }

    /* Second pass: distribute elements */
    int *offsets = (int *)calloc(num_buckets, sizeof(int));
    for (int i = 0; i < n; i++) {
        int b;
        for (b = 0; b < num_pivots; b++) {
            if (arr[i] <= pivots[b]) break;
        }
        buckets[b][offsets[b]++] = arr[i];
    }
    free(offsets);
}
