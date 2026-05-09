#include "parallel_sort.h"
#include "sorting.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  File I/O helpers (proc 0 only)                                    */
/* ------------------------------------------------------------------ */

/*
 * Read comma-separated integers from file into *out_arr.
 * Returns number of elements read via *out_n.
 * Caller must free *out_arr.
 */
static int read_csv_ints(const char *filename, int **out_arr, int *out_n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    /* Get file size */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buf = (char *)malloc(fsize + 1);
    if (!buf) { fclose(fp); return -1; }
    fread(buf, 1, fsize, fp);
    buf[fsize] = '\0';
    fclose(fp);

    /* Count commas to estimate element count */
    int est = 1;
    for (long i = 0; i < fsize; i++)
        if (buf[i] == ',') est++;

    int *arr = (int *)malloc(est * sizeof(int));
    int count = 0;
    char *tok = strtok(buf, ", \t\n\r");
    while (tok) {
        arr[count++] = atoi(tok);
        tok = strtok(NULL, ", \t\n\r");
    }

    free(buf);
    *out_arr = arr;
    *out_n = count;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Phase 1: Pivot-based distribution                                 */
/* ------------------------------------------------------------------ */

void distribute_by_pivots(int rank, int p,
                          const char *input_file, const char *pivot_file,
                          SubArray *local_array, TimingBreakdown *timing) {
    double t_start = MPI_Wtime();

    int *global_array = NULL;
    int N = 0;
    int *pivots = NULL;
    int num_pivots = 0;

    /* Proc 0: read input and pivots */
    if (rank == 0) {
        if (read_csv_ints(input_file, &global_array, &N) != 0) {
            fprintf(stderr, "Error: cannot read input file %s\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (read_csv_ints(pivot_file, &pivots, &num_pivots) != 0) {
            fprintf(stderr, "Error: cannot read pivot file %s\n", pivot_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (num_pivots != p - 1) {
            fprintf(stderr, "Error: pivot file has %d pivots, expected %d (= P-1)\n",
                    num_pivots, p - 1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(stdout, "[Rank 0] Read %d elements, %d pivots\n", N, num_pivots);
    }

    if (rank == 0) {
        /* Partition into P buckets */
        int *buckets[MAX_P];
        int bucket_sizes[MAX_P];
        partition_by_pivots(global_array, N, pivots, num_pivots, buckets, bucket_sizes);

        /* Print bucket sizes for debugging */
        fprintf(stdout, "[Rank 0] Bucket sizes after pivoting:");
        for (int i = 0; i < p; i++)
            fprintf(stdout, " %d", bucket_sizes[i]);
        fprintf(stdout, "\n");

        /* Send buckets to other procs, keep bucket[0] for self */
        for (int dest = 1; dest < p; dest++) {
            MPI_Send(&bucket_sizes[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            if (bucket_sizes[dest] > 0) {
                MPI_Send(buckets[dest], bucket_sizes[dest], MPI_INT,
                         dest, 1, MPI_COMM_WORLD);
            }
        }

        /* Keep bucket[0] */
        local_array->size = bucket_sizes[0];
        local_array->capacity = bucket_sizes[0];
        local_array->data = buckets[0];  /* take ownership */
        local_array->rank = rank;
        local_array->comp_load = 0.0;

        /* Free other buckets (data already sent) */
        for (int i = 1; i < p; i++)
            free(buckets[i]);
        free(global_array);
        free(pivots);

    } else {
        /* Receive from proc 0 */
        int recv_size;
        MPI_Recv(&recv_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_array->size = recv_size;
        local_array->capacity = recv_size;
        local_array->rank = rank;
        local_array->comp_load = 0.0;

        if (recv_size > 0) {
            local_array->data = (int *)malloc(recv_size * sizeof(int));
            MPI_Recv(local_array->data, recv_size, MPI_INT,
                     0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            local_array->data = NULL;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timing->pivot_time = MPI_Wtime() - t_start;
}

/* ------------------------------------------------------------------ */
/*  Phase 4: Local insertion sort                                     */
/* ------------------------------------------------------------------ */

void sort_local(int rank, SubArray *local_array, TimingBreakdown *timing) {
    double t_start = MPI_Wtime();
    if (local_array->size > 0) {
        insertion_sort(local_array->data, local_array->size);
    }
    timing->sort_time = MPI_Wtime() - t_start;
}

/* ------------------------------------------------------------------ */
/*  Cleanup                                                           */
/* ------------------------------------------------------------------ */

void subarray_free(SubArray *sa) {
    if (sa->data) {
        free(sa->data);
        sa->data = NULL;
    }
    sa->size = 0;
    sa->capacity = 0;
}
