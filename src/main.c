#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "load_balance.h"

/* ------------------------------------------------------------------ */
/*  CSV input reading (proc 0 only)                                    */
/* ------------------------------------------------------------------ */

/*
 * Read comma-separated integers from file into *out_arr.
 * Returns number of elements read via *out_n.
 * Caller must free *out_arr.
 */
static int read_csv_ints(const char *filename, int **out_arr, int *out_n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

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
/*  Pivot-based partitioning (proc 0 only)                             */
/* ------------------------------------------------------------------ */

/*
 * Partitions arr into (num_pivots + 1) buckets using num_pivots partition points.
 * Bucket b contains elements where pivots[b-1] < x <= pivots[b].
 * Bucket 0: x <= pivots[0]
 * Bucket num_pivots: x > pivots[num_pivots-1]  (catch-all)
 *
 * buckets and bucket_sizes must have room for (num_pivots + 1) entries.
 */
static void partition_by_pivots(int *arr, int n, int *pivots, int num_pivots,
                                int **buckets, int *bucket_sizes) {
    int num_buckets = num_pivots + 1;

    memset(bucket_sizes, 0, num_buckets * sizeof(int));
    for (int i = 0; i < n; i++) {
        int b;
        for (b = 0; b < num_pivots; b++) {
            if (arr[i] <= pivots[b]) break;
        }
        bucket_sizes[b]++;
    }

    for (int b = 0; b < num_buckets; b++) {
        buckets[b] = (int *)malloc((bucket_sizes[b] ? bucket_sizes[b] : 1) * sizeof(int));
    }

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

/* ------------------------------------------------------------------ */
/*  Phase 1: Pivot-based distribution                                  */
/* ------------------------------------------------------------------ */

static void distribute_by_pivots(int rank, int p,
                                 const char *input_file, const char *pivot_file,
                                 SubArray *local_array, TimingBreakdown *timing) {
    double t_start = MPI_Wtime();

    int *global_array = NULL;
    int N = 0;
    int *pivots = NULL;
    int num_pivots = 0;

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
        int *buckets[MAX_P];
        int bucket_sizes[MAX_P];
        partition_by_pivots(global_array, N, pivots, num_pivots, buckets, bucket_sizes);

        fprintf(stdout, "[Rank 0] Bucket sizes after pivoting:");
        for (int i = 0; i < p; i++)
            fprintf(stdout, " %d", bucket_sizes[i]);
        fprintf(stdout, "\n");

        for (int dest = 1; dest < p; dest++) {
            MPI_Send(&bucket_sizes[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            if (bucket_sizes[dest] > 0) {
                MPI_Send(buckets[dest], bucket_sizes[dest], MPI_INT,
                         dest, 1, MPI_COMM_WORLD);
            }
        }

        local_array->size = bucket_sizes[0];
        local_array->capacity = bucket_sizes[0];
        local_array->data = buckets[0];  /* take ownership */
        local_array->rank = rank;
        local_array->comp_load = 0.0;

        for (int i = 1; i < p; i++)
            free(buckets[i]);
        free(global_array);
        free(pivots);

    } else {
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
/*  Phase 4: Local insertion sort                                      */
/* ------------------------------------------------------------------ */

static void insertion_sort(int *arr, int n) {
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

static void sort_local(int rank, SubArray *local_array, TimingBreakdown *timing) {
    double t_start = MPI_Wtime();
    if (local_array->size > 0) {
        insertion_sort(local_array->data, local_array->size);
    }
    timing->sort_time = MPI_Wtime() - t_start;
    (void)rank;
}

/* ------------------------------------------------------------------ */
/*  Phase 5: Final PSOR verification                                   */
/* ------------------------------------------------------------------ */

static int verify_psor_final(int rank, int p, SubArray *local_array) {
    int local_min = 0, local_max = 0;
    if (local_array->size > 0) {
        /* After sorting, min is first element, max is last */
        local_min = local_array->data[0];
        local_max = local_array->data[local_array->size - 1];
    }

    int all_mins[MAX_P], all_maxs[MAX_P], all_sizes[MAX_P];
    MPI_Gather(&local_min, 1, MPI_INT, all_mins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max, 1, MPI_INT, all_maxs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_array->size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int psor_ok = 1;
    if (rank == 0) {
        printf("\n=== Phase 5: Final Verification (Post-Sort) ===\n");
        int total = 0;
        for (int i = 0; i < p; i++) {
            printf("  Proc %d: size=%d, min=%d, max=%d\n",
                   i, all_sizes[i], all_mins[i], all_maxs[i]);
            total += all_sizes[i];
        }
        for (int i = 0; i < p - 1; i++) {
            if (all_maxs[i] > all_mins[i + 1]) {
                printf("  PSOR VIOLATION: max(A(%d))=%d > min(A(%d))=%d\n",
                       i, all_maxs[i], i + 1, all_mins[i + 1]);
                psor_ok = 0;
            }
        }
        printf("  Total elements: %d\n", total);
        printf("  PSOR after sorting: %s\n", psor_ok ? "PASSED" : "FAILED");

        for (int i = 0; i < p; i++) {
            if (all_sizes[i] > 0 && all_mins[i] > all_maxs[i]) {
                printf("  ERROR: Proc %d subarray not sorted (min=%d > max=%d)\n",
                       i, all_mins[i], all_maxs[i]);
                psor_ok = 0;
            }
        }
    }

    MPI_Bcast(&psor_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return psor_ok;
}

/* ------------------------------------------------------------------ */
/*  Phase 6: Token-passing file write                                  */
/* ------------------------------------------------------------------ */

static void write_sorted_array_token_pass(int rank, int p,
                                          SubArray *sa, const char *output_file) {
    int token = 0;

    if (rank == 0) {
        FILE *fp = fopen(output_file, "w");
        if (!fp) {
            fprintf(stderr, "Error: cannot open output file %s\n", output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "Rank %d:", rank);
        for (int i = 0; i < sa->size; i++)
            fprintf(fp, " %d", sa->data[i]);
        fprintf(fp, "\n");
        fflush(fp);
        fclose(fp);

        if (p > 1)
            MPI_Send(&token, 1, MPI_INT, 1, 500, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 500,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        FILE *fp = fopen(output_file, "a");
        if (!fp) {
            fprintf(stderr, "Rank %d: cannot open output file %s\n",
                    rank, output_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(fp, "Rank %d:", rank);
        for (int i = 0; i < sa->size; i++)
            fprintf(fp, " %d", sa->data[i]);
        fprintf(fp, "\n");
        fflush(fp);
        fclose(fp);

        if (rank + 1 < p)
            MPI_Send(&token, 1, MPI_INT, rank + 1, 500, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/* ------------------------------------------------------------------ */
/*  Phase 6: Statistics file                                           */
/* ------------------------------------------------------------------ */

static void write_stats_to_file(int rank, int p,
                                TimingBreakdown *timing, LoadMetrics *metrics,
                                const char *stat_file, int enable_lb) {
    double max_pivot_time, max_sort_time, max_lb_time, max_total_time;
    MPI_Reduce(&timing->pivot_time, &max_pivot_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing->sort_time, &max_sort_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing->lb_time, &max_lb_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing->total_time, &max_total_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double parallel_time;
        if (enable_lb) {
            parallel_time = max_pivot_time + max_lb_time + max_sort_time;
        } else {
            parallel_time = max_pivot_time + max_sort_time;
        }
        timing->parallel_time = parallel_time;

        FILE *fp = fopen(stat_file, "w");
        if (!fp) {
            fprintf(stderr, "Error: cannot open stats file %s\n", stat_file);
            return;
        }

        fprintf(fp, "=== Parallel Sort Statistics ===\n");
        fprintf(fp, "Processors: %d\n", p);
        fprintf(fp, "Mode: %s\n", enable_lb ? "with_lb" : "no_lb");
        fprintf(fp, "\n--- Timing (max across ranks) ---\n");
        fprintf(fp, "Pivot time:    %.6f sec\n", max_pivot_time);
        fprintf(fp, "Sort time:     %.6f sec\n", max_sort_time);
        fprintf(fp, "LB time:       %.6f sec\n", max_lb_time);
        fprintf(fp, "IO time:       %.6f sec\n", timing->io_time);
        fprintf(fp, "Parallel time: %.6f sec\n", parallel_time);
        if (timing->sequential_baseline > 0) {
            double speedup = timing->sequential_baseline / parallel_time;
            double efficiency = speedup / p;
            fprintf(fp, "Sequential baseline: %.6f sec\n", timing->sequential_baseline);
            fprintf(fp, "Speedup:       %.4f\n", speedup);
            fprintf(fp, "Efficiency:    %.4f\n", efficiency);
        }

        fprintf(fp, "\n--- Imbalance Metrics ---\n");
        fprintf(fp, "Initial quantitative: %.6f\n", metrics->initial_quant_imbalance);
        fprintf(fp, "Initial qualitative:  %.6f\n", metrics->initial_qual_imbalance);
        fprintf(fp, "Final quantitative:   %.6f\n", metrics->final_quant_imbalance);
        fprintf(fp, "Final qualitative:    %.6f\n", metrics->final_qual_imbalance);

        fclose(fp);

        printf("\n=== Final Statistics ===\n");
        printf("  Parallel time (%s): %.6f sec\n",
               enable_lb ? "with LB" : "no LB", parallel_time);
        printf("  Pivot: %.6f, Sort: %.6f, LB: %.6f\n",
               max_pivot_time, max_sort_time, max_lb_time);
        if (timing->sequential_baseline > 0) {
            double speedup = timing->sequential_baseline / parallel_time;
            double efficiency = speedup / p;
            printf("  Sequential baseline: %.6f sec\n", timing->sequential_baseline);
            printf("  Speedup: %.4f, Efficiency: %.4f\n", speedup, efficiency);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  CLI parsing                                                        */
/* ------------------------------------------------------------------ */

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input_file> <pivot_file> [--mode=no_lb|with_lb] [--cl=default|rand]\n", prog);
    fprintf(stderr, "  --mode=no_lb   : skip load balancing (default)\n");
    fprintf(stderr, "  --mode=with_lb : enable load balancing\n");
}

static int parse_lb_mode(int argc, char *argv[]) {
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--mode=with_lb") == 0) return 1;
        if (strcmp(argv[i], "--mode=no_lb") == 0) return 0;
    }
    return 0;  /* default: no LB */
}

static int parse_cl_mode(int argc, char *argv[]) {
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--cl=rand") == 0) return 1;
        if (strcmp(argv[i], "--cl=default") == 0) return 0;
    }
    return 0;  /* default: deterministic CL */
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Validate P is power of 2 */
    if (p < 1 || p > MAX_P || (p & (p - 1)) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: P must be a power of 2 between 1 and %d (got %d)\n", MAX_P, p);
        MPI_Finalize();
        return 1;
    }

    if (argc < 3) {
        if (rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *pivot_file = argv[2];
    int enable_lb = parse_lb_mode(argc, argv);
    int rand_cl = parse_cl_mode(argc, argv);

    estimate_cl_func = rand_cl ? estimate_cl_rand : estimate_cl;

    if (rank == 0)
        printf("=== Parallel Sort: P=%d, mode=%s, cl=%s ===\n",
               p, enable_lb ? "with_lb" : "no_lb", rand_cl ? "rand" : "default");

    SubArray local_array = {NULL, 0, 0, 0.0, rank};
    TimingBreakdown timing = {0};
    LoadMetrics metrics = {0};

    double t_total_start = MPI_Wtime();

    srand(rank+67); // Random seed

    /* ---- Phase 1: Pivot distribution ---- */
    distribute_by_pivots(rank, p, input_file, pivot_file, &local_array, &timing);

    /* ---- Phase 2: Initial Load Estimation ---- */
    local_array.comp_load = estimate_cl_func(local_array.data, local_array.size);

    int all_sizes[MAX_P];
    double all_loads[MAX_P];
    MPI_Allgather(&local_array.size, 1, MPI_INT,
                  all_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_array.comp_load, 1, MPI_DOUBLE,
                  all_loads, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    metrics.initial_quant_imbalance = compute_quantitative_imbalance(all_sizes, p);
    metrics.initial_qual_imbalance = compute_qualitative_imbalance(all_loads, p);

    printf("[Rank %d] size=%d, CL=%.1f, pivot_time=%.6f sec\n",
           rank, local_array.size, local_array.comp_load, timing.pivot_time);

    /* Verify PSOR after pivoting */
    int local_min = 0, local_max = 0;
    if (local_array.size > 0) {
        local_min = local_array.data[0];
        local_max = local_array.data[0];
        for (int i = 1; i < local_array.size; i++) {
            if (local_array.data[i] < local_min) local_min = local_array.data[i];
            if (local_array.data[i] > local_max) local_max = local_array.data[i];
        }
    }

    int all_mins[MAX_P], all_maxs[MAX_P];
    MPI_Gather(&local_min, 1, MPI_INT, all_mins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max, 1, MPI_INT, all_maxs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Phase 1: PSOR Verification ===\n");
        int total = 0;
        int psor_ok = 1;
        for (int i = 0; i < p; i++) {
            printf("  Proc %d: size=%d, min=%d, max=%d\n",
                   i, all_sizes[i], all_mins[i], all_maxs[i]);
            total += all_sizes[i];
        }
        for (int i = 0; i < p - 1; i++) {
            if (all_maxs[i] > all_mins[i + 1]) {
                printf("  PSOR VIOLATION: max(A(%d))=%d > min(A(%d))=%d\n",
                       i, all_maxs[i], i + 1, all_mins[i + 1]);
                psor_ok = 0;
            }
        }
        printf("  Total elements: %d\n", total);
        printf("  PSOR: %s\n", psor_ok ? "PASSED" : "FAILED");
        printf("  Pivot time: %.6f sec\n", timing.pivot_time);

        printf("\n=== Phase 2: Initial Load Estimation ===\n");
        for (int i = 0; i < p; i++) {
            printf("  Proc %d: n(i)=%d, CL(A(i))=%.1f\n",
                   i, all_sizes[i], all_loads[i]);
        }
        printf("  Quantitative imbalance (StdDev(n)/avg(n)): %.6f\n",
               metrics.initial_quant_imbalance);
        printf("  Qualitative imbalance (StdDev(CL)/avg(CL)): %.6f\n",
               metrics.initial_qual_imbalance);
    }

    /* ---- Phase 3: Load Balancing (conditional) ---- */
    if (enable_lb) {
        if (rank == 0)
            printf("\n=== Phase 3: Load Balancing (%d rounds max) ===\n", LB_MAX_ROUNDS);

        synchronous_lb_linear(rank, p, &local_array, LB_MAX_ROUNDS, &timing, &metrics);

        if (rank == 0)
            printf("  LB time: %.6f sec\n", timing.lb_time);

        /* Re-verify PSOR after LB */
        local_min = 0; local_max = 0;
        if (local_array.size > 0) {
            local_min = local_array.data[0];
            local_max = local_array.data[0];
            for (int i = 1; i < local_array.size; i++) {
                if (local_array.data[i] < local_min) local_min = local_array.data[i];
                if (local_array.data[i] > local_max) local_max = local_array.data[i];
            }
        }

        MPI_Gather(&local_min, 1, MPI_INT, all_mins, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_max, 1, MPI_INT, all_maxs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_array.size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\n=== Post-LB PSOR Verification ===\n");
            int total = 0;
            int psor_ok = 1;
            for (int i = 0; i < p; i++) {
                printf("  Proc %d: size=%d, min=%d, max=%d\n",
                       i, all_sizes[i], all_mins[i], all_maxs[i]);
                total += all_sizes[i];
            }
            for (int i = 0; i < p - 1; i++) {
                if (all_maxs[i] > all_mins[i + 1]) {
                    printf("  PSOR VIOLATION: max(A(%d))=%d > min(A(%d))=%d\n",
                           i, all_maxs[i], i + 1, all_mins[i + 1]);
                    psor_ok = 0;
                }
            }
            printf("  Total elements: %d\n", total);
            printf("  PSOR after LB: %s\n", psor_ok ? "PASSED" : "FAILED");
        }
    } else {
        if (rank == 0)
            printf("\n=== Phase 3: Load Balancing SKIPPED (no_lb mode) ===\n");
        /* Set final imbalance = initial (no LB performed) */
        metrics.final_quant_imbalance = metrics.initial_quant_imbalance;
        metrics.final_qual_imbalance = metrics.initial_qual_imbalance;
    }

    /* ---- Phase 4: Local Sorting ---- */
    if (rank == 0)
        printf("\n=== Phase 4: Local Sorting (Insertion Sort) ===\n");

    sort_local(rank, &local_array, &timing);

    printf("[Rank %d] sort_time=%.6f sec, n=%d\n",
           rank, timing.sort_time, local_array.size);

    /* ---- Phase 5: Final Verification ---- */
    verify_psor_final(rank, p, &local_array);

    /* ---- Phase 6: Output & Statistics ---- */
    double t_io_start = MPI_Wtime();

    /*
     * Derive a tag from the input filename so output files are unique per run.
     * e.g. "Inputs/A_vector_100000_8_2.txt" -> tag "100000_8_2"
     */
    char tag[128];
    {
        const char *base = strrchr(input_file, '/');
        base = base ? base + 1 : input_file;
        const char *prefix = "A_vector_";
        if (strncmp(base, prefix, strlen(prefix)) == 0)
            base += strlen(prefix);
        strncpy(tag, base, sizeof(tag) - 1);
        tag[sizeof(tag) - 1] = '\0';
        char *dot = strrchr(tag, '.');
        if (dot) *dot = '\0';
    }

    char output_file[256];
    snprintf(output_file, sizeof(output_file), "outputs/sorted_%s%s_%s.txt",
             enable_lb ? "with_lb" : "no_lb",
	     rand_cl ? "_rand" : "",
	     tag);

    write_sorted_array_token_pass(rank, p, &local_array, output_file);

    timing.io_time = MPI_Wtime() - t_io_start;
    timing.total_time = MPI_Wtime() - t_total_start;

    if (enable_lb) {
        timing.parallel_time = timing.pivot_time + timing.lb_time + timing.sort_time;
    } else {
        timing.parallel_time = timing.pivot_time + timing.sort_time;
    }

    if (p == 1) {
        timing.sequential_baseline = timing.sort_time;
    }

    char stat_file[256];
    snprintf(stat_file, sizeof(stat_file), "outputs/stats_%s%s_%s.txt",
             enable_lb ? "with_lb" : "no_lb",
	     rand_cl ? "_rand" : "",
	     tag);

    write_stats_to_file(rank, p, &timing, &metrics, stat_file, enable_lb);

    if (rank == 0) {
        printf("\n=== Output Written ===\n");
        printf("  Sorted array: %s\n", output_file);
        printf("  Statistics:   %s\n", stat_file);
        printf("  Total wall-clock: %.6f sec\n", timing.total_time);
    }

    subarray_free(&local_array);
    MPI_Finalize();
    return 0;
}
