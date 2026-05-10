#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "load_balance.h"

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

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file> <pivot_file> <N> [--mode=no_lb|with_lb] [--cl=default|rand]\n", argv[0]);
            fprintf(stderr, "  N              : number of ints in <input_file> (required)\n");
            fprintf(stderr, "  --mode=no_lb   : skip load balancing (default)\n");
            fprintf(stderr, "  --mode=with_lb : enable load balancing\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_file = argv[1];
    const char *pivot_file = argv[2];
    int N = atoi(argv[3]);
    if (N <= 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N must be a positive integer (got '%s')\n", argv[3]);
        MPI_Finalize();
        return 1;
    }

    /* ---- CLI flag parsing ---- */
    int enable_lb = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--mode=with_lb") == 0) { enable_lb = 1; break; }
        if (strcmp(argv[i], "--mode=no_lb")   == 0) { enable_lb = 0; break; }
    }
    int rand_cl = 0;
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--cl=rand")    == 0) { rand_cl = 1; break; }
        if (strcmp(argv[i], "--cl=default") == 0) { rand_cl = 0; break; }
    }

    estimate_cl_func = rand_cl ? estimate_cl_rand : estimate_cl;

    if (rank == 0)
        printf("=== Parallel Sort: P=%d, mode=%s, cl=%s ===\n",
               p, enable_lb ? "with_lb" : "no_lb", rand_cl ? "rand" : "default");

    SubArray local_array = {NULL, 0, 0, 0.0, rank};
    TimingBreakdown timing = {0};
    LoadMetrics metrics = {0};

    double t_total_start = MPI_Wtime();

    srand(rank + 67); /* Random seed */

    /* ================================================================ */
    /*  Phase 1: Pivot-based distribution                                */
    /* ================================================================ */
    {
        double t_start = MPI_Wtime();

        if (rank == 0) {
            int num_pivots = p - 1;
            int *global_array = (int *)malloc(N * sizeof(int));
            int *pivots = (int *)malloc((num_pivots ? num_pivots : 1) * sizeof(int));

            /* Read input_file: exactly N comma/whitespace-separated ints */
            {
                FILE *fp = fopen(input_file, "r");
                if (!fp) {
                    fprintf(stderr, "Error: cannot read input file %s\n", input_file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (int i = 0; i < N; i++) {
                    if (fscanf(fp, " %d", &global_array[i]) != 1) {
                        fprintf(stderr, "Error: input file %s has fewer than %d ints (got %d)\n",
                                input_file, N, i);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    int sep_ret = fscanf(fp, " ,"); (void)sep_ret;  /* optional separator */
                }
                int extra;
                if (fscanf(fp, " %d", &extra) == 1) {
                    fprintf(stderr, "Error: input file %s has more than %d ints\n",
                            input_file, N);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                fclose(fp);
            }

            /* Read pivot_file: exactly p-1 ints */
            {
                FILE *fp = fopen(pivot_file, "r");
                if (!fp) {
                    fprintf(stderr, "Error: cannot read pivot file %s\n", pivot_file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                for (int i = 0; i < num_pivots; i++) {
                    if (fscanf(fp, " %d", &pivots[i]) != 1) {
                        fprintf(stderr, "Error: pivot file %s has fewer than %d ints (got %d)\n",
                                pivot_file, num_pivots, i);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    int sep_ret = fscanf(fp, " ,"); (void)sep_ret;  /* optional separator */
                }
                int extra;
                if (fscanf(fp, " %d", &extra) == 1) {
                    fprintf(stderr, "Error: pivot file %s has more than %d ints\n",
                            pivot_file, num_pivots);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                fclose(fp);
            }

            fprintf(stdout, "[Rank 0] Read %d elements, %d pivots\n", N, num_pivots);

            /* Partition global_array into p buckets:
             *   bucket b = { x : pivots[b-1] < x <= pivots[b] }
             *   bucket 0 = { x <= pivots[0] }; bucket p-1 = { x > pivots[p-2] } */
            int *buckets[MAX_P];
            int bucket_sizes[MAX_P];
            memset(bucket_sizes, 0, p * sizeof(int));
            for (int i = 0; i < N; i++) {
                int b;
                for (b = 0; b < num_pivots; b++)
                    if (global_array[i] <= pivots[b]) break;
                bucket_sizes[b]++;
            }
            for (int b = 0; b < p; b++)
                buckets[b] = (int *)malloc((bucket_sizes[b] ? bucket_sizes[b] : 1) * sizeof(int));

            int *offsets = (int *)calloc(p, sizeof(int));
            for (int i = 0; i < N; i++) {
                int b;
                for (b = 0; b < num_pivots; b++)
                    if (global_array[i] <= pivots[b]) break;
                buckets[b][offsets[b]++] = global_array[i];
            }
            free(offsets);

            fprintf(stdout, "[Rank 0] Bucket sizes after pivoting:");
            for (int i = 0; i < p; i++)
                fprintf(stdout, " %d", bucket_sizes[i]);
            fprintf(stdout, "\n");

            /* Send buckets 1..p-1, keep bucket 0 locally */
            for (int dest = 1; dest < p; dest++) {
                MPI_Send(&bucket_sizes[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                if (bucket_sizes[dest] > 0)
                    MPI_Send(buckets[dest], bucket_sizes[dest], MPI_INT,
                             dest, 1, MPI_COMM_WORLD);
            }

            local_array.size = bucket_sizes[0];
            local_array.capacity = bucket_sizes[0];
            local_array.data = buckets[0]; /* take ownership */

            for (int i = 1; i < p; i++)
                free(buckets[i]);
            free(global_array);
            free(pivots);
        } else {
            int recv_size;
            MPI_Recv(&recv_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_array.size = recv_size;
            local_array.capacity = recv_size;

            if (recv_size > 0) {
                local_array.data = (int *)malloc(recv_size * sizeof(int));
                MPI_Recv(local_array.data, recv_size, MPI_INT,
                         0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                local_array.data = NULL;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        timing.pivot_time = MPI_Wtime() - t_start;
    }

    /* ================================================================ */
    /*  Phase 2: Initial Load Estimation                                 */
    /* ================================================================ */
    local_array.comp_load = estimate_cl_func(local_array.data, local_array.size);

    int all_sizes[MAX_P];
    double all_loads[MAX_P];
    MPI_Allgather(&local_array.size, 1, MPI_INT,
                  all_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_array.comp_load, 1, MPI_DOUBLE,
                  all_loads, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    metrics.initial_quant_imbalance = compute_quantitative_imbalance(all_sizes, p);
    metrics.initial_qual_imbalance  = compute_qualitative_imbalance(all_loads, p);

    printf("[Rank %d] size=%d, CL=%.1f, pivot_time=%.6f sec\n",
           rank, local_array.size, local_array.comp_load, timing.pivot_time);

    /* ---- Verify PSOR after pivoting ---- */
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

    int psor_ok = 1;
    if (rank == 0) {
        printf("\n=== Phase 1: PSOR Verification ===\n");
        int total = 0;
        for (int i = 0; i < p; i++) {
            printf("  Proc %d: size=%d, min=%d, max=%d\n",
                   i, all_sizes[i], all_mins[i], all_maxs[i]);
            total += all_sizes[i];
        }
        for (int i = 0; i < p - 1; i++) {
            if (all_maxs[i] > all_mins[i + 1]) {
                fprintf(stderr,
                        "FATAL: PSOR VIOLATION post-pivot: max(A(%d))=%d > min(A(%d))=%d\n",
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
    MPI_Bcast(&psor_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!psor_ok) {
        if (rank == 0)
            fprintf(stderr, "FATAL: post-pivot PSOR violated; aborting.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ================================================================ */
    /*  Phase 3: Load Balancing (conditional)                            */
    /* ================================================================ */
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

        int post_lb_psor_ok = 1;
        if (rank == 0) {
            printf("\n=== Post-LB PSOR Verification ===\n");
            int total = 0;
            for (int i = 0; i < p; i++) {
                printf("  Proc %d: size=%d, min=%d, max=%d\n",
                       i, all_sizes[i], all_mins[i], all_maxs[i]);
                total += all_sizes[i];
            }
            for (int i = 0; i < p - 1; i++) {
                if (all_maxs[i] > all_mins[i + 1]) {
                    fprintf(stderr,
                            "FATAL: PSOR VIOLATION post-LB: max(A(%d))=%d > min(A(%d))=%d\n",
                            i, all_maxs[i], i + 1, all_mins[i + 1]);
                    post_lb_psor_ok = 0;
                }
            }
            printf("  Total elements: %d\n", total);
            printf("  PSOR after LB: %s\n", post_lb_psor_ok ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&post_lb_psor_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!post_lb_psor_ok) {
            if (rank == 0)
                fprintf(stderr, "FATAL: post-LB PSOR violated; aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        if (rank == 0)
            printf("\n=== Phase 3: Load Balancing SKIPPED (no_lb mode) ===\n");
        /* Set final imbalance = initial (no LB performed) */
        metrics.final_quant_imbalance = metrics.initial_quant_imbalance;
        metrics.final_qual_imbalance  = metrics.initial_qual_imbalance;
    }

    /* ================================================================ */
    /*  Phase 4: Local Sorting (insertion sort)                          */
    /* ================================================================ */
    if (rank == 0)
        printf("\n=== Phase 4: Local Sorting (Insertion Sort) ===\n");
    {
        double t_start = MPI_Wtime();
        for (int i = 1; i < local_array.size; i++) {
            int key = local_array.data[i];
            int j = i - 1;
            while (j >= 0 && local_array.data[j] > key) {
                local_array.data[j + 1] = local_array.data[j];
                j--;
            }
            local_array.data[j + 1] = key;
        }
        timing.sort_time = MPI_Wtime() - t_start;
    }

    printf("[Rank %d] sort_time=%.6f sec, n=%d\n",
           rank, timing.sort_time, local_array.size);

    /* ================================================================ */
    /*  Phase 5: Final PSOR + sortedness verification                    */
    /* ================================================================ */
    {
        /* Each rank checks its own subarray is fully sorted (linear scan) */
        int local_sorted = 1;
        for (int i = 1; i < local_array.size; i++) {
            if (local_array.data[i - 1] > local_array.data[i]) {
                fprintf(stderr,
                        "FATAL [rank %d]: local subarray not sorted at index %d "
                        "(data[%d]=%d > data[%d]=%d)\n",
                        rank, i, i - 1, local_array.data[i - 1],
                        i, local_array.data[i]);
                local_sorted = 0;
                break;
            }
        }

        /* After sorting: min = first element, max = last element */
        int sorted_min = 0, sorted_max = 0;
        if (local_array.size > 0) {
            sorted_min = local_array.data[0];
            sorted_max = local_array.data[local_array.size - 1];
        }

        int final_mins[MAX_P], final_maxs[MAX_P], final_sizes[MAX_P];
        MPI_Gather(&sorted_min,        1, MPI_INT, final_mins,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&sorted_max,        1, MPI_INT, final_maxs,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_array.size,  1, MPI_INT, final_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int global_sorted;
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI_INT,
                      MPI_LAND, MPI_COMM_WORLD);

        int final_psor_ok = 1;
        if (rank == 0) {
            printf("\n=== Phase 5: Final Verification (Post-Sort) ===\n");
            int total = 0;
            for (int i = 0; i < p; i++) {
                printf("  Proc %d: size=%d, min=%d, max=%d\n",
                       i, final_sizes[i], final_mins[i], final_maxs[i]);
                total += final_sizes[i];
            }
            for (int i = 0; i < p - 1; i++) {
                if (final_maxs[i] > final_mins[i + 1]) {
                    fprintf(stderr,
                            "FATAL: PSOR VIOLATION post-sort: max(A(%d))=%d > min(A(%d))=%d\n",
                            i, final_maxs[i], i + 1, final_mins[i + 1]);
                    final_psor_ok = 0;
                }
            }
            printf("  Total elements: %d\n", total);
            printf("  PSOR after sorting: %s\n", final_psor_ok ? "PASSED" : "FAILED");
            printf("  Local sortedness: %s\n", global_sorted ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&final_psor_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!final_psor_ok || !global_sorted) {
            if (rank == 0)
                fprintf(stderr, "FATAL: post-sort verification failed; aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* ================================================================ */
    /*  Phase 6: Output (token-passed file write) + statistics           */
    /* ================================================================ */
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

    /* ---- Token-passing write: rank 0 truncates, then 1..p-1 append in order ---- */
    {
        int token = 0;
        if (rank == 0) {
            FILE *fp = fopen(output_file, "w");
            if (!fp) {
                fprintf(stderr, "Error: cannot open output file %s\n", output_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fprintf(fp, "Rank %d:", rank);
            for (int i = 0; i < local_array.size; i++)
                fprintf(fp, " %d", local_array.data[i]);
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
            for (int i = 0; i < local_array.size; i++)
                fprintf(fp, " %d", local_array.data[i]);
            fprintf(fp, "\n");
            fflush(fp);
            fclose(fp);

            if (rank + 1 < p)
                MPI_Send(&token, 1, MPI_INT, rank + 1, 500, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    timing.io_time    = MPI_Wtime() - t_io_start;
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

    /* ---- Write statistics: reduce timings to rank 0, then dump ---- */
    {
        double max_pivot_time, max_sort_time, max_lb_time, max_total_time;
        MPI_Reduce(&timing.pivot_time, &max_pivot_time, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timing.sort_time,  &max_sort_time,  1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timing.lb_time,    &max_lb_time,    1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timing.total_time, &max_total_time, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double parallel_time = enable_lb
                ? (max_pivot_time + max_lb_time + max_sort_time)
                : (max_pivot_time + max_sort_time);
            timing.parallel_time = parallel_time;

            FILE *fp = fopen(stat_file, "w");
            if (!fp) {
                fprintf(stderr, "Error: cannot open stats file %s\n", stat_file);
            } else {
                fprintf(fp, "=== Parallel Sort Statistics ===\n");
                fprintf(fp, "Processors: %d\n", p);
                fprintf(fp, "Mode: %s\n", enable_lb ? "with_lb" : "no_lb");
                fprintf(fp, "\n--- Timing (max across ranks) ---\n");
                fprintf(fp, "Pivot time:    %.6f sec\n", max_pivot_time);
                fprintf(fp, "Sort time:     %.6f sec\n", max_sort_time);
                fprintf(fp, "LB time:       %.6f sec\n", max_lb_time);
                fprintf(fp, "IO time:       %.6f sec\n", timing.io_time);
                fprintf(fp, "Parallel time: %.6f sec\n", parallel_time);
                if (timing.sequential_baseline > 0) {
                    double speedup    = timing.sequential_baseline / parallel_time;
                    double efficiency = speedup / p;
                    fprintf(fp, "Sequential baseline: %.6f sec\n", timing.sequential_baseline);
                    fprintf(fp, "Speedup:       %.4f\n", speedup);
                    fprintf(fp, "Efficiency:    %.4f\n", efficiency);
                }

                fprintf(fp, "\n--- Imbalance Metrics ---\n");
                fprintf(fp, "Initial quantitative: %.6f\n", metrics.initial_quant_imbalance);
                fprintf(fp, "Initial qualitative:  %.6f\n", metrics.initial_qual_imbalance);
                fprintf(fp, "Final quantitative:   %.6f\n", metrics.final_quant_imbalance);
                fprintf(fp, "Final qualitative:    %.6f\n", metrics.final_qual_imbalance);

                fclose(fp);
            }

            printf("\n=== Final Statistics ===\n");
            printf("  Parallel time (%s): %.6f sec\n",
                   enable_lb ? "with LB" : "no LB", parallel_time);
            printf("  Pivot: %.6f, Sort: %.6f, LB: %.6f\n",
                   max_pivot_time, max_sort_time, max_lb_time);
            if (timing.sequential_baseline > 0) {
                double speedup    = timing.sequential_baseline / parallel_time;
                double efficiency = speedup / p;
                printf("  Sequential baseline: %.6f sec\n", timing.sequential_baseline);
                printf("  Speedup: %.4f, Efficiency: %.4f\n", speedup, efficiency);
            }
        }
    }

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
