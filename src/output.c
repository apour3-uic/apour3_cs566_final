#include "output.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
/*  Phase 6: Token-passing file write                                  */
/* ------------------------------------------------------------------ */

void write_sorted_array_token_pass(int rank, int p,
                                   SubArray *sa, const char *output_file) {
    double t_start = MPI_Wtime();
    int token = 0;

    if (rank == 0) {
        /* Rank 0 writes first (create/truncate the file) */
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

        /* Pass token to rank 1 */
        if (p > 1)
            MPI_Send(&token, 1, MPI_INT, 1, 500, MPI_COMM_WORLD);
    } else {
        /* Wait for token from rank-1 */
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 500,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Append my sorted subarray */
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

        /* Pass token to next rank */
        if (rank + 1 < p)
            MPI_Send(&token, 1, MPI_INT, rank + 1, 500, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* io_time not recorded here — caller handles it */
    (void)t_start;
}

/* ------------------------------------------------------------------ */
/*  Phase 6: Statistics file                                           */
/* ------------------------------------------------------------------ */

void write_stats_to_file(int rank, int p,
                         TimingBreakdown *timing, LoadMetrics *metrics,
                         const char *stat_file, int enable_lb) {
    /* Gather max times across all ranks (wall-clock parallel time) */
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

        /* Also print to stdout */
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
/*  Phase 5: Final PSOR verification                                   */
/* ------------------------------------------------------------------ */

int verify_psor_final(int rank, int p, SubArray *local_array) {
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

        /* Verify each subarray is locally sorted (by checking min == first, max == last) */
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
