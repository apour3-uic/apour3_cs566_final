#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "parallel_sort.h"
#include "load_analysis.h"
#include "load_balance.h"
#include "output.h"
#include "config.h"

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
     * Strip directory, strip "A_vector_" prefix, strip ".txt" suffix.
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

    /* Build output filename: sorted_{mode}_{N}_{P}_{s}.txt */
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "outputs/sorted_%s%s_%s.txt",
             enable_lb ? "with_lb" : "no_lb", 
	     rand_cl ? "_rand" : "", 
	     tag);

    write_sorted_array_token_pass(rank, p, &local_array, output_file);

    timing.io_time = MPI_Wtime() - t_io_start;
    timing.total_time = MPI_Wtime() - t_total_start;

    /* Compute parallel time for this run */
    if (enable_lb) {
        timing.parallel_time = timing.pivot_time + timing.lb_time + timing.sort_time;
    } else {
        timing.parallel_time = timing.pivot_time + timing.sort_time;
    }

    /* For P=1, also record as sequential baseline */
    if (p == 1) {
        timing.sequential_baseline = timing.sort_time;
    }

    /* Write stats */
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
