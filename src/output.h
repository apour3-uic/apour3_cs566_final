#ifndef OUTPUT_H
#define OUTPUT_H

#include "parallel_sort.h"
#include "load_analysis.h"

/*
 * Phase 6: Token-passing file write.
 * Each processor writes its sorted subarray in rank order.
 * Token starts at rank 0; each rank opens file in append mode when it holds the token.
 * Output format: "Rank X: elem1 elem2 ... elemN\n"
 */
void write_sorted_array_token_pass(int rank, int p,
                                   SubArray *sa, const char *output_file);

/*
 * Write statistics to file (proc 0 only).
 * Gathers timing data from all ranks via MPI_Reduce (MPI_MAX for parallel times).
 * Writes CSV summary row.
 */
void write_stats_to_file(int rank, int p,
                         TimingBreakdown *timing, LoadMetrics *metrics,
                         const char *stat_file, int enable_lb);

/*
 * Phase 5: Lightweight PSOR verification after sorting.
 * Each proc sends min, max, size to rank 0.
 * Rank 0 verifies PSOR and element completeness.
 * Returns 1 if PSOR passed (on all ranks), 0 if failed.
 */
int verify_psor_final(int rank, int p, SubArray *local_array);

#endif /* OUTPUT_H */
