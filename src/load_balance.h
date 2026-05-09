#ifndef LOAD_BALANCE_H
#define LOAD_BALANCE_H

#include "parallel_sort.h"
#include "load_analysis.h"

/*
 * Phase 3: Synchronous load balancing on linear chain (rank order).
 *
 * Transfer decisions are based on CL(A(i)) — the estimated insertion sort
 * computational load. Elements are exchanged between rank-adjacent processors
 * only (i with i-1 and i+1) to preserve PSOR.
 *
 * Subarrays are unsorted at this stage. Uses O(n) partial selection
 * (nth_element-style partition) to find k smallest/largest elements.
 *
 * max_rounds: maximum number of LB iterations.
 * Early termination if qualitative imbalance drops below LB_IMBALANCE_THRESH.
 *
 * Records lb_time in timing.
 * Updates metrics->final_quant_imbalance and final_qual_imbalance.
 */
void synchronous_lb_linear(int rank, int p,
                           SubArray *local_array,
                           int max_rounds,
                           TimingBreakdown *timing,
                           LoadMetrics *metrics);

#endif /* LOAD_BALANCE_H */
