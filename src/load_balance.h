#ifndef LOAD_BALANCE_H
#define LOAD_BALANCE_H

#include <mpi.h>

#define MAX_P         128
#define LB_MAX_ROUNDS 10

typedef struct {
    int *data;        /* elements (unsorted during Phases 1-3, sorted after Phase 4) */
    int size;         /* current number of elements n(i) */
    int capacity;     /* allocated memory */
} SubArray;

typedef struct {
    /* Imbalance after pivoting, before LB */
    double initial_quant_imbalance;   /* StdDev(n(i)) / (N/P) */
    double initial_qual_imbalance;    /* StdDev(CL(A(i))) / avg(CL(A(i))) */

    /* Imbalance after LB, before final sort */
    double final_quant_imbalance;
    double final_qual_imbalance;

} LoadMetrics;

/*
 * Estimate CL(A) = insertion sort computational load in O(n).
 *
 * Counts the number of "out-of-order" adjacent pairs (inversions):
 *   inversions = sum of (arr[i] > arr[i+1]) for i in [0, n-2]
 *
 * Returns CL = n + inversions * (n - 1).
 *
 * Rationale: insertion sort does n-1 comparisons minimum (fully sorted),
 * and each adjacent inversion roughly corresponds to O(n) additional
 * shift work in the worst case region. This gives:
 *   - Fully sorted:  CL ≈ n  (0 inversions)
 *   - Fully reverse:  CL ≈ n + (n-1)*(n-1) ≈ n^2  (n-1 inversions)
 * which tracks the Theta(n) to Theta(n^2) range of insertion sort.
 */
double estimate_cl(int *arr, int n);

/*
 * Randomized variant of estimate_cl which samples all inversions
 */
double estimate_cl_rand(int *arr, int n);

/*
 * Function pointer used by callers to compute CL.
 * Defaults to estimate_cl; main() sets it to estimate_cl_rand when
 * --cl=rand is passed. New variants can be wired in by assigning here.
 */
typedef double (*estimate_cl_fn)(int *arr, int n);
extern estimate_cl_fn estimate_cl_func;

/*
 * Quantitative imbalance: StdDev(sizes) / (N/P).
 */
double compute_quantitative_imbalance(int *sizes, int p);

/*
 * Qualitative imbalance: StdDev(loads) / avg(loads).
 */
double compute_qualitative_imbalance(double *loads, int p);

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
 * Updates metrics->final_quant_imbalance and final_qual_imbalance.
 */
void synchronous_lb_linear(int rank, int p,
                           SubArray *local_array,
                           int max_rounds,
                           LoadMetrics *metrics);

/*
 * Free SubArray data.
 */
void subarray_free(SubArray *sa);

#endif /* LOAD_BALANCE_H */
