#ifndef LOAD_ANALYSIS_H
#define LOAD_ANALYSIS_H

typedef struct {
    /* Imbalance after pivoting, before LB */
    double initial_quant_imbalance;   /* StdDev(n(i)) / (N/P) */
    double initial_qual_imbalance;    /* StdDev(CL(A(i))) / avg(CL(A(i))) */

    /* Imbalance after LB, before final sort */
    double final_quant_imbalance;
    double final_qual_imbalance;

    /* Performance */
    double speedup;
    double efficiency;
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
 * sizes: array of P subarray sizes.
 * Returns 0.0 if all sizes equal.
 */
double compute_quantitative_imbalance(int *sizes, int p);

/*
 * Qualitative imbalance: StdDev(loads) / avg(loads).
 * loads: array of P CL values.
 * Returns 0.0 if avg(loads) == 0.
 */
double compute_qualitative_imbalance(double *loads, int p);

#endif /* LOAD_ANALYSIS_H */
