#include "load_analysis.h"
#include <math.h>
#include <stdlib.h>

estimate_cl_fn estimate_cl_func = estimate_cl;

// An estimate of the computational load of an array.
double estimate_cl(int *arr, int n) {
    if (n <= 1) return (double)n;

    /* Count adjacent inversions: arr[i] > arr[i+1] */
    long inversions = 0;
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1])
            inversions++;
    }

    /*
     * CL = n + inversions * (n - 1)
     *
     * Fully sorted (0 inversions):   CL = n          ~ Theta(n)
     * Fully reverse (n-1 inversions): CL = n + (n-1)^2 ~ Theta(n^2)
     */
    return (double)n + (double)inversions * (double)(n - 1);
}

// Randomized estimate of computational load: 
// sample all inversions including non-adjacent ones
double estimate_cl_rand(int *arr, int n) {
    if (n <= 1) return (double)n;

    /* Sample inversions: arr[a] > arr[b] */
    long inversions = 0;

    for (int i = 0; i < n - 1; i++) {
	// Choose uniform random increasing pair
	int a = rand() % n;
	int b = rand() % (n-1);
	if (b >= a) {
	    b++;
	} else {
	    int tmp = a;
	    a = b;
	    b = tmp;
	}
	if (arr[a] > arr[b]) {
	    inversions++;
	}
    }

    return (double)n + (double)inversions * (double)(n - 1);
}

double compute_quantitative_imbalance(int *sizes, int p) {
    /* mean = N/P */
    double sum = 0.0;
    for (int i = 0; i < p; i++)
        sum += sizes[i];
    double mean = sum / p;

    if (mean == 0.0) return 0.0;

    /* StdDev */
    double var = 0.0;
    for (int i = 0; i < p; i++) {
        double diff = sizes[i] - mean;
        var += diff * diff;
    }
    double stddev = sqrt(var / p);

    return stddev / mean;
}

double compute_qualitative_imbalance(double *loads, int p) {
    double sum = 0.0;
    for (int i = 0; i < p; i++)
        sum += loads[i];
    double mean = sum / p;

    if (mean == 0.0) return 0.0;

    double var = 0.0;
    for (int i = 0; i < p; i++) {
        double diff = loads[i] - mean;
        var += diff * diff;
    }
    double stddev = sqrt(var / p);

    return stddev / mean;
}
