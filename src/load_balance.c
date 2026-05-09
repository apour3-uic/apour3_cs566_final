#include "load_balance.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define LB_IMBALANCE_THRESH 0.2

/* ------------------------------------------------------------------ */
/*  CL estimation                                                      */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Partial selection: partition-based O(n) average                    */
/* ------------------------------------------------------------------ */

/* Partition arr[lo..hi] around pivot arr[hi]. Returns pivot index. */
static int partition(int *arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            i++;
        }
    }
    int tmp = arr[i]; arr[i] = arr[hi]; arr[hi] = tmp;
    return i;
}

/*
 * Rearrange arr so that the k smallest elements are in arr[0..k-1]
 * (not necessarily sorted). O(n) average via quickselect.
 */
static void select_k_smallest(int *arr, int n, int k) {
    if (k <= 0 || k >= n) return;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = partition(arr, lo, hi);
        if (p == k) break;
        else if (p < k) lo = p + 1;
        else hi = p - 1;
    }
}

/*
 * Rearrange arr so that the k largest elements are in arr[n-k..n-1]
 * (not necessarily sorted). O(n) average via quickselect.
 */
static void select_k_largest(int *arr, int n, int k) {
    if (k <= 0 || k >= n) return;
    /* The k largest = elements at positions [n-k, n-1] after selecting
       the (n-k)-th smallest as partition point */
    int target = n - k;  /* we want arr[target] to be the pivot */
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = partition(arr, lo, hi);
        if (p == target) break;
        else if (p < target) lo = p + 1;
        else hi = p - 1;
    }
}

/* ------------------------------------------------------------------ */
/*  SubArray helpers                                                   */
/* ------------------------------------------------------------------ */

static void subarray_grow(SubArray *sa, int extra) {
    int new_size = sa->size + extra;
    if (new_size > sa->capacity) {
        int new_cap = new_size + new_size / 4 + 64;  /* some headroom */
        sa->data = (int *)realloc(sa->data, new_cap * sizeof(int));
        sa->capacity = new_cap;
    }
}

void subarray_free(SubArray *sa) {
    if (sa->data) {
        free(sa->data);
        sa->data = NULL;
    }
    sa->size = 0;
    sa->capacity = 0;
}

/* ------------------------------------------------------------------ */
/*  Compute transfer size k from CL difference                        */
/* ------------------------------------------------------------------ */

static int compute_k(double my_load, int my_size,
                     double neighbor_load, int max_k) {
    if (my_load <= neighbor_load) return 0;
    if (my_size <= 1) return 0;

    double load_diff = my_load - neighbor_load;
    double avg_cl_per_elem = my_load / my_size;
    if (avg_cl_per_elem <= 0.0) return 0;

    int k = (int)(load_diff / (2.0 * avg_cl_per_elem) + 0.5);
    if (k < 1) k = 1;
    if (k > max_k) k = max_k;
    return k;
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Synchronous LB on linear chain                          */
/* ------------------------------------------------------------------ */

void synchronous_lb_linear(int rank, int p,
                           SubArray *local_array,
                           int max_rounds,
                           TimingBreakdown *timing,
                           LoadMetrics *metrics) {
    double t_start = MPI_Wtime();

    int all_sizes[MAX_P];
    double all_loads[MAX_P];

    for (int round = 0; round < max_rounds; round++) {
        /* Step 1: Gather global load info */
        local_array->comp_load = estimate_cl_func(local_array->data, local_array->size);

        MPI_Allgather(&local_array->size, 1, MPI_INT,
                      all_sizes, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&local_array->comp_load, 1, MPI_DOUBLE,
                      all_loads, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        /* Check early termination */
        double qual_imb = compute_qualitative_imbalance(all_loads, p);
        if (rank == 0) {
            printf("  [LB Round %d] qual_imbalance=%.6f", round + 1, qual_imb);
            for (int i = 0; i < p; i++)
                printf(" n(%d)=%d", i, all_sizes[i]);
            printf("\n");
        }
        if (qual_imb < LB_IMBALANCE_THRESH) {
            if (rank == 0)
                printf("  [LB] Early termination: imbalance %.6f < threshold %.1f\n",
                       qual_imb, LB_IMBALANCE_THRESH);
            break;
        }

        /* Step 2: Compute k for left and right neighbors */
        int k_left = 0, k_right = 0;
        int max_transfer = local_array->size / 4;
        if (max_transfer < 1) max_transfer = 1;

        if (rank > 0) {
            k_left = compute_k(all_loads[rank], all_sizes[rank],
                               all_loads[rank - 1], max_transfer);
        }
        if (rank < p - 1) {
            k_right = compute_k(all_loads[rank], all_sizes[rank],
                                all_loads[rank + 1], max_transfer);
        }

        /* Dual-send guard: k_left + k_right <= size - 1 */
        if (k_left + k_right >= local_array->size) {
            double total_k = k_left + k_right;
            int budget = local_array->size - 1;
            if (budget < 0) budget = 0;
            k_left  = (int)(k_left  * budget / total_k);
            k_right = (int)(k_right * budget / total_k);
            if (k_left + k_right > budget) k_right = budget - k_left;
        }

        /*
         * Step 3: Execute transfers with deadlock-free scheduling.
         *
         * We use a simple paired exchange pattern:
         *   - Phase A: even ranks exchange with right neighbor (rank+1)
         *   - Phase B: even ranks exchange with left neighbor (rank-1)
         *
         * In each phase, the "sender" (higher CL) sends elements and
         * the "receiver" (lower CL) receives. We use MPI_Sendrecv so
         * both sides participate symmetrically.
         */

        /* Phase A: even-odd pairs (0↔1, 2↔3, 4↔5, 6↔7) */
        if (rank % 2 == 0 && rank + 1 < p) {
            /* I'm even, partner is rank+1 */
            int partner = rank + 1;
            if (k_right > 0) {
                /* I send k_right largest to right */
                select_k_largest(local_array->data, local_array->size, k_right);
                MPI_Send(local_array->data + (local_array->size - k_right),
                         k_right, MPI_INT, partner, 100 + round,
                         MPI_COMM_WORLD);
                local_array->size -= k_right;
            } else {
                /* Send 0 marker */
                MPI_Send(NULL, 0, MPI_INT, partner, 100 + round,
                         MPI_COMM_WORLD);
            }
            /* Receive from partner */
            MPI_Status status;
            MPI_Probe(partner, 200 + round, MPI_COMM_WORLD, &status);
            int incoming;
            MPI_Get_count(&status, MPI_INT, &incoming);
            if (incoming > 0) {
                subarray_grow(local_array, incoming);
                MPI_Recv(local_array->data + local_array->size,
                         incoming, MPI_INT, partner, 200 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_array->size += incoming;
            } else {
                MPI_Recv(NULL, 0, MPI_INT, partner, 200 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        } else if (rank % 2 == 1) {
            /* I'm odd, partner is rank-1 */
            int partner = rank - 1;
            /* Receive from partner first */
            MPI_Status status;
            MPI_Probe(partner, 100 + round, MPI_COMM_WORLD, &status);
            int incoming;
            MPI_Get_count(&status, MPI_INT, &incoming);
            if (incoming > 0) {
                subarray_grow(local_array, incoming);
                MPI_Recv(local_array->data + local_array->size,
                         incoming, MPI_INT, partner, 100 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_array->size += incoming;
            } else {
                MPI_Recv(NULL, 0, MPI_INT, partner, 100 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            /* Then send k_left smallest to left */
            if (k_left > 0) {
                select_k_smallest(local_array->data, local_array->size, k_left);
                MPI_Send(local_array->data, k_left, MPI_INT,
                         partner, 200 + round, MPI_COMM_WORLD);
                memmove(local_array->data,
                        local_array->data + k_left,
                        (local_array->size - k_left) * sizeof(int));
                local_array->size -= k_left;
            } else {
                MPI_Send(NULL, 0, MPI_INT, partner, 200 + round,
                         MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /* Phase B: odd-even pairs (1↔2, 3↔4, 5↔6) */
        if (rank % 2 == 1 && rank + 1 < p) {
            /* I'm odd, partner is rank+1 */
            int partner = rank + 1;
            /* Recompute k_right since sizes changed in Phase A */
            local_array->comp_load = estimate_cl_func(local_array->data, local_array->size);
            double partner_load = all_loads[partner]; /* approximate */
            int kr = compute_k(local_array->comp_load, local_array->size,
                               partner_load, local_array->size / 4 > 0 ? local_array->size / 4 : 1);
            if (kr > 0) {
                select_k_largest(local_array->data, local_array->size, kr);
                MPI_Send(local_array->data + (local_array->size - kr),
                         kr, MPI_INT, partner, 300 + round,
                         MPI_COMM_WORLD);
                local_array->size -= kr;
            } else {
                MPI_Send(NULL, 0, MPI_INT, partner, 300 + round,
                         MPI_COMM_WORLD);
            }
            /* Receive from partner */
            MPI_Status status;
            MPI_Probe(partner, 400 + round, MPI_COMM_WORLD, &status);
            int incoming;
            MPI_Get_count(&status, MPI_INT, &incoming);
            if (incoming > 0) {
                subarray_grow(local_array, incoming);
                MPI_Recv(local_array->data + local_array->size,
                         incoming, MPI_INT, partner, 400 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_array->size += incoming;
            } else {
                MPI_Recv(NULL, 0, MPI_INT, partner, 400 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

        } else if (rank % 2 == 0 && rank > 0) {
            /* I'm even (>0), partner is rank-1 */
            int partner = rank - 1;
            /* Receive from partner first */
            MPI_Status status;
            MPI_Probe(partner, 300 + round, MPI_COMM_WORLD, &status);
            int incoming;
            MPI_Get_count(&status, MPI_INT, &incoming);
            if (incoming > 0) {
                subarray_grow(local_array, incoming);
                MPI_Recv(local_array->data + local_array->size,
                         incoming, MPI_INT, partner, 300 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_array->size += incoming;
            } else {
                MPI_Recv(NULL, 0, MPI_INT, partner, 300 + round,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            /* Send k smallest to left */
            local_array->comp_load = estimate_cl_func(local_array->data, local_array->size);
            double partner_load = all_loads[partner]; /* approximate */
            int kl = compute_k(local_array->comp_load, local_array->size,
                               partner_load, local_array->size / 4 > 0 ? local_array->size / 4 : 1);
            if (kl > 0) {
                select_k_smallest(local_array->data, local_array->size, kl);
                MPI_Send(local_array->data, kl, MPI_INT,
                         partner, 400 + round, MPI_COMM_WORLD);
                memmove(local_array->data,
                        local_array->data + kl,
                        (local_array->size - kl) * sizeof(int));
                local_array->size -= kl;
            } else {
                MPI_Send(NULL, 0, MPI_INT, partner, 400 + round,
                         MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Final imbalance measurement after LB, before sorting */
    local_array->comp_load = estimate_cl_func(local_array->data, local_array->size);
    MPI_Allgather(&local_array->size, 1, MPI_INT,
                  all_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_array->comp_load, 1, MPI_DOUBLE,
                  all_loads, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    metrics->final_quant_imbalance = compute_quantitative_imbalance(all_sizes, p);
    metrics->final_qual_imbalance = compute_qualitative_imbalance(all_loads, p);

    if (rank == 0) {
        printf("\n=== Phase 3: Post-LB State ===\n");
        for (int i = 0; i < p; i++)
            printf("  Proc %d: n(i)=%d, CL(A(i))=%.1f\n",
                   i, all_sizes[i], all_loads[i]);
        printf("  Final quantitative imbalance: %.6f\n",
               metrics->final_quant_imbalance);
        printf("  Final qualitative imbalance: %.6f\n",
               metrics->final_qual_imbalance);
    }

    timing->lb_time = MPI_Wtime() - t_start;
}
