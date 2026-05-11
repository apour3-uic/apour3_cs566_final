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

/* ------------------------------------------------------------------ */
/*  Compute transfer size k from CL difference                        */
/* ------------------------------------------------------------------ */

static int compute_k(double my_load, int my_size,
                     double neighbor_load) {
    if (my_load <= neighbor_load) return 0;
    if (my_size <= 1) return 0;

    double load_diff = my_load - neighbor_load;
    double avg_cl_per_elem = my_load / my_size;
    if (avg_cl_per_elem <= 0.0) return 0;

    int k = (int)(load_diff / (2.0 * avg_cl_per_elem) + 0.5);
    if (k < 1) k = 1;
    if (k > my_size - 1) k = my_size - 1;
    return k;
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Synchronous LB on linear chain                            */
/* ------------------------------------------------------------------ */

/*
 * Pairwise transfer step. Both ranks of an adjacent pair call this with
 * each other as `partner`. They exchange CL, the higher-load side selects
 * its `k` extreme elements (largest if it's the lower-rank, smallest if
 * higher-rank) and sends; the lower-load side computes k=0 and sends an
 * empty message. Both sides receive whatever the partner sent.
 *
 * Tags `tag_lo_send` / `tag_hi_send` must be unique per (round, phase) and
 * differ from each other (so the deadlock-free Send-then-Recv ordering on
 * the lower-rank side and Recv-then-Send on the higher-rank side can match
 * messages by tag).
 */
static void lb_exchange(SubArray *local_array, int rank, int partner,
                        int tag_lo_send, int tag_hi_send) {
    /* Exchange current CL pairwise (no Allgather needed). */
    double my_cl = estimate_cl_func(local_array->data, local_array->size);
    double partner_cl;
    MPI_Sendrecv(&my_cl,      1, MPI_DOUBLE, partner, tag_lo_send,
                 &partner_cl, 1, MPI_DOUBLE, partner, tag_lo_send,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int k = compute_k(my_cl, local_array->size, partner_cl);

    int is_lower = (rank < partner);
    int my_send_tag = is_lower ? tag_lo_send + 1 : tag_hi_send + 1;
    int my_recv_tag = is_lower ? tag_hi_send + 1 : tag_lo_send + 1;

    /* Pick what to send: lower-rank sends largest, higher-rank sends smallest. */
    if (k > 0) {
        if (is_lower) {
            select_k_largest(local_array->data, local_array->size, k);
        } else {
            select_k_smallest(local_array->data, local_array->size, k);
        }
    }

    /* Deadlock-free: lower-rank sends first then recvs; higher-rank reverses. */
    if (is_lower) {
        if (k > 0) {
            MPI_Send(local_array->data + (local_array->size - k),
                     k, MPI_INT, partner, my_send_tag, MPI_COMM_WORLD);
            local_array->size -= k;
        } else {
            MPI_Send(NULL, 0, MPI_INT, partner, my_send_tag, MPI_COMM_WORLD);
        }
    }

    MPI_Status status;
    MPI_Probe(partner, my_recv_tag, MPI_COMM_WORLD, &status);
    int incoming;
    MPI_Get_count(&status, MPI_INT, &incoming);
    if (incoming > 0) {
        subarray_grow(local_array, incoming);
        MPI_Recv(local_array->data + local_array->size, incoming, MPI_INT,
                 partner, my_recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_array->size += incoming;
    } else {
        MPI_Recv(NULL, 0, MPI_INT, partner, my_recv_tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (!is_lower) {
        if (k > 0) {
            MPI_Send(local_array->data, k, MPI_INT,
                     partner, my_send_tag, MPI_COMM_WORLD);
            memmove(local_array->data, local_array->data + k,
                    (local_array->size - k) * sizeof(int));
            local_array->size -= k;
        } else {
            MPI_Send(NULL, 0, MPI_INT, partner, my_send_tag, MPI_COMM_WORLD);
        }
    }
}

void synchronous_lb_linear(int rank, int p,
                           SubArray *local_array,
                           int max_rounds,
                           LoadMetrics *metrics) {
    double all_loads[p];

    for (int round = 0; round < max_rounds; round++) {
        /* Global imbalance check (early termination). This is the only step
         * that needs an Allgather; the per-pair transfer decisions below are
         * purely pairwise. */
        double local_cl = estimate_cl_func(local_array->data, local_array->size);
        MPI_Allgather(&local_cl, 1, MPI_DOUBLE,
                      all_loads, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if (compute_qualitative_imbalance(all_loads, p) < LB_IMBALANCE_THRESH) break;

        /* Phase A: pairs (0,1), (2,3), (4,5), ... */
        int tag_a_lo = 100 + round * 10;
        int tag_a_hi = 200 + round * 10;
        if (rank % 2 == 0 && rank + 1 < p) {
            lb_exchange(local_array, rank, rank + 1, tag_a_lo, tag_a_hi);
        } else if (rank % 2 == 1) {
            lb_exchange(local_array, rank, rank - 1, tag_a_lo, tag_a_hi);
        }

        /* Phase B: pairs (1,2), (3,4), (5,6), ... */
        int tag_b_lo = 300 + round * 10;
        int tag_b_hi = 400 + round * 10;
        if (rank % 2 == 1 && rank + 1 < p) {
            lb_exchange(local_array, rank, rank + 1, tag_b_lo, tag_b_hi);
        } else if (rank % 2 == 0 && rank > 0) {
            lb_exchange(local_array, rank, rank - 1, tag_b_lo, tag_b_hi);
        }
    }

    /* Final imbalance measurement after LB, before sorting. Only rank 0
     * consumes the result, so gather rather than allgather. */
    int all_sizes[p];
    double final_cl = estimate_cl_func(local_array->data, local_array->size);
    MPI_Gather(&local_array->size, 1, MPI_INT,
               all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&final_cl, 1, MPI_DOUBLE,
               all_loads, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        metrics->final_quant_imbalance = compute_quantitative_imbalance(all_sizes, p);
        metrics->final_qual_imbalance = compute_qualitative_imbalance(all_loads, p);
    }
}
