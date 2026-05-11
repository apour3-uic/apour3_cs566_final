# Parallel Quick-Sort with Load Balancing

ECE/CS 566 Spring 2026 — Parallel Processing Final Project

Parallel sorting system combining quicksort-style pivot partitioning with insertion sort and adjacent-only load balancing on an 8-processor 3D hypercube (MPI).

## Overview

The program sorts a 100K-element integer array across P processors (P = 1, 2, 4, or 8) in six phases:

1. **Pivot Distribution** — Proc 0 reads input, partitions into P buckets using external pivots, sends to each proc
2. **Load Estimation** — Each proc estimates insertion sort cost (CL) on its unsorted subarray in O(n)
3. **Load Balancing** (optional) — Synchronous element exchanges between rank-adjacent processors on a linear chain, driven by CL estimates
4. **Local Sorting** — Each proc sorts its subarray with insertion sort
5. **Verification** — Gather min/max/size to verify PSOR (Partial Sort Order) and element completeness
6. **Output** — Token-passing file write + statistics

Two execution modes are supported via `--mode=no_lb` and `--mode=with_lb`, run as separate experiments for fair timing comparison.

## Project Structure

```
src/
├── main.c              # MPI init, CLI parsing, phase dispatch
├── config.h            # Constants (MAX_N, MAX_P, LB_MAX_ROUNDS, threshold)
├── sorting.c/h         # insertion_sort, partition_by_pivots (no MPI)
├── parallel_sort.c/h   # Phase 1 (distribute_by_pivots), Phase 4 (sort_local)
├── load_analysis.c/h   # O(n) CL estimation, imbalance metrics
├── load_balance.c/h    # Phase 3: synchronous LB on linear chain
└── output.c/h          # Phase 5 (verify_psor_final), Phase 6 (token-pass write, stats)

scripts/
├── gen_inputs.sh            # Generates input/pivot files for the experiment grid
├── run_full_experiment.sh   # Full experiment suite (Slurm batch)
├── run_quick_test.sh        # Quick interactive test
└── run_rand_test.sh         # Compares default vs random CL estimator

Makefile                # Build with mpicc
```

## Prerequisites

- Access to the UIC Lakeshore HPC cluster
- Account: `csece566_dutt`
- OpenMPI module

## Setup on the Cluster

```bash
# SSH into Lakeshore
ssh <netid>@lakeshore.acer.uic.edu

# Navigate to working directory
cd /home/<netid>/csece566_dutt_link/<netid>/final

# Copy all project files here (src/, scripts/, Makefile, input files)
# Ensure these input files are present:
#   A_vector_100000_8_2.txt       pivot_vector_100000_8_2.txt
#   A_vector_100000_8_8.txt       pivot_vector_100000_8_8.txt

# Load MPI and build
module load OpenMPI
make clean && make
mkdir -p outputs
```

## Running Experiments

### Option 1: Batch Submission (recommended)

Runs the full suite — N={100k, 1M} x P={8,16,32,64,128} x s={2, P} x {no_lb, with_lb}:

```bash
sbatch scripts/run_full_experiment.sh
```

Monitor with `squeue -u $USER`. Results go to `outputs/psort_full_<jobid>.log`.

### Option 2: Interactive Session

```bash
salloc --job-name "ParSort_Test" --cpus-per-task 1 --ntasks=8 \
       --time 00:10:00 --account=csece566_dutt

module load OpenMPI
make clean && make
mkdir -p outputs

# Run a single experiment
srun --mpi=pmix -n 8 ./parallel_sort \
    A_vector_100000_8_2.txt pivot_vector_100000_8_2.txt --mode=with_lb

# Or run the quick test (P=8, both modes, skewed pivots)
bash scripts/run_quick_test.sh
```

## Usage

```
mpirun -np <P> ./parallel_sort <input_file> <pivot_file> [--mode=no_lb|with_lb]
```

| Argument | Description |
|----------|-------------|
| `P` | Number of processors: 1, 2, 4, or 8 |
| `input_file` | Comma-separated integer array (e.g., `A_vector_100000_8_2.txt`) |
| `pivot_file` | P comma-separated pivot boundaries (e.g., `pivot_vector_100000_8_2.txt`) |
| `--mode=no_lb` | Skip load balancing (default) |
| `--mode=with_lb` | Enable load balancing |

On Lakeshore, use `srun --mpi=pmix -n <P>` instead of `mpirun -np <P>`.

## Output

Each run produces two files in `outputs/`:

- `sorted_<mode>_P<n>.txt` — Sorted subarrays per rank
- `stats_<mode>_P<n>.txt` — Timing breakdown, imbalance metrics, speedup/efficiency

Console output includes per-phase diagnostics: PSOR verification, LB round progress, sort times per rank, and final statistics.

## Input Files

Two test inputs are provided with different pivot distributions:

| Input | Distribution | Purpose |
|-------|-------------|---------|
| `*_8_2.txt` | Halving pattern (50K/25K/12.5K/...) | Extreme imbalance — stress-tests LB |
| `*_8_8.txt` | Varied sizes | Moderate imbalance |

Pivots are externally provided (not computed at runtime) to ensure reproducibility.

## Configuration

Edit `src/config.h` to adjust:

```c
#define MAX_N             100000   // Input size
#define MAX_P             8        // Max processors
#define LB_MAX_ROUNDS     10       // Max load balancing iterations
#define LB_IMBALANCE_THRESH 0.2    // Early termination threshold
```

Rebuild after changes: `make clean && make`.
