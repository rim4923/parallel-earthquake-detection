# Parallel Earthquake Hotspot Detection

## Overview

This project analyzes global earthquake activity (1930–2025) to identify seismic hotspots using a parallel K-Means clustering pipeline.

The dataset contains **273,000+ earthquake records**, which are processed into geographic grid cells and clustered into:

* Quiet regions
* Moderate regions
* High-risk (hotspot) regions

The main goal is to compare performance across different parallel computing paradigms.

---

## Implementations

The K-Means algorithm was implemented and benchmarked using:

* Sequential (baseline)
* Pthreads (manual multithreading)
* OpenMP (shared-memory parallelism)
* MPI (distributed computing)
* CUDA (GPU acceleration)

---

## Data Processing Pipeline

* Load earthquake dataset (CSV)
* Validate geographic coordinates
* Bin events into 1° × 1° grid cells
* Compute features per cell:

  * Earthquake count
  * Average magnitude
* Apply K-Means clustering

Only the **K-Means computation** is parallelized.

---

## Technologies

* C
* CUDA
* MPI
* OpenMP
* Pthreads

---

## Performance Results

All implementations were benchmarked on the same dataset:

* Sequential: 5.093 ms (baseline)
* MPI (1 process): **0.895 ms (~5.7× speedup)** → fastest CPU implementation
* CUDA (block size 256): **2.701 ms (~1.88× speedup)** → best GPU performance
* OpenMP: strong performance at low thread count, degrades with oversubscription
* Pthreads: limited scalability due to synchronization overhead

---

## Key Optimizations

### CUDA (GPU)

* Shared memory usage for aggregation
* Tiling to reduce global memory access
* Memory coalescing (SoA layout)
* Reduced atomic operations

Optimal configuration:

* **256 threads per block** achieved best performance

---

## Project Structure

* `src/` → implementations (Sequential, Pthreads, OpenMP, MPI, CUDA)
* `results/` → benchmarking outputs
* `docs/` → project report

---

## Key Learnings

* Trade-offs between parallel models (CPU vs distributed vs GPU)
* Impact of synchronization and communication overhead
* Importance of memory optimization in GPU programming
* Performance tuning (thread count, block size, workload distribution)

---

## Notes

This project was developed as part of coursework at Lebanese American University.
