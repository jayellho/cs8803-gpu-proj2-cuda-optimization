# CS-8803 GPU Hardware and Software Project 2
> CUDA kernel code for parallel sorting of integers using the bitonic sort algorithm, optimized for performance on NVIDIA GPUs.

## About
This project is for Georgia Tech's on-campus CS-8803 GPU Hardware and Software course as taken in Fall 2024.

## Description of files in repository
1. `project_2.pdf` - Project description and specifications.
2. `kernel_original.cu` - Provided CUDA kernel framework.
3. `kernel.cu` - Completed CUDA kernel code - <b>RUN THIS</b>.
4. `report.pdf` - Report covering thought process, steps taken and techniques used throughout the project.
5. `grade.py` - Grading script used to evaluate CUDA kernel performance.

## Optimization techniques explored/ used
NOTE: Please read `report.pdf` for detailed coverage of techniques and steps taken.
- Memory pinning.
- CUDA Streams.
- Padding of arrays to powers of 2.
- [Explored, not used] Unified Virtual Addressing.
- cudaMemSet.
- Bitshifting.
- Shared memory and multiple CUDA kernels.
- CPU and GPU interleaving/ overlap.
- Modal with detailed view of properties of selected resource.
- Search widget with search/filter by region/type.

## Results
Speedup of 114 times over CPU using NVIDIA L40s GPU on array of size 10 million, with achieved occupancy of 93.38% and throughput of 701.9 million elements per second (MEPS).
![image](https://github.com/user-attachments/assets/9673af1f-626f-469f-9393-33e870422154)

## Getting Started
### Prerequisites
- Linux/Ubuntu
- nvcc
- NVIDIA GPU
- CUDA

### How to Run
1. Compile the code:
```
nvcc -x cu kernel.cu
```
NOTE: For debugging CUDA code, compile with flags `-g` and `-G`, then use cuda-gdb - this will allow use of `gdb` within kernel code.

2. Run an individual array (example: array size of 10000):
```
./a.out 10000
```

3. Evaluate via throughput and occupancy.
```
# Memory Throughput
ncu --metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
--print-summary per-gpu a.out 10000000

# Achieved Occupancy
ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active
--print-summary per-gpu a.out 10000000
```

4. How to run NSight Compute for profiling
```
ncu ./<program name> # stats for each kernel on stdout
ncu -o profile ./<program name> # output file for NSight Compute GUI
ncu --query-metrics --query-metrics-mode all # lists all existing metrics.
ncu --metrics [metric_1],[metric_2],... ./<program name> # check individual metrics.
```

