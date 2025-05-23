#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void bitonicMergeGlobal(int* d_arr, uint arrayLength, uint size, uint stride, uint dir) {
    uint globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint localIdx = globalIdx & ((arrayLength >> 1) - 1);

    // bitonic merge
    uint ddd = dir ^((localIdx & (size >> 1)) != 0);
    uint pos = (globalIdx << 1) - (globalIdx & (stride - 1));
    if ((d_arr[pos] > d_arr[pos + stride]) == ddd) {
        // Swap values directly using a temporary variable
        uint temp = d_arr[pos];
        d_arr[pos] = d_arr[pos + stride];
        d_arr[pos + stride] = temp;
    }
}

__global__ void bitonicMergeShared(int* d_arr, uint arrayLength, uint size, uint dir) {
    
    // create thread block group.
    cg::thread_block cta = cg::this_thread_block();

    // allocate shared memory.
    __shared__  uint s_arr[1024];

    // calculate the thread index within the block and the global index.
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = (blockIdx.x << 10) | tid;

    // load 2 elements per thread.
    s_arr[tid] = d_arr[global_tid];
    s_arr[tid + 512] = d_arr[global_tid + 512];


    // bitonic merge.
    uint localIdx = (blockIdx.x * blockDim.x + threadIdx.x) & ((arrayLength/2) - 1);
    uint ddd = dir ^ ((localIdx & (size/2)) != 0);

    for (uint stride = 512 ; stride > 0; stride >>= 1) {
        cg::sync(cta);
        uint pos = (tid << 1) - (tid & (stride - 1));
        if ((s_arr[pos] > s_arr[pos + stride]) == ddd) {
            // Swap values directly in shared memory
            uint temp = s_arr[pos];
            s_arr[pos] = s_arr[pos + stride];
            s_arr[pos + stride] = temp;
        }
    }
    cg::sync(cta);
    // Write the sorted data from shared memory back to global memory
    d_arr[global_tid] = s_arr[tid];  // First element
    d_arr[global_tid + 512 ] = s_arr[tid + 512];  // Second element
    // 

}

__global__ void bitonicSortShared(int* d_arr, uint arrayLength, uint dir) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Shared memory storage for the array to be sorted
    __shared__ int s_arr[1024];

    // Calculate thread index and block index
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = (blockIdx.x << 10) + tid; // might need to change 1024.


    // Each thread loads two elements from global memory into shared memory
    s_arr[tid] = d_arr[global_tid];  // First element
    s_arr[tid + 512] = d_arr[global_tid + 512];  // Second element

    // Ensure all threads have loaded their data into shared memory
    cg::sync(cta);

    // Bitonic sort in shared memory
    for (uint size = 2; size <= arrayLength; size <<= 1) {
        // Calculate direction for the current bitonic merge
        uint ddd = dir ^ ((tid & (size >> 1)) != 0);

        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            // Synchronize threads before comparing elements
            cg::sync(cta);

            // Calculate the positions of the elements to compare
            uint pos = (tid << 1) - (tid & (stride - 1));

            int valA = s_arr[pos];
            int valB = s_arr[pos + stride];


            // Comparator logic: Compare and swap if needed
            if ((valA > valB) == ddd) {
                // Swap values
                s_arr[pos] = valB;
                s_arr[pos + stride] = valA;
            }

        }
    }
    {
        for (uint stride = arrayLength >> 1; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = (tid << 1) - (tid & (stride - 1));
            int valA = s_arr[pos];
            int valB = s_arr[pos + stride];

            // Comparator logic: Compare and swap if needed
            if ((valA > valB) == dir) {
                // Swap values
                s_arr[pos] = valB;
                s_arr[pos + stride] = valA;
            }

        }
    }

    // Ensure all sorting is done before writing back to global memory
    cg::sync(cta);

    // Write the sorted data from shared memory back to global memory
    d_arr[global_tid] = s_arr[tid];  // First element
    d_arr[global_tid + 512] = s_arr[tid + 512];  // Second element
}


__global__ void partialBitonicSortShared(int* d_arr, uint arrayLength) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Shared memory storage for the array to be sorted
    __shared__ int s_arr[1024];

    // Calculate thread index and global index
    unsigned int tid = threadIdx.x;
    unsigned int global_tid = (blockIdx.x << 10) + tid;

    // Each thread loads two elements from global memory into shared memory
    s_arr[tid] = d_arr[global_tid];                              // First element
    s_arr[tid + 512] = d_arr[global_tid + 512];    // Second element

    // Ensure all threads have loaded their data into shared memory
    // cg::sync(cta);

    // Bitonic sort in shared memory
    for (uint size = 2; size < 1024; size <<= 1) {
        // Calculate direction for the current bitonic merge
        uint ddd = (tid & (size >> 1)) != 0;

        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            // Synchronize threads before comparing elements
            cg::sync(cta);

            // Calculate the positions of the elements to compare
            uint pos = (tid << 1) - (tid & (stride - 1));

            // Perform the swap with a temporary variable
            if ((s_arr[pos] > s_arr[pos + stride]) == ddd) {
                // Swap values directly using a temporary variable
                int temp = s_arr[pos];
                s_arr[pos] = s_arr[pos + stride];
                s_arr[pos + stride] = temp;
            }
        }
    }

    uint ddd = blockIdx.x & 1;

    // Final sorting phase
    for (uint stride = 512; stride > 0; stride >>= 1) {
        cg::sync(cta);

        uint pos = (tid << 1) - (tid & (stride - 1));

        // Perform the swap with a temporary variable
        if ((s_arr[pos] > s_arr[pos + stride]) == ddd) {
            // Swap values directly using a temporary variable
            int temp = s_arr[pos];
            s_arr[pos] = s_arr[pos + stride];
            s_arr[pos + stride] = temp;
        }
    }

    // Ensure all sorting is done before writing back to global memory
    cg::sync(cta);

    // Write the sorted data from shared memory back to global memory
    d_arr[global_tid] = s_arr[tid];                               // First element
    d_arr[global_tid + 512] = s_arr[tid + 512];    // Second element
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // ======================================================================
    // arCpu contains the input random array
    // arrSortedGpu should contain the sorted array copied from GPU to CPU
    // ======================================================================
    int* arrCpu = (int*)malloc(size * sizeof(int));
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // ======================================================================
    // Transfer data (arr_cpu) to device
    // ======================================================================

    // your code goes here .......

    // calculate next power of 2 with lambda function.
    auto nextPowerOfTwo = [](int num) {
        if (num <= 1) return 1; // Special case for 0 or 1
        num--;
        num |= num >> 1;
        num |= num >> 2;
        num |= num >> 4;
        num |= num >> 8;
        num |= num >> 16;
        return num + 1;
    };

    // Get the next power of 2 greater than or equal to size
    cudaStream_t stream1, stream2, stream3;//, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    int paddedSize = nextPowerOfTwo(size);
    // printf("paddedSize w lambda = %d\n", paddedSize);

    int justPaddingSize = paddedSize - size;
    
    // calc of sizes in bytes.
    size_t paddedSizeInBytes = paddedSize * sizeof(int);
    size_t sizeInBytes = size * sizeof(int);
    size_t justPaddingSizeInBytes = justPaddingSize * sizeof(int);

    // device array shenanigans.
    int* d_arr;
    cudaMalloc((void**)&d_arr, paddedSizeInBytes); // Allocate for padded size

    // create CUDA streams for concurrent processing of the host -> device memory copying.
    cudaMemsetAsync(d_arr, 0, justPaddingSizeInBytes, stream1); // <DATATRF METHOD4: set device arr to zero immed> 

    cudaHostRegister(arrCpu, sizeInBytes, cudaHostRegisterDefault); // consider cudaHostAlloc for both arrcpu and arrSortedGpu

    cudaMemcpyAsync(d_arr + justPaddingSize, arrCpu, sizeInBytes, cudaMemcpyHostToDevice, stream2); // TIP: try using the same stream.
    
    // end of my code for here ........................................

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
    // ======================================================================
    // Perform bitonic sort on GPU
    // ======================================================================

    // your code goes here .......
    
    // unregister arrCpu from pinned memory.
    // #~


    // get metrics about cuda device(s).
    // int device;
    // cudaDeviceProp prop;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    // printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    // printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    // printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    // printf("Number of SMs: %d\n", prop.multiProcessorCount);
    // printf("Max shared mem per block: %d\n", prop.sharedMemPerBlock);
    // printf("Number of threads per warp: %d\n", prop.warpSize);

    uint SHARED_MEM_SIZE = 1024;
    int threadsPerBlock = SHARED_MEM_SIZE >> 1;  
    int numBlocks = max(1, paddedSize / SHARED_MEM_SIZE);
    
    if (paddedSize <= SHARED_MEM_SIZE) {
        // printf("using bitonicSortShared for paddedSize = %d\n", paddedSize);
        // bitonicSortShared<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_arr, paddedSize);
        bitonicSortShared<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_arr, paddedSize, 1);

    }
    else {
        partialBitonicSortShared<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_arr, paddedSize);
        // cudaHostUnregister(arrCpu);

        // Start merging bitonic sequences across blocks
        for (int k = SHARED_MEM_SIZE << 1; k <= paddedSize; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                if (j >= SHARED_MEM_SIZE) {
                    
                    bitonicMergeGlobal<<<paddedSize >> 9, 256, 0, stream3>>>(d_arr, paddedSize, k, j, 1);
                    // printf("using bitonicMergeGlobal for j=%d, k=%d, numBlocks=%d, threads=%d\n", j, k, paddedSize/512, 256);
                }
                else {
                    // printf("using bitonicMergeShared for j=%d, k=%d, numBlocks=%d, threads=%d\n", j, k, numBlocks, threadsPerBlock);
                    bitonicMergeShared<<<numBlocks, threadsPerBlock, 0, stream3>>>(d_arr, paddedSize, k, 1);
                    break;
                }
                
            }
        }
    }
    cudaHostRegister(arrSortedGpu, sizeInBytes, cudaHostRegisterDefault);

    // cudaStreamSynchronize(stream2);

    // end of my code ============================
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

    // ======================================================================
    // Transfer sorted data back to host (copied to arr_sorted_gpu)
    // ======================================================================

    // your code goes here .......   

    cudaMemcpyAsync(arrSortedGpu, d_arr + (paddedSize - size), sizeInBytes, cudaMemcpyDeviceToHost, 0);

    cudaStreamDestroy(stream1); // TIP: do at the part where we free up memory.
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    // cudaHostUnregister(arrSortedGpu);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    

    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    cudaFreeAsync(d_arr, 0);
    free(arrCpu);
    free(arrSortedGpu);
    


    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}

