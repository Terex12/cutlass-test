
// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::conv::device::Conv, the generic Conv computation template class.
#include "cutlass/conv/device/conv.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSconvNN(
        int M,
        int N,
        int K,
        float alpha,
        float const *A,
        int lda0, lda1, lda2,
        float const *B,
        int ldb0, ldb1, ldb2,
        float beta,
        float *C,
        int ldc0, ldc1, ldc2) {

    // Define type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/conv/device/default_conv_configuration.h` for more details.
    //
    // To view the full conv device API interface, see `cutlass/conv/device/conv.h`

    using TensorNCHW = cutlass::layout::TensorNCHW;

    using CutlassConv = cutlass::conv::device::Conv<float,        // Data-type of A matrix
            TensorNCHW,  // Layout of A matrix
            float,        // Data-type of B matrix
            TensorNCHW,  // Layout of B matrix
            float,        // Data-type of C matrix
            TensorNCHW>; // Layout of C matrix

    /// Stride vector
    using Stride = Coord<4, Index>;

    /// Construct stride for tensors
    Stride strideA = make_Coord(lda0, lda1, lda2);
    Stride strideB = make_Coord(ldb0, ldb1, ldb2);
    Stride strideC = make_Coord(ldc0, ldc1, ldc2);

    // Define a CUTLASS GEMM type
    CutlassConv conv_operator;

    ///Creat the arguments struct from input
    CutlassConv::Arguments args({M, N, K},  // Conv Problem dimensions
                                {},          // padding ...
                                {A, strideA},    // Tensor-ref for source matrix A
                                {B, strideB},    // Tensor-ref for source matrix B
                                {C, strideC},    // Tensor-ref for source matrix C
                                {alpha, beta}); // Scalars used in the Epilogue


    /// Launch the CUTLASS GEMM kernel.
    cutlass::Status status = conv_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}

///Yufan: check it later to modify
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
        float *matrix,
        int ldm,
        int rows,
        int columns,
        int seed = 0) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i * ldm + j;

        // Generate arbitrary elements.
        int const k = 16807;
        int const m = 16;
        float value = float(((offset + seed) * k % m) - m / 2);

        matrix[offset] = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.

cudaError_t InitializeMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {

    dim3 block(16, 16);
    dim3 grid(
            (rows + block.x - 1) / block.x,
            (columns + block.y - 1) / block.y
    );

    InitializeMatrix_kernel << < grid, block >> > (matrix, ldm, rows, columns, seed);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int ldm, int rows, int columns, int seed = 0) {
    cudaError_t result;

    size_t sizeof_matrix = sizeof(float) * ldm * rows;

    // Allocate device memory.
    result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Clear the allocation.
    result = cudaMemset(*matrix, 0, sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Initialize matrix elements to arbitrary small integers.
    result = InitializeMatrix(*matrix, ldm, rows, columns, seed);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceConv_kernel(
        int M,
        int N,
        int K,
        float alpha,
        float const *A,
        int lda,
        float const *B,
        int ldb,
        float beta,
        float *C,
        int ldc) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
            //accumulator += A[i + k * lda] * B[k + j * ldb];
            accumulator += A[i * lda + k] * B[k * ldb + j];
        }

        //C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
        C[i * ldc + j] = alpha * accumulator + beta * C[i * ldc + j];
    }
}

/// Reference GEMM computation.
cudaError_t ReferenceConv(
        int M,
        int N,
        int K,
        float alpha,
        float const *A,
        int lda,
        float const *B,
        int ldb,
        float beta,
        float *C,
        int ldc) {

    dim3 block(16, 16);
    dim3 grid(
            (M + block.x - 1) / block.x,
            (N + block.y - 1) / block.y
    );

    ReferenceConv_kernel << < grid, block >> > (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassConv(int M, int N, int K, float alpha, float beta) {
    cudaError_t result;

    //
    // Define several matrices to be used as operands to GEMM kernels.
    //

    // Compute leading dimensions for each matrix.
//  int lda = M;
//  int ldb = K;
//  int ldc = M;


    int lda = K;
    int ldb = N;
    int ldc = N;

    // Compute size in bytes of the C matrix.
    //size_t sizeof_C = sizeof(float) * ldc * N;
    size_t sizeof_C = sizeof(float) * ldc * M;

    // Define pointers to matrices in GPU device memory.
    float *A;
    float *B;
    float *C_cutlass;
    float *C_reference;

    //
    // Allocate matrices in GPU device memory with arbitrary seeds.
    //

    result = AllocateMatrix(&A, lda, M, K, 0);

    if (result != cudaSuccess) {
        return result;
    }

    result = AllocateMatrix(&B, ldb, K, N, 17);

    if (result != cudaSuccess) {
        cudaFree(A);
        return result;
    }

    result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return result;
    }

    result = AllocateMatrix(&C_reference, ldc, M, N, 101);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return result;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Launch CUTLASS GEMM.
    //

    result = CutlassSconvNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Verify.
    //

    // Launch reference GEMM
    result = ReferenceConv(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    if (result != cudaSuccess) {
        std::cerr << "Reference GEMM kernel failed: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    // Copy to host and verify equivalence.
    std::vector<float> host_cutlass(ldc * M, 0);
    std::vector<float> host_reference(ldc * M, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy Reference GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return result;
    }

    //
    // Free device memory allocations.
    //

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    //
    // Test for bit equivalence of results.
    //

//  for(std::vector<float>::iterator it = host_cutlass.begin(); it != host_cutlass.end(); it++){
//      printf("value = %0.2f\n", *it);
//  }
    if (host_cutlass != host_reference) {
        std::cerr << "CUTLASS results incorrect." << std::endl;

        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_conv example.
//
// usage:
//
//   00_basic_conv <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //

    // GEMM problem dimensions.
    int problem[3] = {128, 64, 32};

    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }

    // Scalars used for linear scaling the result of the matrix product.
    float scalars[2] = {1, 0};

    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(arg[i]);
        ss >> scalars[i - 4];
    }

    //
    // Run the CUTLASS GEMM test.
    //

    cudaError_t result = TestCutlassConv(
            problem[0],     // GEMM M dimension
            problem[1],     // GEMM N dimension
            problem[2],     // GEMM K dimension
            scalars[0],     // alpha
            scalars[1]      // beta
    );

    if (result == cudaSuccess) {
        std::cout << "Passed." << std::endl;
    }

    // Exit.
    return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
