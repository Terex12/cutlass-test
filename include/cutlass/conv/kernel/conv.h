#pragma once
#include <cstdio>
#include "cutlass/cutlass.h"

#include "cutlass/conv/conv.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"


namespace cutlass {
namespace conv {
namespace kernel {
    ///specialize from default_conv.h
    template <
            typename Mma_,                  ///! Threadblock-level mma
            ///Yufan: no need, later remove
            typename Epilogue_,             ///! Epilogue (could be relu or something else)
            typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
            bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
    >
    struct Conv {
        using Mma = Mma_;
        ///Yufan: Epilogue is for output here
        using Epilogue = Epilogue_;
        using OutputOp = typename Epilogue::OutputOp;
        using ThreadblockSwizzle = ThreadblockSwizzle_;
        static bool const kSplitKSerial = SplitKSerial;

        /// Warp count (concept: GemmShape)
        using WarpCount = typename Mma::WarpCount;
        static int const kThreadCount = 32 * WarpCount::kCount;

        ///Yufan: define in device/conv.h
        struct Params {
            ///Yufan: what should be here??
            cutlass::conv::xxx problem_size;
            cutlass::conv::xxx grid_tiled_shape;
            typename Mma::IteratorA::Params params_A;   ///Yufan: ctor Params(Layout const &layout)
            typename Mma::IteratorA::TensorRef ref_A;
            typename Mma::IteratorB::Params params_B;
            typename Mma::IteratorB::TensorRef ref_B;

            typename Epilogue::OutputTileIterator::Params params_D;
            typename Epilogue::OutputTileIterator::TensorRef ref_D;

            typename OutputOp::Params output_op;
            int *semaphore;
            int gemm_k_iterations;
            int gemm_k_size;

            CUTLASS_HOST_DEVICE
            Params() { }

            CUTLASS_HOST_DEVICE
            Params(
                    cutlass::gemm::xxx const & problem_size,
                    cutlass::gemm::xxx const & grid_tiled_shape,
                    typename Mma::IteratorA::TensorRef ref_A,
                    typename Mma::IteratorB::TensorRef ref_B,
                    /*typename Epilogue::OutputTileIterator::TensorRef ref_C,*/
                    typename Epilogue::OutputTileIterator::TensorRef ref_D,
                    typename OutputOp::Params output_op = typename OutputOp::Params(),
                    int *semaphore = nullptr
            ):
                    problem_size(problem_size),
                    grid_tiled_shape(grid_tiled_shape),
                    params_A(ref_A.layout()),
                    ref_A(ref_A),
                    params_B(ref_B.layout()),
                    ref_B(ref_B),
                    /*params_C(ref_C.layout()),
                    ref_C(ref_C),*/
                    params_D(ref_D.layout()),
                    ref_D(ref_D),
                    output_op(output_op),
                    semaphore(semaphore) {
                ///Yufan: ceil (k/ Tile_k)
                int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
                ///Yufan:value is same as total_gemm_k_iterations because grid_tiled_shape.k() == 1
                int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
                ///Yufan: extent of K == problem_size.k()
                gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
            }
        };

        /// Shared memory storage structure
        union SharedStorage {
            typename Mma::SharedStorage main_loop;
            typename Epilogue::SharedStorage epilogue;
        };

        CUTLASS_HOST_DEVICE
        Conv() {}
        /// Executes one CONV
        CUTLASS_DEVICE
        void operator()(Params const &params, SharedStorage &shared_storage) {
            // Compute threadblock location
            ThreadblockSwizzle threadblock_swizzle;
            cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset();

            // Early exit if CTA is out of range
            if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
                params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) { return; }

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{
                    threadblock_tile_offset.m() * Mma::Shape::kM,
                    threadblock_tile_offset.k() * params.gemm_k_size,
            };

            cutlass::MatrixCoord tb_offset_B{
                    threadblock_tile_offset.k() * params.gemm_k_size,
                    threadblock_tile_offset.n() * Mma::Shape::kN
            };

            // Problem size is a function of threadblock index in the K dimension
            int problem_size_k = min(
                    params.problem_size.k(),
                    (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

            // Compute threadblock-scoped matrix multiply-add
            ///Yufan: How many tiles in reduction K
            int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
            ///Yufan:
            printf("Tile M %d, Tile N %d, Tile K %d \n ", Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK);

            // Compute position within threadblock
            int thread_idx = threadIdx.x;

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(
                    params.params_A,  ///Yufan: Precomputed parameters object --> layout
                    params.ref_A.data(),  ///Yufan: Pointer to start of tensor
                    {params.problem_size.m(), problem_size_k}, ///Yufan: Extent of tensor
                    thread_idx,
                    tb_offset_A); ///Yufan: Initial offset of threadblock

            typename Mma::IteratorB iterator_B(
                    params.params_B,
                    params.ref_B.data(),
                    {problem_size_k, params.problem_size.n()},
                    thread_idx,
                    tb_offset_B);

            int warp_idx = threadIdx.x / 32;
            int lane_idx = threadIdx.x % 32;

            //
            // Main loop
            //
            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
            ///Yufan:include/cutlass/gemm/threadblock/mma_pipelined.h-- constructor
            typename Mma::FragmentC accumulators;

            accumulators.clear();

            if (!kSplitKSerial || gemm_k_iterations > 0) {
                // Compute threadblock-scoped matrix multiply-add
                ///Yufan: include/cutlass/gemm/kernel/gemm.h mma_pipline.h
                mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
            }

            //
            // Epilogue
            //

            OutputOp output_op(params.output_op);

            //
            // Masked tile iterators constructed from members
            //

            threadblock_tile_offset = threadblock_swizzle.get_tile_offset();

            //assume identity swizzle
            MatrixCoord threadblock_offset(
                    threadblock_tile_offset.m() * Mma::Shape::kM,
                    threadblock_tile_offset.n() * Mma::Shape::kN
            );

            int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

            ///Yufan: Do I need Epilogue???
            // Tile iterator writing to destination tensor.
            typename Epilogue::OutputTileIterator iterator_D(
                    params.params_D,
                    params.ref_D.data(),
                    params.problem_size.mn(),
                    thread_idx,
                    threadblock_offset
            );

            Epilogue epilogue(
                    shared_storage.epilogue,
                    thread_idx,
                    warp_idx,
                    lane_idx);
            ///Yufan: C-> D not ture in our case. need check epilogue
            // Execute the epilogue operator to update the destination tensor.
            epilogue(output_op, iterator_D, accumulators, iterator_C);
        }//end of operator()
    };//end of conv struct
}
}
}