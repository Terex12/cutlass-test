#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/conv/conv.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

namespace cutlass {
namespace conv {
namespace threadblock {
    ///Yufan: Specialize it in default_mma_core_*.h:line 393 to define lower level policy
    ///Use this type as Mmacore::MmaPolicy ...
    template <
            /// Warp-level GEMM operator (concept: gemm::warp::Mma)
            typename Operator_,
            /// Padding used for A operand in shared memory (concept: MatrixShape)
            typename SmemPaddingA_,
            /// Padding used for B operand in shared memory (concept: MatrixShape)
            typename SmemPaddingB_,
            /// Number of partitions of K dimension of GEMM
            int PartitionsK = 1>
    struct MmaPolicy {
        /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
        using Operator = Operator_;

        /// Padding used for A operand in shared memory
        using SmemPaddingA = SmemPaddingA_;

        /// Padding used for B operand in shared memory
        using SmemPaddingB = SmemPaddingB_;

        /// Number of partitions of K dimension
        static int const kPartitionsK = PartitionsK;
    };


    /// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
    /// Yufan: inherited by mma_pipelined.
    template <
            /// Yufan: Size of the problem,  basically is TB shape in configuration.h
            typename Shape_,
            /// Policy describing tuning details (concept: MmaPolicy)
            /// Yufan: specialize as MmaCore::MmaPolicy in TB/default_mma.h
            typename Policy_,
            /// Number of stages, 2 as double buffering
            int Stages,
            /// Used for partial specialization
            typename Enable = bool>
    class MmaBase {
    public:
        /// Yufan: Size of the problem,  basically is TB shape in configuration.h
        using Shape = Shape_;
        ///< Policy describing tuning details
        /// Yufan: policy struct is above !!
        using Policy = Policy_;
        /// Warp-level Mma
        using Operator = typename Policy::Operator;
        /// Shape describing the overall GEMM computed from shared memory
        /// by each warp.
        using WarpGemm = typename Policy::Operator::Shape;

        /// Shape describing the number of warps filling the CTA
        using WarpCount = ConvShape<Shape::kM / WarpGemm::kM,
                Shape::kN / WarpGemm::kN,
                Shape::kK / WarpGemm::kK>;

        /// Number of warp-level GEMM oeprations
        static int const kWarpGemmIterations =
                (WarpGemm::kK / Operator::Policy::MmaShape::kK);

        /// Number of stages
        static int const kStages = Stages;

        /// Tensor reference to the A operand
        using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

        /// Tensor reference to the B operand
        using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;


        /// Yufan: SMEM size for Input A and B
        class SharedStorage {
        public:
            /// Shape of the A matrix operand in shared memory
            using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                    Shape::kK * kStages +
                    Policy::SmemPaddingA::kColumn>;
            /// Shape of the B matrix operand in shared memory
            using ShapeB =
            MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;
        public:
            /// Buffer for A operand
            AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;
            /// Buffer for B operand
            AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

        public:
            /// Returns a layout object for the A matrix
            CUTLASS_DEVICE
            static typename Operator::LayoutA LayoutA() {
                return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
            }

            /// Returns a layout object for the B matrix
            CUTLASS_HOST_DEVICE
            static typename Operator::LayoutB LayoutB() {
                return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
            }

            /// Returns a TensorRef to the A operand
            CUTLASS_HOST_DEVICE
                    TensorRefA operand_A_ref() {
                return TensorRefA{operand_A.data(), LayoutA()};
            }

            /// Returns a TensorRef to the B operand
            CUTLASS_HOST_DEVICE
                    TensorRefB operand_B_ref() {
                return TensorRefB{operand_B.data(), LayoutB()};
            }
        };//end nested smem struct

    protected:
        /// Iterator to load a warp-scoped tile of A operand from shared memory
        typename Operator::IteratorA warp_tile_iterator_A_;

        /// Iterator to load a warp-scoped tile of B operand from shared memory
        typename Operator::IteratorB warp_tile_iterator_B_;

    public:
        /// Construct from tensor references
        ///Yufan: construct at MmaPipelined ctor
        CUTLASS_DEVICE
        MmaBase(
                ///< Shared storage needed for internal use by threadblock-scoped GEMM
                SharedStorage &shared_storage,
                ///< ID within the threadblock
                int thread_idx,
                ///< ID of warp
                int warp_idx,
                ///< ID of each thread within a warp
                int lane_idx
        ):
                warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
                warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {

        }
    }; //end of mmabase
}
}
}