#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

///Yufan: no need, later remove
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/conv/conv.h"
#include "cutlass/conv/kernel/conv.h"
#include "cutlass/conv/kernel/conv_pipelined.h"

#include "cutlass/conv/threadblock/default_mma.h"
#include "cutlass/conv/threadblock/default_mma_core_simt.h"
#include "cutlass/conv/threadblock/threadblock_swizzle.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"


namespace cutlass {
namespace conv {
namespace kernel {
    template <
            /// Element type for A matrix operand
            typename ElementA_,
            /// Layout type for A matrix operand
            typename LayoutA_,
            /// Access granularity of A matrix in units of elements
            int kAlignmentA,
            /// Element type for B matrix operand
            typename ElementB_,
            /// Layout type for B matrix operand
            typename LayoutB_,
            /// Access granularity of B matrix in units of elements
            int kAlignmentB,
            /// Element type for C and D matrix operands
            typename ElementC_,
            /// Layout type for C and D matrix operands
            typename LayoutC_,
            /// Element type for internal accumulation
            typename ElementAccumulator,
            /// Operator class tag
            typename OperatorClass,
            /// Tag indicating architecture to tune for
            typename ArchTag,
            /// Threadblock-level tile size (concept: GemmShape)
            typename ThreadblockShape,
            /// Warp-level tile size (concept: GemmShape)
            typename WarpShape,
            /// Warp-level tile size (concept: GemmShape)
            typename InstructionShape,
            /// Epilogue output operator
            typename EpilogueOutputOp,
            /// Threadblock-level swizzling operator
            typename ThreadblockSwizzle,
            /// Number of stages used in the pipelined mainloop
            int Stages,
            /// If true, kernel is configured to support serial reduction in the
            /// epilogue
            bool SplitKSerial,
            /// Operation performed by GEMM
            typename Operator,
            /// Beta is zero or not
            bool IsBetaZero = false>
    struct DefaultConv;
    
/// Partial specialization for SIMT
    template <
            /// Element type for A matrix operand
            typename ElementA,
            /// Layout type for A matrix operand
            typename LayoutA,
            /// Access granularity of A matrix in units of elements
            int kAlignmentA,
            /// Element type for B matrix operand
            typename ElementB,
            /// Layout type for B matrix operand
            typename LayoutB,
            /// Access granularity of A matrix in units of elements
            int kAlignmentB,
            /// Element type for C and D matrix operands
            typename ElementC,
            /// Element type for internal accumulation
            typename ElementAccumulator,
            /// Tag indicating architecture to tune for
            typename ArchTag,
            /// Threadblock-level tile size (concept: GemmShape)
            typename ThreadblockShape,
            /// Warp-level tile size (concept: GemmShape)
            typename WarpShape,
            /// Epilogue output operator
            typename EpilogueOutputOp,
            /// Threadblock-level swizzling operator
            typename ThreadblockSwizzle,
            /// If true, kernel is configured to support serial reduction in the epilogue
            bool SplitKSerial,
            /// Operation performed by GEMM
            typename Operator
    >
    struct DefaultConv<
            ElementA,
            LayoutA,
            kAlignmentA,
            ElementB,
            LayoutB,
            kAlignmentB,
            ElementC,
            layout::RowMajor,
            ElementAccumulator,
            arch::OpClassSimt,
            ArchTag,
            ThreadblockShape,
            WarpShape,
            GemmShape<1, 1, 1>,
            EpilogueOutputOp,
            ThreadblockSwizzle,
            2 /* Stages*/,
            SplitKSerial,
            Operator> {
            /// Define the threadblock-scoped matrix multiply-accumulate
                using Mma = typename cutlass::conv::threadblock::DefaultMma<
                        ElementA,
                        LayoutA,
                        kAlignmentA,
                        ElementB,
                        LayoutB,
                        kAlignmentB,
                        ElementAccumulator,
                        layout::RowMajor,
                        arch::OpClassSimt,
                        arch::Sm50,
                        ThreadblockShape,
                        WarpShape,
                        GemmShape<1, 1, 1>,
                        2,
                        Operator>::ThreadblockMma;
            
                static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
                static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");
            
                /// Define the epilogue
                using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
                        ThreadblockShape,
                        typename Mma::Operator,
                        EpilogueOutputOp,
                        kEpilogueElementsPerAccess
                >::Epilogue;
            
                /// Define the kernel-level GEMM operator.
                using ConvKernel = kernel::Conv<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
            };
}
}
}