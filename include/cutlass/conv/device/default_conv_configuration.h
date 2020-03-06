//
// Created by yufan on 3/5/20.
//

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/conv/conv.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
namespace cutlass {
namespace conv {
namespace device {
    template <
            typename ArchTag,
            typename ElementA,
            typename ElementB,
            typename ElementC,
            typename ElementAccumulator>
    struct DefaultGemmConfiguration<
            arch::OpClassSimt,
            ArchTag,
            ElementA,
            ElementB,
            ElementC,
            ElementAccumulator> {

        static int const kAlignmentA = 1;
        static int const kAlignmentB = 1;
        //using ThreadblockShape = GemmShape<128, 128, 8>;
        using ThreadblockShape = GemmShape<64, 64, 8>;
        using WarpShape = GemmShape<32, 64, 8>;
        using InstructionShape = GemmShape<1, 1, 1>;
        static int const kStages = 2;
//        using EpilogueOutputOp = epilogue::thread::LinearCombination<
//                ElementC,
//                1,
//                ElementAccumulator,
//                ElementAccumulator
//        >;
        using Operator = arch::OpMultiplyAdd;
    };

}
}
}