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

//#define NR
//#define NS
///Nr,Ns should get from template parameter

namespace cutlass {
namespace conv {
namespace device {
////////////////////////////////////////////////////////////////////////////////

        template <
                typename OperatorClass,
                typename ArchTag,
                typename ElementA,
                typename ElementB,
                typename ElementC,
                typename ElementAccumulator,
                int Tx, int Ty
        >
        struct DefaultConvConfiguration;

////////////////////////////////////////////////////////////////////////////////

        template <
                typename ArchTag,
                typename ElementA,
                typename ElementB,
                typename ElementC,
                typename ElementAccumulator,
                int Tx,
                int Ty>
        struct DefaultConvConfiguration<
                arch::OpClassSimt,
                ArchTag,
                ElementA,
                ElementB,
                ElementC,
                ElementAccumulator, Tx, Ty> {

            static int const kAlignmentA = 1;
            static int const kAlignmentB = 1;
//            static int const Tx = 1;
//            static int const Ty = 1;
            ///Yufan: why 16 is not good?
            using ThreadblockShape = ConvShape<Tx*Ty, 64, 8>;
            using ImageShape = ConvShape<Tx, Ty, 8>;
            using WarpShape = ConvShape<32, 64, 8>;
            using InstructionShape = ConvShape<1, 1, 1>;
            static int const kStages = 2;

            using EpilogueOutputOp = epilogue::thread::LinearCombination<
                    ElementC,
                    1,
                    ElementAccumulator,
                    ElementAccumulator
            >;

            using Operator = arch::OpMultiplyAdd;
        };

////////////////////////////////////////////////////////////////////////////////

//        template <
//                typename ArchTag,
//                typename ElementC>
//        struct DefaultConvConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {
//
//            static int const kAlignmentA = 4;
//            static int const kAlignmentB = 4;
//            using ThreadblockShape = ConvShape<128, 128, 32>;
//            using WarpShape = ConvShape<32, 64, 32>;
//            using InstructionShape = ConvShape<1, 1, 4>;
//            static int const kStages = 2;
//
//            using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
//                    ElementC,
//                    1,
//                    int32_t,
//                    float
//            >;
//
//            using Operator = arch::OpMultiplyAdd;
//        };


}
}
}