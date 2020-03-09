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
    template <
            int Tx,
            int Ty,
            int Tf,
            int Tn,
            int Tc,
            int pH, int pW, int sH, int sW, int dH, int dW,
            typename OperatorClass,
            typename ArchTag,
            typename ElementA,
            typename ElementB,
            typename ElementC,
            typename ElementAccumulator
    >
    struct DefaultGemmConfiguration;



    template <
            int Tx,
            int Ty,
            int Tf,
            int Tn,
            int Tc,
            int pH, int pW, int sH, int sW, int dH, int dW,
            typename ArchTag,
            typename ElementA,
            typename ElementB,
            typename ElementC,
            typename ElementAccumulator
            >
    struct DefaultConvConfiguration<
            Tx,
            Ty,
            Tf,
            Tn,
            Tc,
            pH, pW, sH, sW, dH, dW,
            arch::OpClassSimt,
            ArchTag,
            ElementA,
            ElementB,
            ElementC,
            ElementAccumulator> {

        static int const kAlignmentA = 1;
        static int const kAlignmentB = 1;
        ///Yufan, for reduction axis, we can only tile on C
        using ThreadblockShape = ConvShape<Tx*Ty, Tf*Tn, NR*NS*Tc>;
        using ImageShape = ConvShape<Tx+NS-1, Ty+NR-1, Tc>;
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

}
}
}