#pragma once
#include <cstdio>
#include "cutlass/cutlass.h"

#include "cutlass/conv/conv.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"


namespace cutlass {
namespace conv {
namespace kernel {
    template <
            typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
            ///Yufan: no need, later remove
            typename Epilogue_,             ///! Epilogue
            typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
            bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
    >
    struct Conv {
        using Mma = Mma_;
        ///Yufan: no need, later remove
        using Epilogue = Epilogue_;
        using OutputOp = typename Epilogue::OutputOp;
        using ThreadblockSwizzle = ThreadblockSwizzle_;
        static bool const kSplitKSerial = SplitKSerial;

        /// Warp count (concept: GemmShape)
        using WarpCount = typename Mma::WarpCount;
        static int const kThreadCount = 32 * WarpCount::kCount;
    };

}

}

}