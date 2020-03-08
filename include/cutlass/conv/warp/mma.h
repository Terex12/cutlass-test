#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace conv {
namespace warp {

/// Query the number of threads per warp
    template <typename OperatorClass>
    struct WarpSize {
        static int const value = 32;
    };

}
}
}
