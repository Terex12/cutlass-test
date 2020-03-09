#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
    namespace conv {
        namespace warp {

/// Describes the arrangement and configuration of per-lane operations in warp-level matrix multiply 
            template <
                    typename WarpShape_,              ///< shape of the warp in lanes (concept: MatrixShape)
                    typename LaneLayout_,             ///< layout function of lanes
                    typename LaneMmaShape_            ///< size of each lane's thread-level matrix product (concept: GemmShape)
            >
            struct MmaSimtPolicy {
                using WarpShape = WarpShape_;
                using LaneLayout = LaneLayout_;
                using LaneMmaShape = LaneMmaShape_;
                using MmaShape = LaneMmaShape;

                /// Returns a layout functor mapping lane position in the warp to thread ID
                CUTLASS_HOST_DEVICE
                static LaneLayout get_lane_layout() {
                    return LaneLayout::packed({WarpShape::kRow, WarpShape::kColumn});
                }
            };
        }
    }
}