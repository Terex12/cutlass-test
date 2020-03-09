

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeThreadIdxX() {
            return threadIdx.x;
        }

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeThreadIdxY() {
            return threadIdx.y;
        }

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeThreadIdxZ() {
            return threadIdx.z;
        }

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockIdxX() {
            return blockIdx.x;
        }

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockIdxY() {
            return blockIdx.y;
        }

/// Helper to rematerialize block Idx. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockIdxZ() {
            return blockIdx.z;
        }

/// Helper to rematerialize block Dim. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockDimX() {
            return blockDim.x;
        }

/// Helper to rematerialize block Dim. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockDimY() {
            return blockDim.y;
        }

/// Helper to rematerialize block Dim. Reduces register liveness.
        CUTLASS_DEVICE
        int RematerializeBlockDimZ() {
            return blockDim.z;
        }

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
        struct ConvIdentityThreadblockSwizzle {

            CUTLASS_HOST_DEVICE
            ConvIdentityThreadblockSwizzle() {}

            int const kTile = 1;

            /// Returns the shape of the problem in units of logical tiles
            CUTLASS_HOST_DEVICE
            ConvCoord get_tiled_shape(
                    ConvCoord problem_size,
                    ConvCoord tile_size,
                    int split_k_slices) const {

                ///Yufan:  problem_size.m() = NX*NY problem_size.n() = NF*NN
                ///tile_size.m() = Tx*Ty tile_size.n() = Tf*Tn
                return ConvCoord(
                        (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
                        (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
                        split_k_slices);
            }

            /// Computes CUDA grid dimensions given a size in units of logical tiles
            CUTLASS_HOST_DEVICE
            dim3 get_grid_shape(ConvCoord tiled_shape) const {
                return dim3(tiled_shape.m() * kTile, (tiled_shape.n() + kTile - 1) / kTile, tiled_shape.k());
            }

            /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
            CUTLASS_DEVICE
            ConvCoord get_tile_offset() const {

                int block_idx_x = RematerializeBlockIdxX();
                int block_idx_y = RematerializeBlockIdxY();

                /// RematerializeBlockIdxZ == 0 because z dimension length is 1
                return ConvCoord{
                        (block_idx_x / kTile),
                        (block_idx_y * kTile) + (block_idx_x % kTile),
                        RematerializeBlockIdxZ()
                };
            }
        };

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace conv
} // namespace cutlass

