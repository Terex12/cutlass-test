#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/conv/conv.h"
#include "cutlass/arch/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: conv::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Concept: arch::OpMultiplyAdd or arch::Mma<>
  typename Operator = arch::OpMultiplyAdd,
  /// Used for partial specialization
  typename Enable = bool
>
struct Mma;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Overloads specialized for existing architectures
//

//#include "cutlass/conv/thread/mma_sm50.h"
#include "cutlass/conv/thread/mma_sm60.h"
//#include "cutlass/conv/thread/mma_sm61.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
