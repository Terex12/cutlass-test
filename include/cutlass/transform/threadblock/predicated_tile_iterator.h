/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileIterator
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
/// Regular tile iterator using a precomputed control structure to minimize register liveness
/// and integer arithmetic.
///
/// Layout is assumed to be invariant at the time the precomputed "Params" object is constructed.
///
/// Base pointer and tensor extents may be specified at the time the iterator is constructed.
/// Subsequently, they are assumed to be immutable.
///
/// Adding a logical coordinate offset may be performed at the time the iterator is constructed.
/// Subsequent additions to logical coordinate offset may be performed but are relatively expensive.
///
/// Vistitation order is intended to first visit a "residual" tile that may be partially full in
/// both the advance dimension and the steady-state dimension. This is assumed to be the last
/// tile in the iteration sequence. Advancing an iterator that has just been constructed moves to
/// the first tile that is full in the advance dimension and recomputes predicates. Subsequent
/// accesses may be performed without updating internal predicates and are efficient in terms of
/// live register state and pointer arithmetic instructions.
///
/// To be efficient, this assumes the iteraor will be dereferenced and advanced at least once
/// outside any looping structure to minimize integer arithmetic. 
///
/// Acceses out of bounds are safe so long as `clear_mask()` is called prior to dereferencing
/// the iterator.
///
///
/// Example:
///
/// An efficient pipeline structure may be constructed as follows:
///
// template <typename Iterator>
// __global__ void kernel(
//   typename Iterator::Params params, 
//   typename Iterator::Element *ptr,
//   TensorCoord extent) {
//
//   typename Iterator::Fragment fragment;
//
//   TensorCoord threadblock_offset(0, 0);
//
//   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offsets);
//
//
//   fragment = *iter;        // load "residue" tile first
//   ++iter;                  // advance to first "steady state" tile and update internal masks
//
//
//   #pragma unroll
//   for (int i = Remaining - 1; i >= 0; --i) {
//
//     f(fragment);
//
//     if (!i) {
//       iter.clear_mask();   // light-weight operation to clear masks - subsequent loads become NO-OPs.
//     }
//  
//     fragment = *iter;      // load tile during "steady state" phase
//     ++iter;                // advance to next tile - lightweight due to steady-state masks
//   }
// }
//
// void host(TensorView<Element, 2, layout::PitchLinear> view) {
//
//   using Iterator = transform::threadblock::PredicatedTileIterator;
//
//   typename Iterator::Params params(view.layout());
//
//   kernel<Iterator>(params, view.data());
// }
///
///
template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int AccessSize = ThreadMap::kElementsPerAccess
>
class PredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PredicatedTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank,
                             ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_; ///loading tile shape
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator =
      PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                   ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PredicatedTileIterator;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : params_(layout) { }
    
    CUTLASS_HOST_DEVICE
    Params() { }
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { address_iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);
          
          address_iterator_.set_iteration_index(idx);
          auto ptr = (address_iterator_.get() + pointer_offset);

          if (address_iterator_.valid()) {
              frag_ptr[idx] = *ptr;
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          if (address_iterator_.valid()) {
            *(address_iterator_.get() + pointer_offset) = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PredicatedTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_, AccessSize> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:
    
    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) {

    }
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset         ///< Initial offset of threadblock
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.row(), extent.column()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column())
    ) { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PredicatedTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_, AccessSize> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:
    
    CUTLASS_HOST_DEVICE
    Params() { } 

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) {

    };
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset         ///< Initial offset of threadblock
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.column(), extent.row()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row())
    ) { }

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for interleaved data.  It is mapped
/// to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::ColumnMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
      layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                               Shape::kColumn / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row() * kInterleavedK,
                                           extent.column() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.row() * kInterleavedK,
                      threadblock_offset.column() / kInterleavedK)) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator for interleaved-32 data.  It is
/// mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PredicatedTileIterator<Shape_, Element_,
                             layout::RowMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator<
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                               Shape::kRow / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;
  
  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column() * kInterleavedK,
                                           extent.row() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.column() * kInterleavedK,
                      threadblock_offset.row() / kInterleavedK)) {}

  /// Construct a PredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PredicatedTileIterator operator++(int) {
    PredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
