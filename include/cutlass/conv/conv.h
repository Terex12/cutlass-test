#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

namespace cutlass {
namespace conv {
    enum class Operand {
        kA, /// A multiplicand  Input
        kB, /// B multiplicand  Kernel
        kC  /// Destination accumulator Output
    };
    /// Shape of a matrix multiply-add operation
    template <
            /// Rows of matrix product (X*Y)
            int M = 1,
            /// Columns of matrix product (F)
            int N = 1,
            /// Reduction (R*S*C)
            int K = 1
    >
    struct ConvShape {
        static int const kM = M;
        static int const kN = N;
        static int const kK = K;

        static int const kMN = M * N;
        static int const kMK = M * K;
        static int const kKN = N * K;
        static int const kMNK = M * N * K;

        static int const kCount = kMNK;
        
        /// Returns a Coord object
        CUTLASS_HOST_DEVICE
        static Coord<3> toCoord() {
            return make_Coord(kM, kN, kK);
        }
    };

    ///Yufan: problem size should consider 5 ele or 3
    struct ConvCoord : public Coord<3, int> {

        /// Integer-valued index
        typedef int Index;

        /// Base type is a Coord of rank=4
        typedef Coord<3, Index> Base;

        /// Conv M dimension - rows of the output C matrix
        static int const kM = 0;

        /// Conv N dimension - columns of the output C matrix
        static int const kN = 1;

        /// Conv K dimension - inner dimension of the GEMM problem
        static int const kK = 2;

        /// Default ctor
        CUTLASS_HOST_DEVICE
        ConvCoord() {}

        /// Constructs from Coord<3> and a batch
        CUTLASS_HOST_DEVICE
        ConvCoord(Coord<3, Index> const &coord): Base(make_Coord(coord[0], coord[1], coord[2])) { }

        /// Helper to construct from a K, N, M, batch variables
        CUTLASS_HOST_DEVICE
        ConvCoord(Index m, Index n, Index k): Base(make_Coord(m, n, k)) { }

        /// Returns the GEMM M coordinate
        CUTLASS_HOST_DEVICE
                Index const & m() const { return this->at(kM); }

        /// Returns reference to the GEMM M coordinate
        CUTLASS_HOST_DEVICE
                Index & m() { return this->at(kM); }

        /// Returns the GEMM N coordinate
        CUTLASS_HOST_DEVICE
                Index const & n() const { return this->at(kN); }

        /// Returns reference to the GEMM N coordinate
        CUTLASS_HOST_DEVICE
                Index & n() { return this->at(kN); }

        /// Returns the GEMM K coordinate
        CUTLASS_HOST_DEVICE
                Index const & k() const { return this->at(kK); }

        /// Returns reference to the GEMM K coordinate
        CUTLASS_HOST_DEVICE
                Index & k() { return this->at(kK); }

        /// Obtains a Coord<3> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<3> mnk() const {
            return make_Coord(m(), n(), k());
        }

        /// Obtains a Coord<3> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<3> knm() const {
            return make_Coord(k(), n(), m());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> nm() const {
            return make_Coord(n(), m());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> mn() const {
            return make_Coord(m(), n());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> mk() const {
            return make_Coord(m(), k());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> km() const {
            return make_Coord(k(), m());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> nk() const {
            return make_Coord(n(), k());
        }

        /// Obtains a Coord<2> from ConvCoord
        CUTLASS_HOST_DEVICE
                Coord<2> kn() const {
            return make_Coord(k(), n());
        }

        //
        // Coord operators
        //

        /// Element-wise addition
        CUTLASS_HOST_DEVICE
                ConvCoord operator+(Base const& b) const {
            return ConvCoord(Base::operator+(b));
        }

        /// Element-wise subtraction
        CUTLASS_HOST_DEVICE
                ConvCoord operator-(Base const& b) const {
            return ConvCoord(Base::operator-(b));
        }

        /// Element-wise multiplication
        CUTLASS_HOST_DEVICE
                ConvCoord operator*(Base const& b) const {
            return ConvCoord(Base::operator*(b));
        }

        /// Element-wise division
        CUTLASS_HOST_DEVICE
                ConvCoord operator/(Base const& b) const {
            return ConvCoord(Base::operator/(b));
        }

        /// In-place addition
        CUTLASS_HOST_DEVICE
                ConvCoord& operator+=(Base const& b) {
            Base::operator+=(b);
            return *this;
        }

        /// In-place subtraction
        CUTLASS_HOST_DEVICE
                ConvCoord& operator-=(Base const& b) {
            Base::operator-=(b);
            return *this;
        }

        /// In-place multiplication
        CUTLASS_HOST_DEVICE
                ConvCoord& operator*=(Base const& b) {
            Base::operator*=(b);
            return *this;
        }

        /// In-place division
        CUTLASS_HOST_DEVICE
                ConvCoord& operator/=(Base const& b) {
            Base::operator/=(b);
            return *this;
        }
    };

    /// Auxiliary info like padding, stride, dilation
    struct AuxiliaryCoord : public Coord<6, int>{
        /// Integer-valued index
        typedef int Index;

        /// Base type is a Coord of rank=4
        typedef Coord<6, Index> Base;

        CUTLASS_HOST_DEVICE
        AuxiliaryCoord() : Base(make_Coord(0, 0 , 1, 1, 0, 0)) { }

        CUTLASS_HOST_DEVICE
        AuxiliaryCoord(Coord<6, Index> const &coord) : Base(make_Coord(coord[0], coord[1], coord[2], coord[3], coord[4], coord[5])) { }

        CUTLASS_HOST_DEVICE
        AuxiliaryCoord(Index pH, Index pW, Index sH, Index sW, Index dH, Index dW): Base(make_Coord(pH, pW, sH, sW, dH, dW)) { }

    };

}
}
