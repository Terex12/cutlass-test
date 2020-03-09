#pragma once
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/conv/threadblock/threadblock_swizzle.h"
#include "cutlass/conv/kernel/conv.h"

#include "cutlass/conv/kernel/default_conv.h"
#include "cutlass/conv/device/default_conv_configuration.h"

namespace cutlass {
namespace conv {
namespace device {
    template <
            int Tx,
            int Ty,
            int Tf,
            int Tn,
            int Tc,
            int pH, int pW, int sH, int sW, int dH = 1, int dW = 1
            /// Element type for A matrix operand
            typename ElementA_,
            /// Layout type for A matrix operand
            typename LayoutA_,
            /// Element type for B matrix operand
            typename ElementB_,
            /// Layout type for B matrix operand
            typename LayoutB_,
            /// Element type for C and D matrix operands
            typename ElementC_,
            /// Layout type for C and D matrix operands
            typename LayoutC_,
            /// Element type for internal accumulation
            typename ElementAccumulator_ = ElementC_,
            /// Operator class tag
            typename OperatorClass_ = arch::OpClassSimt,
            /// Tag indicating architecture to tune for
            typename ArchTag_ = arch::Sm60,
            ///Yufan: define image input tile shape
            typename ImageShape_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::ImageShape,
            /// Threadblock-level tile size (concept: ConvShape)
            typename ThreadblockShape_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::ThreadblockShape,
            /// Warp-level tile size (concept: ConvShape)
            typename WarpShape_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::WarpShape,
            /// Instruction-level tile size (concept: ConvShape)
            typename InstructionShape_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::InstructionShape,
            /// Epilogue output operator
            ///Yufan: no need, later remove
            typename EpilogueOutputOp_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::EpilogueOutputOp,
            /// Threadblock-level swizzling operator
            typename ThreadblockSwizzle_ = threadblock::ConvIdentityThreadblockSwizzle,
            /// Number of stages used in the pipelined mainloop
            int Stages =
            typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::kStages,
            /// Access granularity of A matrix in units of elements
            int AlignmentA =
            typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::kAlignmentA,
            /// Access granularity of B matrix in units of elements
            int AlignmentB =
            typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::kAlignmentB,
            /// If true, kernel supports split-K with serial reduction
            bool SplitKSerial = false,
            /// Operation performed by GEMM
            typename Operator_ = typename DefaultConvConfiguration<
                    OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
                    ElementAccumulator_>::Operator,
            /// Whether Beta is zero or not
            ///Yufan: no need, later remove
            bool IsBetaZero = false>
    class Conv {
    public:
        using ElementA = ElementA_;
        using LayoutA = LayoutA_;
        using TensorRefA = TensorRef<ElementA const, LayoutA>; //include/cutlass/tensor_ref.h:146
        using ElementB = ElementB_;
        using LayoutB = LayoutB_;
        using TensorRefB = TensorRef<ElementB const, LayoutB>;
        using ElementC = ElementC_;
        using LayoutC = LayoutC_;
        /*using TensorRefC = TensorRef<ElementC const, LayoutC>;*/
        using TensorRefD = TensorRef<ElementC, LayoutC>;
        using ElementAccumulator = ElementAccumulator_;
        using OperatorClass = OperatorClass_;
        using ArchTag = ArchTag_;
        /// Input Image Shape
        using ImageShape = ImageShape_;
        using ThreadblockShape = ThreadblockShape_;
        using WarpShape = WarpShape_;
        using InstructionShape = InstructionShape_;
        using EpilogueOutputOp = EpilogueOutputOp_;
        using ThreadblockSwizzle = ThreadblockSwizzle_;
        using Operator = Operator_;
        static int const kStages = Stages;
        static int const kAlignmentA = AlignmentA;
        static int const kAlignmentB = AlignmentB;
        static int const kAlignmentC = EpilogueOutputOp::kCount;
        static bool const kSplitKSerial = SplitKSerial;
        ///Yufan: no need, later remove
        static bool const kIsBetaZero = IsBetaZero;

        ///Yufan: specialize default_conv.h
        using ConvKernel = typename kernel::DefaultConv<
                ElementA,
                LayoutA,
                kAlignmentA,
                ElementB,
                LayoutB,
                kAlignmentB,
                ElementC,  ///Yufan: a bad naming, D is using C's layout and datatype
                LayoutC,
                ElementAccumulator,
                OperatorClass,
                ArchTag,
                //ImageShape,
                ThreadblockShape,
                WarpShape,
                InstructionShape,
                EpilogueOutputOp,
                ThreadblockSwizzle,
                kStages,
                kSplitKSerial,
                Operator,
                kIsBetaZero
        >::ConvKernel;

        /// Argument structure
        struct Arguments {
            ConvCoord problem_size;
            AuxiliaryCoord aux;
            TensorRef<ElementA const, LayoutA> ref_A;
            TensorRef<ElementB const, LayoutB> ref_B;
            TensorRef<ElementC, LayoutC> ref_C;
            typename EpilogueOutputOp::Params epilogue;
            int split_k_slices;

            /// Default ctor
            CUTLASS_HOST_DEVICE
            Arguments(): problem_size(0, 0, 0), split_k_slices(1) {}

            /// Constructs an Arguments structure
            CUTLASS_HOST_DEVICE
            Arguments(
                    ConvCoord problem_size_,
                    AuxiliaryCoord aux_,  ///Yufan: padding, stride, dilation
                    TensorRef<ElementA const, LayoutA> ref_A_,      ///Yufan: layout have more than 1 stride  TensorNHWC in tensor.h
                    TensorRef<ElementB const, LayoutB> ref_B_,
                    TensorRef<ElementC, LayoutC> ref_C_,
                    typename EpilogueOutputOp::Params epilogue_ =
                    typename EpilogueOutputOp::Params(),
                    int split_k_slices = 1
            ):
                    problem_size(problem_size_),
                    aux(aux_);
                    ref_A(ref_A_),
                    ref_B(ref_B_),
                    ref_C(ref_C_),
                    epilogue(epilogue_),
                    split_k_slices(split_k_slices) {}
        };

    private:
        /// Kernel parameters object
        typename ConvKernel::Params params_;
    public:
        Conv() {}

        /// Initializes GEMM state from arguments.
        Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
            // Determine grid shape
            ThreadblockSwizzle threadblock_swizzle;

            ///Yufan: dim3 grid config; We need to change M, N, K based on convolution cal
            ///I think it should map to output dimension
            cutlass::conv::ConvCoord grid_shape = threadblock_swizzle.get_tiled_shape(
                    args.problem_size,
                    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                    args.split_k_slices);

            ///Yufan: assume no splitK
            if (args.split_k_slices > 1) {
                return Status::kErrorInvalidProblem;
            }

            // Initialize the Params structure
            ///Yufan: declare in kernel/conv.h
            params_ = typename ConvKernel::Params{
                    args.problem_size,
                    grid_shape,
                    args.ref_A.non_const_ref(),
                    args.ref_B.non_const_ref(),
                    args.ref_C,
                    args.epilogue,
                    static_cast<int *>(workspace)
            };
            return Status::kSuccess;
        }

        /// Runs the kernel using initialized state.
        Status run(cudaStream_t stream = nullptr) {
            ThreadblockSwizzle threadblock_swizzle;

            dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
            dim3 block(ConvKernel::kThreadCount, 1, 1);
            ///Yufan: print launch configuration
            printf("grid dim %d, %d, %d \n", params_.grid_tiled_shape.n(), params_.grid_tiled_shape.m(), params_.grid_tiled_shape.k());
            printf("block dim %d, %d, %d \n", ConvKernel::kThreadCount, 1, 1);

            cudaError_t result;

            int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
            if (smem_size >= (48 << 10)) {  ///Yufan: Why 48K for Smem?
                result = cudaFuncSetAttribute(Kernel<ConvKernel>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              smem_size);

                if (result != cudaSuccess) {
                    return Status::kErrorInternal;
                }

                result = cudaFuncSetAttribute(
                        Kernel<ConvKernel>,
                        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

                if (result != cudaSuccess) {
                    return Status::kErrorInternal;
                }
            }

            cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);
            result = cudaGetLastError();

            return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
        }

        /// Runs the kernel using initialized state.
        Status operator()(cudaStream_t stream = nullptr) {
            return run(stream);
        }

        /// Runs the kernel using initialized state.
        Status operator()( Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
            Status status = initialize(args, workspace);
            if (status == Status::kSuccess) { status = run(stream);}
            return status;
        }
    };
}
}
}