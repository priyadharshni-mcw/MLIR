#include "numeric/Passes.h"
#include "numeric/numericOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/ParallelCombiningOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Shared templated pattern for add/sub/mul
//===----------------------------------------------------------------------===//

template <typename NumericOp, typename IntOp, typename FloatOp>
struct ConvertNumericBinaryOp : public OpConversionPattern<NumericOp> {
  using OpConversionPattern<NumericOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<NumericOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(NumericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();
    Type elementType = resultType;
    if (auto tensorType = dyn_cast<TensorType>(resultType))
      elementType = tensorType.getElementType();

    if (isa<FloatType>(elementType)) {
      rewriter.template replaceOpWithNewOp<FloatOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
    } else if (isa<IntegerType>(elementType)) {
      rewriter.template replaceOpWithNewOp<IntOp>(op, adaptor.getLhs(),
                                                    adaptor.getRhs());
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    return success();
  }
};


struct ConvertNumericConstantOp
    : public OpConversionPattern<numeric::constantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::constantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};




//===----------------------------------------------------------------------===//
// ArangeOp -> tensor.empty + linalg.generic  
//===----------------------------------------------------------------------===//

struct ConvertNumericArangeOp : public OpConversionPattern<numeric::arangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::arangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value start = adaptor.getStart();
    Value end = adaptor.getEnd();
    Value step = adaptor.getStep();
    Type elementType = start.getType();

    if (!isa<IntegerType>(elementType))
      return rewriter.notifyMatchFailure(op, "only integer arange supported so far");

    // 1. n = ceil((end - start) / step)
    Value diff = rewriter.create<arith::SubIOp>(loc, end, start);
    Value nInt = rewriter.create<arith::CeilDivSIOp>(loc, diff, step);
    Value n = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), nInt);

    // 2. empty output tensor of that dynamic size
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value empty = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{n});

    // 3. fill it: value at position i is start + i * step
    SmallVector<AffineMap> maps = {
        AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())};
    SmallVector<utils::IteratorType> iterators = {utils::IteratorType::parallel};

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultType, /*inputs=*/ValueRange{}, /*outputs=*/ValueRange{empty},
        maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value index = b.create<linalg::IndexOp>(nestedLoc, 0);
          Value indexCast =
              b.create<arith::IndexCastOp>(nestedLoc, elementType, index);
          Value scaled = b.create<arith::MulIOp>(nestedLoc, indexCast, step);
          Value value = b.create<arith::AddIOp>(nestedLoc, start, scaled);
          b.create<linalg::YieldOp>(nestedLoc, value);
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct ConvertNumericToArithPass
    : public PassWrapper<ConvertNumericToArithPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNumericToArithPass)

  StringRef getArgument() const final { return "convert-numeric-to-arith"; }
  StringRef getDescription() const final {
    return "Lower all numeric dialect ops to arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<numeric::addOp, numeric::subOp, numeric::mulOp,
                         numeric::constantOp>();
                         
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                        linalg::LinalgDialect, tensor::TensorDialect>();
    target.addIllegalOp<numeric::addOp, numeric::subOp, numeric::mulOp,
                     numeric::constantOp, numeric::arangeOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertNumericBinaryOp<numeric::addOp, arith::AddIOp,
                                         arith::AddFOp>>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::subOp, arith::SubIOp,
                                         arith::SubFOp>>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::mulOp, arith::MulIOp,
                                         arith::MulFOp>>(context);
    patterns.add<ConvertNumericConstantOp>(context);
    patterns.add<ConvertNumericArangeOp>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                       std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace numeric {
void registerConvertNumericToArithPass() {
  PassRegistration<ConvertNumericToArithPass>();
}
} // namespace numeric