#include "numeric/Passes.h"
#include "numeric/numericOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<numeric::addOp, numeric::subOp, numeric::mulOp,
                         numeric::constantOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertNumericBinaryOp<numeric::addOp, arith::AddIOp,
                                         arith::AddFOp>>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::subOp, arith::SubIOp,
                                         arith::SubFOp>>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::mulOp, arith::MulIOp,
                                         arith::MulFOp>>(context);
    patterns.add<ConvertNumericConstantOp>(context);

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