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
// Conversion pattern: numeric.add -> arith.addi / arith.addf
//===----------------------------------------------------------------------===//

struct ConvertNumericAddOp : public OpConversionPattern<numeric::addOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::addOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    Type resultType = op.getResult().getType();
    Type elementType = resultType;
    if (auto tensorType = dyn_cast<TensorType>(resultType))
      elementType = tensorType.getElementType();

    // "Rewrite": replace with the matching arith op.
    if (isa<FloatType>(elementType)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    } else if (isa<IntegerType>(elementType)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    } else {
      return rewriter.notifyMatchFailure(
          op, "numeric.add: unsupported element type for lowering");
    }
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
    return "Lower numeric.add to arith.addi/arith.addf";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<numeric::addOp>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertNumericAddOp>(context);

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