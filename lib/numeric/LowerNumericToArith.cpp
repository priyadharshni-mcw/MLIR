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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

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



struct ConvertNumericAddcmulOp
    : public OpConversionPattern<numeric::addcmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::addcmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = op.getResult().getType();

    
    auto shiftType = RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = DenseElementsAttr::get(shiftType, static_cast<int8_t>(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    Value valueTensor =
        rewriter.create<tensor::FromElementsOp>(loc, ValueRange{adaptor.getValue()});

    // temp = tensor1 * tensor2
    Value temp = rewriter.create<tosa::MulOp>(
        loc, resultType, adaptor.getTensor1(), adaptor.getTensor2(), shift);
    // scaled = temp * value  (broadcast)
    Value scaled = rewriter.create<tosa::MulOp>(
        loc, resultType, temp, valueTensor, shift);
    // result = input + scaled
    Value result = rewriter.create<tosa::AddOp>(
        loc, resultType, adaptor.getInput(), scaled);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertNumericDivideOp
    : public OpConversionPattern<numeric::divideOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::divideOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = op.getResult().getType();
    auto tensorType = cast<RankedTensorType>(resultType);

    if (isa<IntegerType>(tensorType.getElementType())) {
      // tosa.intdiv handles integer division directly.
      rewriter.replaceOpWithNewOp<tosa::IntDivOp>(
          op, resultType, adaptor.getNumerator(), adaptor.getDenominator());
      return success();
    }

    // Float: 
    Value reciprocal = rewriter.create<tosa::ReciprocalOp>(
        loc, resultType, adaptor.getDenominator());

    auto shiftType = RankedTensorType::get({1}, rewriter.getIntegerType(8));
    auto shiftAttr = DenseElementsAttr::get(shiftType, static_cast<int8_t>(0));
    Value shift = rewriter.create<tosa::ConstOp>(loc, shiftType, shiftAttr);

    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op, resultType, adaptor.getNumerator(), reciprocal, shift);
    return success();
  }
};



struct ConvertNumericAddOp : public OpConversionPattern<numeric::addOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(numeric::addOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op.getResult().getType();

    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
      // Tensor case: route through tosa.add for native broadcast support.
      rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType,
                                                adaptor.getLhs(), adaptor.getRhs());
      return success();
    }

    // Scalar case: tosa.add can't take plain scalars -- keep the arith path.
    if (isa<FloatType>(resultType)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                                  adaptor.getRhs());
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
    return "Lower all numeric dialect ops to arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect, tosa::TosaDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

                            
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                        linalg::LinalgDialect, tensor::TensorDialect, tosa::TosaDialect>();
    target.addIllegalOp<numeric::addOp, numeric::subOp, numeric::mulOp,
                     numeric::constantOp, numeric::arangeOp,
                     numeric::addcmulOp, numeric::divideOp>(); 
              

    RewritePatternSet patterns(context);
    patterns.add<ConvertNumericAddOp>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::subOp, arith::SubIOp,
                                         arith::SubFOp>>(context);
    patterns.add<ConvertNumericBinaryOp<numeric::mulOp, arith::MulIOp,
                                         arith::MulFOp>>(context);
    patterns.add<ConvertNumericConstantOp>(context);
    patterns.add<ConvertNumericArangeOp>(context);
    patterns.add<ConvertNumericAddcmulOp>(context);  
    patterns.add<ConvertNumericDivideOp>(context);

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