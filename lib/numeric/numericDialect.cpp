#include "numeric/numericDialect.h"
#include "numeric/numericOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace numeric;

#include "numeric/numericOpsDialect.cpp.inc"
#include "mlir/Dialect/CommonFolders.h"

void numericDialect::initialize() {
  addOperations <
#define GET_OP_LIST
#include "numeric/numericOps.cpp.inc"
      >();
}

Operation *numericDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  auto typed = dyn_cast<TypedAttr>(value);
  if (!typed || typed.getType() != type)
    return nullptr;
  return builder.create<constantOp>(loc, type, typed);
}


static bool isIntZero(Attribute attr) {
  if (auto i = dyn_cast_if_present<IntegerAttr>(attr))
    return i.getValue().isZero();
  if (auto d = dyn_cast_if_present<DenseIntElementsAttr>(attr))
    return d.isSplat() && d.getSplatValue<APInt>().isZero();
  return false;
}

static bool isIntOne(Attribute attr) {
  if (auto i = dyn_cast_if_present<IntegerAttr>(attr))
    return i.getValue().isOne();
  if (auto d = dyn_cast_if_present<DenseIntElementsAttr>(attr))
    return d.isSplat() && d.getSplatValue<APInt>().isOne();
  return false;
}


static Attribute makeIntConstant(Type type, int64_t v) {
  Attribute scalar = IntegerAttr::get(getElementTypeOrSelf(type), v);
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return {};
    return SplatElementsAttr::get(shaped, scalar);
  }
  return scalar;
}



OpFoldResult addOp::fold(FoldAdaptor adaptor) {
  // add(x, 0) -> x. Integer only: for floats, -0.0 + 0.0 is +0.0.
  if (isIntZero(adaptor.getRhs())) return getLhs();
  if (isIntZero(adaptor.getLhs())) return getRhs();

  Type type = getResult().getType();
  Type elt = getElementTypeOrSelf(type);
  if (isa<IntegerType>(elt))
    return constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), type,
        [](const APInt &a, const APInt &b) { return a + b; });
  if (isa<FloatType>(elt))
    return constFoldBinaryOp<FloatAttr>(
        adaptor.getOperands(), type,
        [](const APFloat &a, const APFloat &b) { return a + b; });
  return {};
}

OpFoldResult subOp::fold(FoldAdaptor adaptor) {
  Type type = getResult().getType();
  Type elt = getElementTypeOrSelf(type);

  
  if (isIntZero(adaptor.getRhs())) return getLhs();
  if (isa<IntegerType>(elt) && getLhs() == getRhs())
    return makeIntConstant(type, 0);

  if (isa<IntegerType>(elt))
    return constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), type,
        [](const APInt &a, const APInt &b) { return a - b; });
  if (isa<FloatType>(elt))
    return constFoldBinaryOp<FloatAttr>(
        adaptor.getOperands(), type,
        [](const APFloat &a, const APFloat &b) { return a - b; });
  return {};
}

OpFoldResult mulOp::fold(FoldAdaptor adaptor) {
  Type type = getResult().getType();
  Type elt = getElementTypeOrSelf(type);

  // Integer only: mul(x, 1) -> x, mul(x, 0) -> 0. For floats, 0.0 * x is
  // not 0.0 when x is NaN, inf or negative.
  if (isIntOne(adaptor.getRhs())) return getLhs();
  if (isIntOne(adaptor.getLhs())) return getRhs();
  if (isIntZero(adaptor.getRhs()) || isIntZero(adaptor.getLhs()))
    return makeIntConstant(type, 0);

  if (isa<IntegerType>(elt))
    return constFoldBinaryOp<IntegerAttr>(
        adaptor.getOperands(), type,
        [](const APInt &a, const APInt &b) { return a * b; });
  if (isa<FloatType>(elt))
    return constFoldBinaryOp<FloatAttr>(
        adaptor.getOperands(), type,
        [](const APFloat &a, const APFloat &b) { return a * b; });
  return {};
}

OpFoldResult divideOp::fold(FoldAdaptor adaptor) {
  // divide(x, 1) -> x for integers. Constant-constant folding is left out
  // on purpose: integer division by a constant zero must not fold.
  if (isIntOne(adaptor.getDenominator())) return getNumerator();
  return {};
}
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult constantOp::verify() {
  auto valueType = getValue().getType();
  auto resultType = getResult().getType();
  if (valueType != resultType)
    return emitOpError("result type ") << resultType
        << " does not match value type " << valueType;
  return success();
}

OpFoldResult constantOp::fold(FoldAdaptor adaptor) {
   return getValue();
}

LogicalResult arangeOp::verify() {
  auto tensorType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!tensorType || tensorType.getRank() != 1)
    return emitOpError("result must be a 1-D (ranked) tensor");
  if (tensorType.getElementType() != getStart().getType())
    return emitOpError("result element type must match operand type");
  return success();
}


LogicalResult addcmulOp::verify() {
  Type inputType = getInput().getType();
  if (getTensor1().getType() != inputType || getTensor2().getType() != inputType ||
      getResult().getType() != inputType)
    return emitOpError("input, tensor1, tensor2, and result must all share one type");

  Type expectedValueType = getElementTypeOrSelf(inputType);
  if (getValue().getType() != expectedValueType)
    return emitOpError("value type (") << getValue().getType()
        << ") must match the element type of input/tensor1/tensor2 ("
        << expectedValueType << ")";
  return success();
}

#define GET_OP_CLASSES
#include "numeric/numericOps.cpp.inc"