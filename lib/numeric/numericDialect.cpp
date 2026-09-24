#include "numeric/numericDialect.h"
#include "numeric/numericOps.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace numeric;

#include "numeric/numericOpsDialect.cpp.inc"

void numericDialect::initialize() {
  addOperations <
#define GET_OP_LIST
#include "numeric/numericOps.cpp.inc"
      >();
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