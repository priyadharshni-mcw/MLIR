#include "numeric/numericDialect.h"
#include "numeric/numericOps.h"

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

#define GET_OP_CLASSES
#include "numeric/numericOps.cpp.inc"