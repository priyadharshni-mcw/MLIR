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

#define GET_OP_CLASSES
#include "numeric/numericOps.cpp.inc"