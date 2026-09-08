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

#define GET_OP_CLASSES
#include "numeric/numericOps.cpp.inc"