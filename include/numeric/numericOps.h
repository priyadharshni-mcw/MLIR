#ifndef NUMERIC_NUMERICOPS_H
#define NUMERIC_NUMERICOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "numeric/numericDialect.h"

#define GET_OP_CLASSES
#include "numeric/numericOps.h.inc"

#endif // NUMERIC_NUMERICOPS_H