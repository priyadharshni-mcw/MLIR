#include "numeric-c/Dialects.h"
#include "numeric/numericDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

void numericRegisterAllExtensions(MlirDialectRegistry registry) {
  mlir::DialectRegistry *reg = unwrap(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(*reg);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(*reg);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(*reg);
  mlir::bufferization::registerBufferizableOpInterfaceExternalModels(*reg);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(*reg);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Numeric, numeric, numeric::numericDialect)