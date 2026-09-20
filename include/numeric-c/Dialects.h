#ifndef NUMERIC_C_DIALECTS_H
#define NUMERIC_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Numeric, numeric);

MLIR_CAPI_EXPORTED void numericRegisterAllExtensions(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // NUMERIC_C_DIALECTS_H