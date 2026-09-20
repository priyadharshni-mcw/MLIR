#ifndef NUMERIC_C_PASSES_H
#define NUMERIC_C_PASSES_H

#include "mlir-c/Support.h"
#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void numericRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // NUMERIC_C_PASSES_H