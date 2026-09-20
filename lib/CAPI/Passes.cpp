#include "numeric-c/Passes.h"
#include "numeric/Passes.h"

#include "mlir/InitAllPasses.h"

void numericRegisterAllPasses(void) {
  mlir::registerAllPasses();
  numeric::registerConvertNumericToArithPass();
}