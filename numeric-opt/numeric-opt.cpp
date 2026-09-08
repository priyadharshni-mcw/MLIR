#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "numeric/numericDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);          
  registry.insert<numeric::numericDialect>();     
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "numeric-opt: numeric dialect standalone IR tool\n", registry));
}