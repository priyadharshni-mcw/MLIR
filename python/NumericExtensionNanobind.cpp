#include "numeric-c/Dialects.h"
#include "numeric-c/Passes.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Bufferization.h"
#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
namespace nb = nanobind;

NB_MODULE(_numericDialectsNanobind, m) {
  auto numericM = m.def_submodule("numeric");

  numericRegisterAllPasses();

  numericM.def(
      "register_dialects",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirContext ctx = context.get()->get();

        MlirDialectRegistry registry = mlirDialectRegistryCreate();
        MlirDialectHandle handles[] = {
            mlirGetDialectHandle__numeric__(),
            mlirGetDialectHandle__arith__(),
            mlirGetDialectHandle__func__(),
            mlirGetDialectHandle__memref__(),
            mlirGetDialectHandle__scf__(),
            mlirGetDialectHandle__cf__(),
            mlirGetDialectHandle__linalg__(),
            mlirGetDialectHandle__bufferization__(),
            mlirGetDialectHandle__tensor__(),
            mlirGetDialectHandle__llvm__(),
        };
        for (MlirDialectHandle h : handles)
          mlirDialectHandleInsertDialect(h, registry);

        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);

        if (load)
          for (MlirDialectHandle h : handles)
            mlirDialectHandleLoadDialect(h, ctx);

        MlirDialectRegistry extRegistry = mlirDialectRegistryCreate();
        numericRegisterAllExtensions(extRegistry);
        mlirContextAppendDialectRegistry(ctx, extRegistry);
        mlirDialectRegistryDestroy(extRegistry);
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}