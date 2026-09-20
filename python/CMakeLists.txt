include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(NumericPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NumericPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_numeric"
  TD_FILE dialects/numericOps.td
  SOURCES
    dialects/numeric_nanobind.py
  DIALECT_NAME numeric)

declare_mlir_python_extension(NumericPythonSources.NanobindExtension
  MODULE_NAME _numericDialectsNanobind
  ADD_TO_PARENT NumericPythonSources
  SOURCES
    NumericExtensionNanobind.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPIMemRef
    MLIRCAPISCF
    MLIRCAPIControlFlow
    MLIRCAPILinalg
    MLIRCAPIBufferization
    MLIRCAPITensor
    MLIRCAPILLVM
    MLIRCAPITransforms
    NumericCAPI
)

add_mlir_python_common_capi_library(NumericPythonCAPI
  INSTALL_COMPONENT NumericPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    NumericPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.ExecutionEngine
)

add_mlir_python_modules(NumericPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    NumericPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    NumericPythonCAPI
)