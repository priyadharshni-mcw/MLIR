add_mlir_dialect_library(MLIRNumeric
  numericDialect.cpp
  LowerNumericToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/numeric

  DEPENDS
  MLIRnumericOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRTosaDialect
)