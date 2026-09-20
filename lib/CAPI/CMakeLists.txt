add_mlir_public_c_api_library(NumericCAPI
  Dialects.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRNumeric
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRBufferizationTransforms
  MLIRTransforms
  MLIRRegisterAllPasses
)