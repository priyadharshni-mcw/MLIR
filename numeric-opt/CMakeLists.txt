
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRNumeric
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
)

add_llvm_executable(numeric-opt numeric-opt.cpp)
llvm_update_compile_flags(numeric-opt)
target_link_libraries(numeric-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(numeric-opt)