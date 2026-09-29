// RUN: numeric-opt %s --pass-pipeline='builtin.module(convert-numeric-to-arith,func.func(tosa-to-linalg),convert-elementwise-to-linalg,one-shot-bufferize{bufferize-function-boundaries=true},convert-linalg-to-loops,convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | FileCheck %s

// CHECK-LABEL: llvm.func @expr_tensor
// CHECK-NOT: numeric.
// CHECK-NOT: tosa.
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: memref.
// CHECK-NOT: arith.
// CHECK-NOT: func.func
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.call @malloc
// CHECK-NOT: numeric.
// CHECK-NOT: tosa.
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: memref.
// CHECK-NOT: arith.
// CHECK-NOT: func.func
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.return
func.func @expr_tensor(%a: tensor<4xi32>, %b: tensor<4xi32>,
                        %c: tensor<4xi32>, %d: tensor<4xi32>) -> tensor<4xi32> {
  %bc        = numeric.mul %b, %c : tensor<4xi32>
  %a_plus_bc = numeric.add %a, %bc : tensor<4xi32>
  %result    = numeric.sub %a_plus_bc, %d : tensor<4xi32>
  return %result : tensor<4xi32>
}
