// RUN: numeric-opt %s \
// RUN:   -convert-numeric-to-arith \
// RUN:   -convert-elementwise-to-linalg \
// RUN:   -one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   -convert-linalg-to-loops \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-cf-to-llvm \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @expr_tensor
// CHECK-NOT: numeric.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: memref.
// CHECK-NOT: arith.
// CHECK-NOT: func.func
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.call @malloc
// CHECK-NOT: numeric.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: scf.
// CHECK-NOT: memref.
// CHECK-NOT: arith.
// CHECK-NOT: func.func
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.return
func.func @expr_tensor() -> tensor<4xi32> {
  %a = "numeric.constant"() <{value = dense<[4, 4, 4, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %b = "numeric.constant"() <{value = dense<[3, 3, 3, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %c = "numeric.constant"() <{value = dense<[5, 5, 5, 5]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %d = "numeric.constant"() <{value = dense<[2, 2, 2, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>

  %bc        = numeric.mul %b, %c : tensor<4xi32>
  %a_plus_bc = numeric.add %a, %bc : tensor<4xi32>
  %result    = numeric.sub %a_plus_bc, %d : tensor<4xi32>

  return %result : tensor<4xi32>
}