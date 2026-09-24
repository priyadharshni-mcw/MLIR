// RUN: numeric-opt %s -convert-numeric-to-arith | FileCheck %s

// CHECK-LABEL: func.func @addcmul_tensor
func.func @addcmul_tensor(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                           %t2: tensor<4xi64>, %value: i64) -> tensor<4xi64> {
  // CHECK: %[[TEMP:.*]] = arith.muli %arg1, %arg2 : tensor<4xi64>
  // CHECK: linalg.generic
  // CHECK: arith.muli %{{.*}}, %arg3 : i64
  // CHECK: linalg.yield
  // CHECK: %[[RESULT:.*]] = arith.addi %arg0, %{{.*}} : tensor<4xi64>
  // CHECK-NOT: numeric.
  %0 = numeric.addcmul %input, %t1, %t2, %value
      : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}