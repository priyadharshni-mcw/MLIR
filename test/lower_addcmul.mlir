// RUN: numeric-opt %s -convert-numeric-to-arith | FileCheck %s

// CHECK-LABEL: func.func @addcmul_tensor
func.func @addcmul_tensor(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                           %t2: tensor<4xi64>, %value: i64) -> tensor<4xi64> {
  // CHECK: %[[SHIFT:.*]] = "tosa.const"()
  // CHECK: %[[VAL1D:.*]] = tensor.from_elements %arg3 : tensor<1xi64>
  // CHECK: %[[TEMP:.*]] = tosa.mul %arg1, %arg2, %[[SHIFT]]
  // CHECK: %[[SCALED:.*]] = tosa.mul %[[TEMP]], %[[VAL1D]], %[[SHIFT]]
  // CHECK: %[[RESULT:.*]] = tosa.add %arg0, %[[SCALED]]
  // CHECK-NOT: numeric.
  %0 = numeric.addcmul %input, %t1, %t2, %value
      : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}