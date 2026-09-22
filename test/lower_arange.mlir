// RUN: numeric-opt %s -convert-numeric-to-arith | FileCheck %s

// CHECK-LABEL: func.func @arange_int
func.func @arange_int(%start: i64, %end: i64, %step: i64) -> tensor<?xi64> {
  // CHECK: %[[DIFF:.*]] = arith.subi %arg1, %arg0 : i64
  // CHECK: %[[N:.*]] = arith.ceildivsi %[[DIFF]], %arg2 : i64
  // CHECK: %[[NIDX:.*]] = arith.index_cast %[[N]] : i64 to index
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[NIDX]]) : tensor<?xi64>
  // CHECK: linalg.generic
  // CHECK: linalg.index 0
  // CHECK: arith.index_cast
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: linalg.yield
  // CHECK-NOT: numeric.arange
  %0 = numeric.arange %start, %end, %step : (i64, i64, i64) -> tensor<?xi64>
  return %0 : tensor<?xi64>
}