func.func @divide_int(%n: tensor<4xi64>, %d: tensor<4xi64>) -> tensor<4xi64> {
  %0 = numeric.divide %n, %d : tensor<4xi64>
  return %0 : tensor<4xi64>
}

func.func @divide_float(%n: tensor<4xf32>, %d: tensor<4xf32>) -> tensor<4xf32> {
  %0 = numeric.divide %n, %d : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @add_scalar(%a: i64, %b: i64) -> i64 {
  %0 = numeric.add %a, %b : i64
  return %0 : i64
}

func.func @add_tensor(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi64> {
  %0 = numeric.add %a, %b : tensor<4xi64>
  return %0 : tensor<4xi64>
}