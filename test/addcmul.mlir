func.func @addcmul_tensor(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                           %t2: tensor<4xi64>, %value: i64) -> tensor<4xi64> {
  %0 = numeric.addcmul %input, %t1, %t2, %value
      : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}
