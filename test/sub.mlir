module {
  func.func @sub_scalar_int(%lhs: i64, %rhs: i64) -> i64 {
    %0 = numeric.sub %lhs, %rhs : i64
    return %0 : i64
  }

  func.func @sub_scalar_float(%lhs: f32, %rhs: f32) -> f32 {
    %0 = numeric.sub %lhs, %rhs : f32
    return %0 : f32
  }

  func.func @sub_tensor(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
    %0 = numeric.sub %lhs, %rhs : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}