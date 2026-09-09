module {
  func.func @const_scalar_int() -> i64 {
    %a = "numeric.constant"() <{value = 5 : i64}> : () -> i64
    return %a : i64
  }

  func.func @const_scalar_float() -> f32 {
    %a = "numeric.constant"() <{value = 3.14 : f32}> : () -> f32
    return %a : f32
  }

  func.func @const_tensor_int() -> tensor<4xi32> {
    %a = "numeric.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
    return %a : tensor<4xi32>
  }

  func.func @const_tensor_float() -> tensor<2xf32> {
    %a = "numeric.constant"() <{value = dense<[1.5, 2.5]> : tensor<2xf32>}> : () -> tensor<2xf32>
    return %a : tensor<2xf32>
  }
}