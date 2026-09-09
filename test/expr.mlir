// Output = a + (b * c) - d
module {
  func.func @expr_scalar_int() -> i64 {
    %a = "numeric.constant"() <{value = 4 : i64}> : () -> i64
    %b = "numeric.constant"() <{value = 3 : i64}> : () -> i64
    %c = "numeric.constant"() <{value = 5 : i64}> : () -> i64
    %d = "numeric.constant"() <{value = 2 : i64}> : () -> i64

    %bc        = numeric.mul %b, %c : i64          
    %a_plus_bc = numeric.add %a, %bc : i64          
    %result    = numeric.sub %a_plus_bc, %d : i64   

    return %result : i64
  }

  func.func @expr_scalar_float() -> f32 {
    %a = "numeric.constant"() <{value = 1.5 : f32}> : () -> f32
    %b = "numeric.constant"() <{value = 2.0 : f32}> : () -> f32
    %c = "numeric.constant"() <{value = 3.0 : f32}> : () -> f32
    %d = "numeric.constant"() <{value = 0.5 : f32}> : () -> f32

    %bc        = numeric.mul %b, %c : f32
    %a_plus_bc = numeric.add %a, %bc : f32
    %result    = numeric.sub %a_plus_bc, %d : f32

    return %result : f32
  }

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
}