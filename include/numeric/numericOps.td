
#ifndef NUMERIC_OPS
#define NUMERIC_OPS

include "numeric/numericDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Shared type constraint used by every arithmetic op below: a scalar
// integer or float, or a tensor of integers or floats.
def Numeric_ScalarOrTensor : AnyTypeOf<[AnyInteger, AnyFloat,
                                         TensorOf<[AnyInteger, AnyFloat]>]>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def numeric_constantOp : numeric_Op<"constant", [ConstantLike, Pure]> {
  let summary = "A constant scalar or tensor value";
  let description = [{
    Produces a constant value with no operands. The result type is
    inferred from the attribute itself, and the verifier checks that
    they actually match.

  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Numeric_ScalarOrTensor:$result);
  
  let hasVerifier = 1;
  let hasFolder = 1;

}

//===----------------------------------------------------------------------===//
// AddOp  
//===----------------------------------------------------------------------===//

def numeric_addOp : numeric_Op<"add", [Pure, Commutative,
                                        SameOperandsAndResultRank,
                                        SameOperandsAndResultType]> {
  let summary = "Elementwise addition of scalars or tensors";
  let arguments = (ins Numeric_ScalarOrTensor:$lhs,
                        Numeric_ScalarOrTensor:$rhs);
  let results   = (outs Numeric_ScalarOrTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def numeric_mulOp : numeric_Op<"mul", [Pure, Commutative,
                                        SameOperandsAndResultRank,
                                        SameOperandsAndResultType]> {
  let summary = "Elementwise multiplication of scalars or tensors";
  let arguments = (ins Numeric_ScalarOrTensor:$lhs,
                        Numeric_ScalarOrTensor:$rhs);
  let results   = (outs Numeric_ScalarOrTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// SubOp  (NOT commutative — order matters: lhs - rhs)
//===----------------------------------------------------------------------===//

def numeric_subOp : numeric_Op<"sub", [Pure,
                                        SameOperandsAndResultRank,
                                        SameOperandsAndResultType]> {
  let summary = "Elementwise subtraction of scalars or tensors (lhs - rhs)";
  let arguments = (ins Numeric_ScalarOrTensor:$lhs,
                        Numeric_ScalarOrTensor:$rhs);
  let results   = (outs Numeric_ScalarOrTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // NUMERIC_OPS

def numeric_arangeOp : numeric_Op<"arange", [Pure, SameTypeOperands]> {
  let summary = "Generates a 1-D tensor of evenly spaced values (torch.arange semantics)";
  let description = [{
    Produces a 1-D tensor containing values `[start, start+step, start+2*step, ...]`
    up to but not including `end`, mirroring PyTorch's `torch.arange`. The
    element count is `ceil((end - start) / step)`, so the result generally
    has a dynamic shape.

    Example:
```mlir
    %0 = numeric.arange %start, %end, %step : (i64, i64, i64) -> tensor<?xi64>
```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, AnyFloat]>:$start,
                        AnyTypeOf<[AnyInteger, AnyFloat]>:$end,
                        AnyTypeOf<[AnyInteger, AnyFloat]>:$step);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$start `,` $end `,` $step attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def numeric_addcmulOp : numeric_Op<"addcmul", [Pure]> {
  let summary = "Elementwise multiply-add: out = input + value * tensor1 * tensor2";
  let description = [{
    Mirrors torch.addcmul. Performs the element-wise multiplication of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input
    result = input + value * tensor1 * tensor2

    Example:
    %r = numeric.addcmul %input, %t1, %t2, %value
        : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
  }];

  let arguments = (ins  TensorOf<[AnyInteger, AnyFloat]>:$input,
                        TensorOf<[AnyInteger, AnyFloat]>:$tensor1,
                        TensorOf<[AnyInteger, AnyFloat]>:$tensor2,
                        AnyTypeOf<[AnyInteger, AnyFloat]>:$value);
  let results = (outs TensorOf<[AnyInteger, AnyFloat]>:$result);

  let assemblyFormat = "$input `,` $tensor1 `,` $tensor2 `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}