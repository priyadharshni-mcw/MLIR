
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