#ifndef NUMERIC_OPS
#define NUMERIC_OPS

include "numeric/numericDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def numeric_addOp : numeric_Op<"add", [Pure, Commutative,
                                        SameOperandsAndResultRank,
                                        SameOperandsAndResultType]> {
    let summary = "Elementwise addition of scalars or tensors";
    let arguments = (ins AnyTypeOf<[AnyInteger, TensorOf<[AnyInteger]>]>:$lhs,
                          AnyTypeOf<[AnyInteger, TensorOf<[AnyInteger]>]>:$rhs);
    let results   = (outs AnyTypeOf<[AnyInteger, TensorOf<[AnyInteger]>]>:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // NUMERIC_OPS