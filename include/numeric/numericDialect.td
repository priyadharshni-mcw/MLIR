#ifndef NUMERIC_DIALECT
#define NUMERIC_DIALECT

include "mlir/IR/OpBase.td"

def numeric_Dialect : Dialect {
    let name = "numeric";
    let summary = "A simple numeric dialect for arithmetic operations";
    let cppNamespace = "::numeric";
    let description = [{
        This dialect defines basic arithmetic operations, starting with
        addition.
    }];
}

class numeric_Op<string mnemonic, list<Trait> traits = []> :
    Op<numeric_Dialect, mnemonic, traits>;

#endif // NUMERIC_DIALECT