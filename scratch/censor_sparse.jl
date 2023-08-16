using SparseArrays
using LinearAlgebra

# Hide long sparse arrays
function Base.show(io::IO, _S::SparseArrays.AbstractSparseMatrixCSCInclAdjointAndTranspose)
    SparseArrays._checkbuffers(_S)
    # can't use `findnz`, because that expects all values not to be #undef
    S = _S isa Adjoint || _S isa Transpose ? _S.parent : _S
    I = rowvals(S)
    K = nonzeros(S)
    m, n = size(S)
    if _S isa Adjoint
        print(io, "adjoint(")
    elseif _S isa Transpose
        print(io, "transpose(")
    end
    nn = nnz(S)
    if nn > 20
        print(io, "sparse(<CENSORED>)")
        if _S isa Adjoint || _S isa Transpose
            print(io, ")")
        end
        return
    end
    print(io, "sparse(", I, ", ")
    if length(I) == 0
        print(io, eltype(SparseArrays.getcolptr(S)), "[]")
    else
        print(io, "[")
        il = nnz(S) - 1
        for col = 1:size(S, 2),
            k = SparseArrays.getcolptr(S)[col]:(SparseArrays.getcolptr(S)[col+1]-1)

            print(io, col)
            if il > 0
                print(io, ", ")
                il -= 1
            end
        end
        print(io, "]")
    end
    print(io, ", ", K, ", ", m, ", ", n, ")")
    if _S isa Adjoint || _S isa Transpose
        print(io, ")")
    end
end
