"""
    IncompressibleNavierStokesCUDAExt

CUDA extension for IncompressibleNavierStokes.
"""
module IncompressibleNavierStokesCUDAExt

using CUDA
using CUDA.CUSPARSE
using CUDSS
using IncompressibleNavierStokes
using IncompressibleNavierStokes: PressureBC, laplacian_mat
using SparseArrays

# CUDA version, using CUDSS LDLt decomposition.
function IncompressibleNavierStokes.psolver_direct(::CuArray, setup)
    (; grid, boundary_conditions) = setup
    (; x, Np, Ip) = grid
    T = eltype(x[1])
    L = laplacian_mat(setup)
    isdefinite =
        any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
    if isdefinite
        # No extra DOF
        ftemp = fill!(similar(x[1], prod(Np)), 0)
        ptemp = fill!(similar(x[1], prod(Np)), 0)
        viewrange = (:)
        # structure = "SPD" # Symmetric positive definite
        structure = "G" # General matrix
        _view = 'F' # Full matrix representation
    else
        # With extra DOF
        ftemp = fill!(similar(x[1], prod(Np) + 1), 0)
        ptemp = fill!(similar(x[1], prod(Np) + 1), 0)
        # e = fill!(similar(x[1], prod(Np)), 1)
        e = ones(T, prod(Np))
        L = SparseMatrixCSC(L)
        L = [L e; e' 0]
        viewrange = 1:prod(Np)
        structure = "S" # Symmetric (not positive definite)
        _view = 'L' # Lower triangular representation
    end
    L = CuSparseMatrixCSR(L)
    solver = CudssSolver(L, structure, _view)
    cudss("analysis", solver, ptemp, ftemp)
    cudss("factorization", solver, ptemp, ftemp) # Compute factorization
    function psolve!(p, f)
        T = eltype(p)
        copyto!(view(ftemp, viewrange), view(view(f, Ip), :))
        cudss("solve", solver, ptemp, ftemp)
        copyto!(view(view(p, Ip), :), eltype(p).(view(ptemp, viewrange)))
        p
    end
end

end
