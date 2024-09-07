"""
    IncompressibleNavierStokesCUDSSExt

CUDSS extension for IncompressibleNavierStokes.
This makes `psolver_direct` use a CUDSS decomposition for `CuArray`s.
"""
module IncompressibleNavierStokesCUDSSExt

using CUDSS
using CUDSS.CUDA
using CUDSS.CUDA.CUSPARSE
using IncompressibleNavierStokes
using IncompressibleNavierStokes: PressureBC, laplacian_mat
using PrecompileTools
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

# Same as src/precompile.jl, but for `CuArray`s
PrecompileTools.@compile_workload begin
    for D in (2, 3), T in (Float32, Float64)
        # Periodic
        x = ntuple(d -> range(T(0), T(1), 5), D)
        setup = Setup(x...; Re = T(1000), ArrayType = CuArray)
        ustart = velocityfield(setup, (dim, x...) -> zero(x[1]))
        solve_unsteady(; ustart, setup, Δt = T(1e-3), tlims = (T(0), T(1e-2)))

        # Boundaries, temperature
        x = ntuple(d -> tanh_grid(T(0), T(1), 6), D)
        boundary_conditions = ntuple(d -> (DirichletBC(), PressureBC()), D)
        temperature = temperature_equation(;
            Pr = T(0.71),
            Ra = T(1e6),
            Ge = T(1.0),
            boundary_conditions,
        )
        setup =
            Setup(x...; Re = T(1000), temperature, boundary_conditions, ArrayType = CuArray)
        ustart = velocityfield(setup, (dim, x...) -> zero(x[1]))
        tempstart = temperaturefield(setup, (x...) -> zero(x[1]))
        solve_unsteady(; ustart, tempstart, setup, Δt = T(1e-3), tlims = (T(0), T(1e-2)))
    end
end

end
