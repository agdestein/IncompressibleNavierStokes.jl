"""
    PressureSolver

Pressure solver for the Poisson equation.
"""
abstract type AbstractPressureSolver{T} end

"""
    DirectPressureSolver()

Direct pressure solver using a LU decomposition.
"""
struct DirectPressureSolver{T,F<:Factorization{T}} <: AbstractPressureSolver{T}
    A_fact::F
    function DirectPressureSolver(setup::Setup{T}) where {T}
        fact = factorize(setup.operators.A)
        new{T,typeof(fact)}(fact)
    end
end

"""
    CGPressureSolver(abstol, reltol, maxiter)

Conjugate gradients iterative pressure solver.
"""
struct CGPressureSolver{T} <: AbstractPressureSolver{T}
    A::SparseMatrixCSC{T,Int}
    abstol::T
    reltol::T
    maxiter::Int
    function CGPressureSolver(
        setup::Setup{T};
        abstol = 0,
        reltol = √eps(T),
        maxiter = nothing,
    ) where {T}
        (; A) = setup.operators
        isnothing(maxiter) && (maxiter = size(A, 2))
        new{T}(A, abstol, reltol, maxiter)
    end
end

"""
    FourierPressureSolver()

Fourier transform pressure solver for periodic domains.
"""
struct FourierPressureSolver{T,N} <: AbstractPressureSolver{T}
    Ahat::Array{Complex{T},N}
    phat::Array{Complex{T},N}
    fhat::Array{Complex{T},N}
end

"""
    FourierPressureSolver(setup)

Build Fourier pressure solver from setup.
"""
function FourierPressureSolver(setup::Setup{T,2}) where {T}
    (; grid, boundary_conditions, operators) = setup
    (; hx, hy, Npx, Npy) = grid
    (; A) = operators

    if any(
        !isequal((:periodic, :periodic)),
        (boundary_conditions.u.x, boundary_conditions.v.y),
    )
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = hx[1]
    Δy = hy[1]
    if any(≉(Δx), hx) || any(≉(Δy), hy)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx, Δy and Δz do not need to be the same
    i = 0:(Npx-1)
    j = reshape(0:(Npy-1), 1, :)

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat = @. 4 * Δx * Δy * (sin(i * π / Npx)^2 / Δx^2 + sin(j * π / Npy)^2 / Δy^2)

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    FourierPressureSolver{T,2}(Ahat, phat, fhat)
end

"""
    FourierPressureSolver(setup)

Build Fourier pressure solver from setup.
"""
function FourierPressureSolver(setup::Setup{T,3}) where {T}
    (; grid, boundary_conditions, operators) = setup
    (; hx, hy, hz, Npx, Npy, Npz) = grid
    (; A) = operators

    if any(
        !isequal((:periodic, :periodic)),
        [boundary_conditions.u.x, boundary_conditions.v.y, boundary_conditions.w.z],
    )
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = hx[1]
    Δy = hy[1]
    Δz = hz[1]
    if any(≉(Δx), hx) || any(≉(Δy), hy) || any(≉(Δz), hz)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx, Δy and Δz do not need to be the same
    i = 0:(Npx-1)
    j = reshape(0:(Npy-1), 1, :)
    k = reshape(0:(Npz-1), 1, 1, :)

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat = @. 4 *
       Δx *
       Δy *
       Δz *
       (sin(i * π / Npx)^2 / Δx^2 + sin(j * π / Npy)^2 / Δy^2 + sin(k * π / Npz)^2 / Δz^2)

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    FourierPressureSolver{T,3}(Ahat, phat, fhat)
end
