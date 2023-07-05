"""
    AbstractPressureSolver

Pressure solver for the Poisson equation.
"""
abstract type AbstractPressureSolver{T} end

"""
    DirectPressureSolver()

Direct pressure solver using a LU decomposition.
"""
struct DirectPressureSolver{T,F<:Factorization{T}} <: AbstractPressureSolver{T}
    A_fact::F
    function DirectPressureSolver(setup)
        (; A) = setup.operators
        T = eltype(A)
        fact = factorize(setup.operators.A)
        new{T,typeof(fact)}(fact)
    end
end

"""
    CGPressureSolver(setup; [abstol], [reltol], [maxiter])

Conjugate gradients iterative pressure solver.
"""
struct CGPressureSolver{T,M<:AbstractMatrix{T}} <: AbstractPressureSolver{T}
    A::M
    abstol::T
    reltol::T
    maxiter::Int
    function CGPressureSolver(
        setup;
        abstol = 0,
        reltol = √eps(eltype(setup.operators.A)),
        maxiter = size(setup.operators.A, 2),
    )
        (; A) = setup.operators
        new{eltype(A),typeof(A)}(A, abstol, reltol, maxiter)
    end
end

struct FourierPressureSolver{T,A<:AbstractArray{Complex{T}}} <: AbstractPressureSolver{T}
    Ahat::A
    phat::A
    fhat::A
end

"""
    FourierPressureSolver(setup)

Build Fourier pressure solver from setup.
"""
FourierPressureSolver(setup) = FourierPressureSolver(setup.grid.dimension, setup)

function FourierPressureSolver(::Dimension{2}, setup)
    (; grid, boundary_conditions) = setup
    (; hx, hy, Npx, Npy) = grid

    T = eltype(hx)
    AT = typeof(hx)

    if any(
        !isequal((:periodic, :periodic)),
        (boundary_conditions.u.x, boundary_conditions.v.y),
    )
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = first(hx)
    Δy = first(hy)
    if any(!≈(Δx), hx) || any(!≈(Δy), hy)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx and Δy do not need to be the same
    i = AT(0:(Npx-1))
    j = reshape(AT(0:(Npy-1)), 1, :)

    # Scale with Δx*Δy, since we solve the PDE in integrated form
    Ahat = @. 4 * Δx * Δy * (sin(i * π / Npx)^2 / Δx^2 + sin(j * π / Npy)^2 / Δy^2)

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    FourierPressureSolver{T,typeof(Ahat)}(Ahat, phat, fhat)
end

function FourierPressureSolver(::Dimension{3}, setup)
    (; grid, boundary_conditions) = setup
    (; hx, hy, hz, Npx, Npy, Npz) = grid

    T = eltype(hx)
    AT = typeof(hx)

    if any(
        !isequal((:periodic, :periodic)),
        [boundary_conditions.u.x, boundary_conditions.v.y, boundary_conditions.w.z],
    )
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = first(hx)
    Δy = first(hy)
    Δz = first(hz)
    if any(!≈(Δx), hx) || any(!≈(Δy), hy) || any(!≈(Δz), hz)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx and Δy do not need to be the same
    i = AT(0:(Npx-1))
    j = reshape(AT(0:(Npy-1)), 1, :)
    k = reshape(AT(0:(Npz-1)), 1, 1, :)

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat = @. 4 *
        Δx *
        Δy *
        Δz *
       (
           sin(i * π / Npx)^2 / Δx^2 +
           sin(j * π / Npy)^2 / Δy^2 +
           sin(k * π / Npz)^2 / Δz^2
       )

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    FourierPressureSolver{T,typeof(Ahat)}(Ahat, phat, fhat)
end
