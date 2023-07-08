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

# This moves all the inner arrays to the GPU when calling
# `cu(::SpectralPressureSolver)` from CUDA.jl
# TODO: CUDA does not support `factorize`, `lu`, etc, but maybe `L` and `U` can be
# converted to `CuArray` after factorization on the CPU
Adapt.adapt_structure(to, s::DirectPressureSolver) = error(
    "`DirectPressureSolver` is not yet implemented for CUDA. Consider using `CGPressureSolver`.",
)

"""
    CGPressureSolver(setup; [abstol], [reltol], [maxiter])

Conjugate gradients iterative pressure solver.
"""
struct CGPressureSolver{T,M<:AbstractMatrix{T}} <: AbstractPressureSolver{T}
    A::M
    abstol::T
    reltol::T
    maxiter::Int
end

function CGPressureSolver(
    setup;
    abstol = 0,
    reltol = √eps(eltype(setup.operators.A)),
    maxiter = size(setup.operators.A, 2),
)
    (; A) = setup.operators
    CGPressureSolver{eltype(A),typeof(A)}(A, abstol, reltol, maxiter)
end

# This moves all the inner arrays to the GPU when calling
# `cu(::SpectralPressureSolver)` from CUDA.jl
Adapt.adapt_structure(to, s::CGPressureSolver) = CGPressureSolver(
    adapt(to, s.A),
    adapt(to, s.abstol),
    adapt(to, s.reltol),
    adapt(to, s.maxiter),
)

struct SpectralPressureSolver{T,A<:AbstractArray{Complex{T}}} <: AbstractPressureSolver{T}
    Ahat::A
    phat::A
    fhat::A
end

# This moves all the inner arrays to the GPU when calling
# `cu(::SpectralPressureSolver)` from CUDA.jl
Adapt.adapt_structure(to, s::SpectralPressureSolver) =
    SpectralPressureSolver(adapt(to, s.Ahat), adapt(to, s.phat), adapt(to, s.fhat))

"""
    SpectralPressureSolver(setup)

Build spectral pressure solver from setup.
"""
SpectralPressureSolver(setup) = SpectralPressureSolver(setup.grid.dimension, setup)

function SpectralPressureSolver(::Dimension{2}, setup)
    (; grid, boundary_conditions) = setup
    (; hx, hy, Npx, Npy) = grid

    T = eltype(hx)
    AT = typeof(hx)

    if any(
        !isequal((:periodic, :periodic)),
        (boundary_conditions.u.x, boundary_conditions.v.y),
    )
        error("SpectralPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = first(hx)
    Δy = first(hy)
    if any(!≈(Δx), hx) || any(!≈(Δy), hy)
        error("SpectralPressureSolver requires uniform grid along each dimension")
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

    SpectralPressureSolver{T,typeof(Ahat)}(Ahat, phat, fhat)
end

function SpectralPressureSolver(::Dimension{3}, setup)
    (; grid, boundary_conditions) = setup
    (; hx, hy, hz, Npx, Npy, Npz) = grid

    T = eltype(hx)
    AT = typeof(hx)

    if any(
        !isequal((:periodic, :periodic)),
        [boundary_conditions.u.x, boundary_conditions.v.y, boundary_conditions.w.z],
    )
        error("SpectralPressureSolver only implemented for periodic boundary conditions")
    end

    Δx = first(hx)
    Δy = first(hy)
    Δz = first(hz)
    if any(!≈(Δx), hx) || any(!≈(Δy), hy) || any(!≈(Δz), hz)
        error("SpectralPressureSolver requires uniform grid along each dimension")
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
       (sin(i * π / Npx)^2 / Δx^2 + sin(j * π / Npy)^2 / Δy^2 + sin(k * π / Npz)^2 / Δz^2)

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    SpectralPressureSolver{T,typeof(Ahat)}(Ahat, phat, fhat)
end
