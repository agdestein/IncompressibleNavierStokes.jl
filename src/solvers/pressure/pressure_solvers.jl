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
function SpectralPressureSolver(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, Δ, Np, x) = grid

    D = dimension()
    T = eltype(Δ[1])

    Δx = first.(Array.(Δ))

    @assert(
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions),
        "SpectralPressureSolver only implemented for periodic boundary conditions",
    )

    @assert(
        all(α -> all(≈(Δx[α]), Δ[α]), 1:D),
        "SpectralPressureSolver requires uniform grid along each dimension",
    )


    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same

    k = ntuple(d -> reshape(0:Np[d]-1, ntuple(Returns(1), d - 1)..., :, ntuple(Returns(1), D - d)...), D)

    Ahat = KernelAbstractions.zeros(get_backend(x[1]), Complex{T}, Np...)
    Tπ = T(π) # CUDA doesn't like pi
    for d = 1:D
        @. Ahat += sin(k[d] * Tπ / Np[d])^2 / Δx[d]^2
    end

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat .*= 4 * prod(Δx)

    # Pressure is determined up to constant. By setting the constant
    # scaling factor to 1, we preserve the average.
    Ahat[1:1] .= 1

    # Placeholders for intermediate results
    phat = zero(Ahat)
    fhat = zero(Ahat)

    SpectralPressureSolver{T,typeof(Ahat)}(Ahat, phat, fhat)
end
