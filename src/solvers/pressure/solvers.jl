"""
    AbstractPressureSolver

Pressure solver for the Poisson equation.
"""
abstract type AbstractPressureSolver{T} end

"""
    DirectPressureSolver()

Direct pressure solver using a LU decomposition.
"""
struct DirectPressureSolver{T,S,F,A} <: AbstractPressureSolver{T}
    setup::S
    fact::F
    f::A
    p::A
    function DirectPressureSolver(setup)
        (; x, Np) = setup.grid
        # T = eltype(x[1])
        T = Float64
        backend = get_backend(x[1])
        # f = KernelAbstractions.zeros(backend, T, prod(Np))
        # p = KernelAbstractions.zeros(backend, T, prod(Np))
        f = zeros(T, prod(Np) + 1)
        p = zeros(T, prod(Np) + 1)
        L = laplacian_mat(setup)
        e = ones(T, size(L, 2))
        L = [L e; e' 0]
        # fact = lu(L)
        fact = factorize(L)
        new{T,typeof(setup),typeof(fact),typeof(f)}(setup, fact, f, p)
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

"""
    CGPressureSolverManual(setup; [abstol], [reltol], [maxiter])

Conjugate gradients iterative pressure solver.
"""
struct CGPressureSolverManual{T,S,A,F} <: AbstractPressureSolver{T}
    setup::S
    abstol::T
    reltol::T
    maxiter::Int
    r::A
    L::A
    q::A
    preconditioner::F
end

function create_laplace_diag(setup)
    (; grid) = setup
    (; dimension, Δ, Δu, N, Np, Ip, Ω) = grid
    D = dimension()
    δ = Offset{D}()
    @kernel function _laplace_diag!(z, p, I0)
        I = @index(Global, Cartesian)
        I = I + I0
        d = zero(eltype(z))
        for α = 1:length(I)
            d -= Ω[I] / Δ[α][I[α]] * (1 / Δu[α][I[α]] + 1 / Δu[α][I[α]-1])
        end
        z[I] = -p[I] / d
    end
    ndrange = Np
    I0 = first(Ip)
    I0 -= oneunit(I0)
    function laplace_diag(z, p)
        _laplace_diag!(get_backend(z), WORKGROUP)(z, p, I0; ndrange)
        # synchronize(get_backend(z))
    end
end

CGPressureSolverManual(
    setup;
    abstol = zero(eltype(setup.grid.x[1])),
    reltol = sqrt(eps(eltype(setup.grid.x[1]))),
    maxiter = prod(setup.grid.Np),
    # preconditioner = copy!,
    preconditioner = create_laplace_diag(setup),
) = CGPressureSolverManual(
    setup,
    abstol,
    reltol,
    maxiter,
    KernelAbstractions.zeros(
        get_backend(setup.grid.x[1]),
        eltype(setup.grid.x[1]),
        setup.grid.N,
    ),
    KernelAbstractions.zeros(
        get_backend(setup.grid.x[1]),
        eltype(setup.grid.x[1]),
        setup.grid.N,
    ),
    KernelAbstractions.zeros(
        get_backend(setup.grid.x[1]),
        eltype(setup.grid.x[1]),
        setup.grid.N,
    ),
    preconditioner,
)

struct SpectralPressureSolver{T,A<:AbstractArray{Complex{T}},S,P} <:
       AbstractPressureSolver{T}
    setup::S
    Ahat::A
    phat::A
    fhat::A
    plan::P
end

# This moves all the inner arrays to the GPU when calling
# `cu(::SpectralPressureSolver)` from CUDA.jl
Adapt.adapt_structure(to, s::SpectralPressureSolver) = SpectralPressureSolver(
    adapt(to, s.Ahat),
    adapt(to, s.phat),
    adapt(to, s.fhat),
    adapt(to, s.plan),
)

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

    k = ntuple(
        d -> reshape(
            0:Np[d]-1,
            ntuple(Returns(1), d - 1)...,
            :,
            ntuple(Returns(1), D - d)...,
        ),
        D,
    )

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
    plan = plan_fft(fhat)

    SpectralPressureSolver{T,typeof(Ahat),typeof(setup),typeof(plan)}(
        setup,
        Ahat,
        phat,
        fhat,
        plan,
    )
end
