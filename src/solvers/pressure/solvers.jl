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
        (; grid, boundary_conditions, ArrayType) = setup
        (; x, Np) = grid
        # T = eltype(x[1])
        T = Float64
        L = laplacian_mat(setup)
        if any(bc -> bc[1] isa PressureBC || bc[2] isa PressureBC, boundary_conditions)
            f = zeros(T, prod(Np))
            p = zeros(T, prod(Np))
        else
            f = zeros(T, prod(Np) + 1)
            p = zeros(T, prod(Np) + 1)
            e = ones(T, size(L, 2))
            L = [L e; e' 0]
        end
        fact = factorize(L)
        new{T,typeof(setup),typeof(fact),typeof(f)}(setup, fact, f, p)
    end
end

# """
#     CGPressureSolver(setup; [abstol], [reltol], [maxiter])
#
# Conjugate gradients iterative pressure solver.
# """
# struct CGPressureSolver{T,M<:AbstractMatrix{T}} <: AbstractPressureSolver{T}
#     A::M
#     abstol::T
#     reltol::T
#     maxiter::Int
# end

# function CGPressureSolver(
#     setup;
#     abstol = 0,
#     reltol = √eps(eltype(setup.operators.A)),
#     maxiter = size(setup.operators.A, 2),
# )
#     (; A) = setup.operators
#     CGPressureSolver{eltype(A),typeof(A)}(A, abstol, reltol, maxiter)
# end

# # This moves all the inner arrays to the GPU when calling
# # `cu(::SpectralPressureSolver)` from CUDA.jl
# Adapt.adapt_structure(to, s::CGPressureSolver) = CGPressureSolver(
#     adapt(to, s.A),
#     adapt(to, s.abstol),
#     adapt(to, s.reltol),
#     adapt(to, s.maxiter),
# )

"""
    CGMatrixPressureSolver(setup; [abstol], [reltol], [maxiter])

Conjugate gradients iterative pressure solver.
"""
struct CGMatrixPressureSolver{T,M} <: AbstractPressureSolver{T}
    L::M
    abstol::T
    reltol::T
    maxiter::Int
end

function CGMatrixPressureSolver(
    setup;
    abstol = 0,
    reltol = sqrt(eps(eltype(setup.grid.x[1]))),
    maxiter = prod(setup.grid.Np),
)
    L = laplacian_mat(setup) |> setup.device
    # L = L |> CuSparseMatrixCSR
    CGMatrixPressureSolver{eltype(L),typeof(L)}(L, abstol, reltol, maxiter)
end

"""
    CGPressureSolver(setup; [abstol], [reltol], [maxiter])

Conjugate gradients iterative pressure solver.
"""
struct CGPressureSolver{T,S,A,F} <: AbstractPressureSolver{T}
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
    (; grid, workgroupsize) = setup
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
        _laplace_diag!(get_backend(z), workgroupsize)(z, p, I0; ndrange)
        # synchronize(get_backend(z))
    end
end

CGPressureSolver(
    setup;
    abstol = zero(eltype(setup.grid.x[1])),
    reltol = sqrt(eps(eltype(setup.grid.x[1]))),
    maxiter = prod(setup.grid.Np),
    # preconditioner = copy!,
    preconditioner = create_laplace_diag(setup),
) = CGPressureSolver(
    setup,
    abstol,
    reltol,
    maxiter,
    similar(setup.grid.x[1], setup.grid.N),
    similar(setup.grid.x[1], setup.grid.N),
    similar(setup.grid.x[1], setup.grid.N),
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

    Ahat = fill!(similar(x[1], Complex{T}, Np), 0)
    Tπ = T(π) # CUDA doesn't like pi
    for d = 1:D
        @. Ahat += sin(k[d] * Tπ / Np[d])^2 / Δx[d]^2
    end

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Ahat .*= 4 * prod(Δx)

    # Pressure is determined up to constant. By setting the constant
    # scaling factor to 1, we preserve the average.
    # Note use of singleton range 1:1 instead of scalar index 1
    # (otherwise CUDA gets annoyed)
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

struct LowMemorySpectralPressureSolver{
    T,
    A<:AbstractArray{T},
    P<:AbstractArray{Complex{T}},
    S,
} <: AbstractPressureSolver{T}
    setup::S
    ahat::A
    phat::P
end

"""
    LowMemorySpectralPressureSolver(setup)

Build spectral pressure solver from setup.
This one is slower than `SpectralPressureSolver` but occupies less memory.
"""
function LowMemorySpectralPressureSolver(setup)
    (; grid, boundary_conditions) = setup
    (; dimension, Δ, Np, x) = grid

    D = dimension()
    T = eltype(Δ[1])

    Δx = Array(Δ[1])[1]

    @assert(
        all(bc -> bc[1] isa PeriodicBC && bc[2] isa PeriodicBC, boundary_conditions),
        "SpectralPressureSolver only implemented for periodic boundary conditions",
    )

    @assert(
        all(α -> all(≈(Δx), Δ[α]), 1:D),
        "SpectralPressureSolver requires uniform grid along each dimension",
    )

    @assert all(n -> n == Np[1], Np)

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx[1] and Δx[2] do not need to be the same

    k = 0:Np[1]-1

    Tπ = T(π) # CUDA doesn't like pi
    ahat = fill!(similar(x[1], Np[1]), 0)
    @. ahat = 4 * Δx^D * sin(k * Tπ / Np[1])^2 / Δx^2

    # Placeholders for intermediate results
    phat = fill!(similar(x[1], Complex{T}, Np), 0)

    LowMemorySpectralPressureSolver{T,typeof(ahat),typeof(phat),typeof(setup)}(
        setup,
        ahat,
        phat,
    )
end
