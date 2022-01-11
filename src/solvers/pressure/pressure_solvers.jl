"""
    PressureSolver

Pressure solver for the Poisson equation.
"""
abstract type PressureSolver{T} end

"""
    DirectPressureSolver()

Direct pressure solver using a LU decomposition.
"""
Base.@kwdef mutable struct DirectPressureSolver{T} <: PressureSolver{T}
    A_fact::Factorization{T} = cholesky(spzeros(T, 0, 0))
end

"""
    CGPressureSolver(abstol, reltol, maxiter)

Conjugate gradients iterative pressure solver.
"""
Base.@kwdef mutable struct CGPressureSolver{T} <: PressureSolver{T}
    A::SparseMatrixCSC{T,Int} = spzeros(T, 0, 0)
    abstol::T = 0
    reltol::T = √eps(T)
    maxiter::Int = 0
end

"""
    FourierPressureSolver()

Fourier transform pressure solver for periodic domains.
"""
Base.@kwdef mutable struct FourierPressureSolver{T} <: PressureSolver{T}
    # TODO: Pass dimensionality to create concrete types
    Â::Array{Complex{T}} = zeros(Complex{T}, 0, 0)
    p̂::Array{Complex{T}} = zeros(Complex{T}, 0, 0)
    f̂::Array{Complex{T}} = zeros(Complex{T}, 0, 0)
end

"""
    initialize!(pressure_solver)

Initialize pressure solver.
"""
function initialize! end

initialize!(solver::DirectPressureSolver, setup, A) = (solver.A_fact = factorize(A))

function initialize!(solver::CGPressureSolver, setup, A)
    @pack! solver = A
    solver.maxiter == 0 && (solver.maxiter = size(A, 2))
end

function initialize!(solver::FourierPressureSolver, setup, A)
    (; bc) = setup
    (; hx, hy, hz, Npx, Npy, Npz) = setup.grid
    if any(!isequal((:periodic, :periodic)), [bc.u.x, bc.v.y, bc.w.z])
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end
    if mapreduce(h -> maximum(abs.(diff(h))) > 1e-14, |, [hx, hy, hz])
        error("FourierPressureSolver requires uniform grid in each dimension")
    end
    Δx = hx[1]
    Δy = hy[1]
    Δz = hz[1]
    if any(!≈(Δx), hx) || any(!≈(Δy), hy) || any(!≈(Δz), hz)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx, Δy and Δz do not need to be the same
    i = 0:(Npx-1)
    j = reshape(0:(Npy-1), 1, :)
    k = reshape(0:(Npz-1), 1, 1, :)

    # Scale with Δx*Δy*Δz, since we solve the PDE in integrated form
    Â = @. 4 * Δx * Δy * Δz * (
        sin(i * π / Npx)^2 / Δx^2 +
        sin(j * π / Npy)^2 / Δy^2 +
        sin(k * π / Npz)^2 / Δz^2
    )

    # Pressure is determined up to constant, fix at 0
    Â[1] = 1

    Â = complex(Â)

    # Placeholders for intermediate results
    p̂ = similar(Â)
    f̂ = similar(Â)

    @pack! solver = Â, p̂, f̂
end
