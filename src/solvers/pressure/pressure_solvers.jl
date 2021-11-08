"""
    PressureSolver

Pressure solver for the Poisson equation.
"""
abstract type PressureSolver{T} end

Base.@kwdef mutable struct DirectPressureSolver{T} <: PressureSolver{T}
    A_fact::Factorization{T} = cholesky(spzeros(T, 0, 0))
end
Base.@kwdef mutable struct CGPressureSolver{T} <: PressureSolver{T}
    abstol::T = 0
    reltol::T = √eps(T)
    maxiter::Int = 0 
end
Base.@kwdef mutable struct FourierPressureSolver{T} <: PressureSolver{T}
    Â = zeros(Complex{T}, 0, 0)
    p̂ = zeros(Complex{T}, 0, 0)
    f̂ = zeros(Complex{T}, 0, 0)
end

"""
    initialize!(pressure_solver)

Initialize pressure solver.
"""
function initialize! end

initialize!(solver::DirectPressureSolver, setup, A) = (solver.A_fact = factorize(A))

function initialize!(solver::CGPressureSolver, setup, A)
    solver.maxiter == 0 && (solver.maxiter = size(A, 2))
end

function initialize!(solver::FourierPressureSolver, setup, A)
    @unpack bc = setup
    @unpack hx, hy, Npx, Npy = setup.grid
    if any(!isequal(:periodic), [bc.v.y[1], bc.v.y[2], bc.u.x[1], bc.u.x[1]])
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end
    if maximum(abs.(diff(hx))) > 1e-14 || maximum(abs.(diff(hy))) > 1e-14
        error("FourierPressureSolver requires uniform grid in each dimension")
    end
    Δx = hx[1]
    Δy = hy[1]
    if any(!≈(Δx), hx) || any(!≈(Δy), hy)
        error("FourierPressureSolver requires uniform grid along each dimension")
    end 

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx, Δy and Δz do not need to be the same
    i = 0:(Npx-1)
    j = 0:(Npy-1)

    # Scale with Δx*Δy*Δz, since we solve the PPE in integrated form
    Â = @. 4 * Δx * Δy * (sin(i * π / Npx)^2 / Δx^2 + sin(j' * π / Npy)^2 / Δy^2)

    # Pressure is determined up to constant, fix at 0
    Â[1, 1] = 1

    Â = complex(Â)

    # Placeholders for intermediate results
    p̂ = similar(Â)
    f̂ = similar(Â)

    @pack! solver = Â, p̂, f̂
end
