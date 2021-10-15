"""
    PressureSolver

Lineare pressure solver for the Poisson equation.
"""
abstract type PressureSolver end

struct DirectPressureSolver <: PressureSolver end
struct CGPressureSolver <: PressureSolver end
struct FourierPressureSolver <: PressureSolver end

"""
    initialize(pressure_solver)

Initialize pressure solver.
"""
function initialize! end

initialize!(::DirectPressureSolver, setup, A) = (setup.discretization.A_fact = factorize(A))
initialize!(::CGPressureSolver, setup, A) = nothing

function initialize!(::FourierPressureSolver, setup, A)
    @unpack bc = setup
    @unpack hx, hy, Npx, Npy = setup.grid
    if any(!isequal(:periodic), [bc.v.low, bc.v.up, bc.u.left, bc.u.left])
        error("FourierPressureSolver only implemented for periodic boundary conditions")
    end
    if maximum(abs.(diff(hx))) > 1e-14 || maximum(abs.(diff(hy))) > 1e-14
        error("FourierPressureSolver requires uniform grid in each dimension")
    end
    Δx = hx[1]
    Δy = hy[1]

    # Fourier transform of the discretization
    # Assuming uniform grid, although Δx, Δy and Δz do not need to be the same
    i = 0:(Npx-1)
    j = 0:(Npy-1)

    # Scale with Δx*Δy*Δz, since we solve the PPE in integrated form
    Â = @. 4 * Δx * Δy * (sin(i * π / Npx)^2 / Δx^2 + sin(j' * π / Npy)^2 / Δy^2)

    # Pressure is determined up to constant, fix at 0
    Â[1, 1] = 1

    @pack! setup.solver_settings = Â
end
