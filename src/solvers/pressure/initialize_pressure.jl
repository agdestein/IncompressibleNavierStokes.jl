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

# Fourier, 2D version
function initialize!(solver::FourierPressureSolver, setup::Setup{T,2}, A) where {T}
    (; hx, hy, Npx, Npy) = setup.grid
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
    Ahat = @. 4 * Δx * Δy * (
        sin(i * π / Npx)^2 / Δx^2 +
        sin(j * π / Npy)^2 / Δy^2
    )

    # Pressure is determined up to constant, fix at 0
    Ahat[1] = 1

    Ahat = complex(Ahat)

    # Placeholders for intermediate results
    phat = similar(Ahat)
    fhat = similar(Ahat)

    @pack! solver = Ahat, phat, fhat
end

# Fourier, 3D version
function initialize!(solver::FourierPressureSolver, setup::Setup{T,3}, A) where {T}
    (; hx, hy, hz, Npx, Npy, Npz) = setup.grid
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
    Ahat = @. 4 * Δx * Δy * Δz * (
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

    @pack! solver = Ahat, phat, fhat
end
