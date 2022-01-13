"""
    operator_postprocessing!(setup)

Construct postprocessing operators such as vorticity.
"""
function operator_postprocessing! end

# 2D version
function operator_postprocessing!(setup::Setup{T,2}) where {T}
    # Boundary conditions
    (; grid, operators, bc) = setup
    (; Nx, Ny) = grid
    (; gx, gy) = grid
    (; gxd, gyd) = grid

    ## Vorticity

    # Operators act on internal points only

    # Du/dy, like Su_uy
    diag1 = 1 ./ gy[2:end-1]
    W1D = spdiagm(Ny - 1, Ny, 0 => -diag1, 1 => diag1)

    # Extend to 2D
    Wu_uy = kron(W1D, I(Nx - 1))

    # Dv/dx, like Sv_vx
    diag1 = 1 ./ gx[2:end-1]
    W1D = spdiagm(Nx - 1, Nx, 0 => -diag1, 1 => diag1)

    # Extend to 2D
    Wv_vx = kron(I(Ny - 1), W1D)

    ## For periodic BC, covering entire mesh
    if bc.u.x[1] == :periodic && bc.v.y[1] == :periodic
        # Du/dy, like Su_uy
        diag1 = 1 ./ gyd
        W1D = spdiagm(
            Ny + 1,
            Ny,
            -Ny => diag1[[1]],
            -1 => -diag1[1:end-1],
            0 => diag1[1:end-1],
            Ny - 1 => -diag1[[end - 1]],
        )

        # Extend to 2D
        diag2 = ones(Nx)
        Wu_uy = kron(W1D, spdiagm(Nx + 1, Nx, -Nx => diag2[[1]], 0 => diag2))

        # Dv/dx, like Sv_vx
        diag1 = 1 ./ gxd
        W1D = spdiagm(
            Nx + 1,
            Nx,
            -Nx => diag1[[1]],
            -1 => -diag1[1:end-1],
            0 => diag1[1:end-1],
            Nx - 1 => -diag1[[end - 1]],
        )

        # Extend to 2D
        diag2 = ones(Ny)
        Wv_vx = kron(spdiagm(Ny + 1, Ny, -Ny => diag2[[1]], 0 => diag2), W1D)
    end

    @pack! operators = Wv_vx, Wu_uy

    setup
end

# 3D version
function operator_postprocessing!(setup::Setup{T,3}) where {T}
    # Boundary conditions
    @warn "3D version of operator_postprocessing! not implemented"

    setup
end
