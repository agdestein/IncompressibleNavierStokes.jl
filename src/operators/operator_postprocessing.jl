"""
    operator_postprocessing!(setup)

Construct postprocessing operators such as vorticity.
"""
function operator_postprocessing!(setup)
    # Boundary conditions
    @unpack bc = setup
    @unpack Nx, Ny = setup.grid
    @unpack hx, hy, gx, gy = setup.grid
    @unpack gxi, gyi, gxd, gyd = setup.grid

    # FIXME: 3D implementation

    ## Vorticity

    # Operators act on internal points only

    # Du/dy, like Su_uy
    diag1 = 1 ./ gy[2:end-1]
    W1D = spdiagm(Ny - 1, Ny, 0 => -diag1, 1 => diag1)
    # Extend to 2D
    Wu_uy = kron(W1D, sparse(I, Nx - 1, Nx - 1))

    # Dv/dx, like Sv_vx
    diag1 = 1 ./ gx[2:end-1]
    W1D = spdiagm(Nx - 1, Nx, 0 => -diag1, 1 => diag1)
    # Extend to 2D
    Wv_vx = kron(sparse(I, Ny - 1, Ny - 1), W1D)

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

    @pack! setup.discretization = Wv_vx, Wu_uy

    setup
end
