"""
    operator_postprocessing!(setup)

Construct postprocessing operators such as vorticity.
"""
function operator_postprocessing! end

# 2D version
function operator_postprocessing!(setup::Setup{T,2}) where {T}
    # Boundary conditions
    (; grid, operators) = setup
    (; Nx, Ny, gx, gy, gxd, gyd) = grid

    # For entirely periodic BC, covering entire mesh

    # dv/dx, like Sv_vx
    diag = 1 ./ gxd
    ∂x = spdiagm(
        Nx + 1,
        Nx,
        -Nx => diag[[1]],
        -1 => -diag[1:(end-1)],
        0 => diag[1:(end-1)],
        Nx - 1 => -diag[[end - 1]],
    )
    repeat_x = spdiagm(Nx + 1, Nx, -Nx => [1], 0 => ones(Nx))

    # du/dy, like Su_uy
    diag = 1 ./ gyd
    ∂y = spdiagm(
        Ny + 1,
        Ny,
        -Ny => diag[[1]],
        -1 => -diag[1:(end-1)],
        0 => diag[1:(end-1)],
        Ny - 1 => -diag[[end - 1]],
    )
    repeat_y = spdiagm(Ny + 1, Ny, -Ny => [1], 0 => ones(Ny))

    # Extend to 2D
    Wu_uy = ∂y ⊗ repeat_x
    Wv_vx = repeat_y ⊗ ∂x

    @pack! operators = Wv_vx, Wu_uy

    setup
end

# 3D version
function operator_postprocessing!(setup::Setup{T,3}) where {T}
    # Boundary conditions
    (; grid, operators) = setup
    (; Nx, Ny, Nz, gx, gy, gz, gxd, gyd, gzd) = grid

    # For entirely periodic BC, covering entire mesh

    diag = 1 ./ gxd
    ∂x = spdiagm(
        Nx + 1,
        Nx,
        -Nx => diag[[1]],
        -1 => -diag[1:(end-1)],
        0 => diag[1:(end-1)],
        Nx - 1 => -diag[[end - 1]],
    )
    # FIXME: nonuniform weights: 1/gi / (1/gi + 1/gj) ?
    diag = fill(1 / 2, Nx)
    average_x = spdiagm(Nx + 1, Nx, -Nx => [1 / 2], -1 => diag, 0 => diag, Nx - 1 => [1 / 2])
    repeat_x = spdiagm(Nx + 1, Nx, -Nx => [1], 0 => ones(Nx))

    diag = 1 ./ gyd
    ∂y = spdiagm(
        Ny + 1,
        Ny,
        -Ny => diag[[1]],
        -1 => -diag[1:(end-1)],
        0 => diag[1:(end-1)],
        Ny - 1 => -diag[[end - 1]],
    )
    diag = fill(1 / 2, Ny)
    average_y = spdiagm(Ny + 1, Ny, -Ny => [1 / 2], -1 => diag, 0 => diag, Ny - 1 => [1 / 2])
    repeat_y = spdiagm(Ny + 1, Ny, -Ny => [1], 0 => ones(Ny))

    diag = 1 ./ gzd
    ∂z = spdiagm(
        Nz + 1,
        Nz,
        -Nz => diag[[1]],
        -1 => -diag[1:(end-1)],
        0 => diag[1:(end-1)],
        Nz - 1 => -diag[[end - 1]],
    )
    diag = fill(1 / 2, Nz)
    average_z = spdiagm(Nz + 1, Nz, -Nz => [1 / 2], -1 => diag, 0 => diag, Nz - 1 => [1 / 2])
    repeat_z = spdiagm(Nz + 1, Nz, -Nz => [1], 0 => ones(Nz))

    # Extend to 3D
    Wu_uy = average_z ⊗ ∂y ⊗ repeat_x
    Wu_uz = ∂z ⊗ average_y ⊗ repeat_x
    Wv_vx = average_z ⊗ repeat_y ⊗ ∂x
    Wv_vz = ∂z ⊗ repeat_y ⊗ average_x
    Ww_wy = repeat_z ⊗ ∂y ⊗ average_x
    Ww_wx = repeat_z ⊗ average_y ⊗ ∂x

    @pack! operators = Wu_uy, Wu_uz, Wv_vx, Wv_vz, Ww_wy, Ww_wx

    setup
end
