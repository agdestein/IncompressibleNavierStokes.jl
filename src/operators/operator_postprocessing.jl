"""
    operator_postprocessing(dimension, grid, boundary_conditions)

Construct postprocessing operators such as vorticity.
"""
function operator_postprocessing end

# 2D version
function operator_postprocessing(::Dimension{2}, grid, boundary_conditions)
    (; Nx, Ny, gx, gy, gxd, gyd) = grid

    T = eltype(gx)

    if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
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
        weight = T(1)
        repeat_x = spdiagm(Nx + 1, Nx, -Nx => [weight], 0 => fill(weight, Nx))

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
        weight = T(1)
        repeat_y = spdiagm(Ny + 1, Ny, -Ny => [weight], 0 => fill(weight, Ny))
    else
        # dv/dx, like Sv_vx
        diag = 1 ./ gy[2:(end-1)]
        ∂y = spdiagm(Ny - 1, Ny, 0 => -diag, 1 => diag)
        repeat_y = I(Ny - 1)

        # du/dy, like Su_uy
        diag = 1 ./ gx[2:(end-1)]
        ∂x = spdiagm(Nx - 1, Nx, 0 => -diag, 1 => diag)
        repeat_x = I(Nx - 1)
    end

    # Extend to 2D
    Wu_uy = ∂y ⊗ repeat_x
    Wv_vx = repeat_y ⊗ ∂x

    (; Wv_vx, Wu_uy)
end

# 3D version
function operator_postprocessing(::Dimension{3}, grid, boundary_conditions)
    (; Nx, Ny, Nz, gx, gy, gz, gxd, gyd, gzd) = grid

    T = eltype(gx)

    if all(
        ==(:periodic),
        (
            boundary_conditions.u.x[1],
            boundary_conditions.v.y[1],
            boundary_conditions.w.z[1],
        ),
    )
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
        weight = T(1 // 2)
        average_x = spdiagm(
            Nx + 1,
            Nx,
            -Nx => [weight],
            -1 => fill(weight, Nx),
            0 => fill(weight, Nx),
            Nx - 1 => [weight],
        )
        weight = T(1)
        repeat_x = spdiagm(Nx + 1, Nx, -Nx => [weight], 0 => fill(weight, Nx))

        diag = 1 ./ gyd
        ∂y = spdiagm(
            Ny + 1,
            Ny,
            -Ny => diag[[1]],
            -1 => -diag[1:(end-1)],
            0 => diag[1:(end-1)],
            Ny - 1 => -diag[[end - 1]],
        )
        weight = T(1 // 2)
        average_y = spdiagm(
            Ny + 1,
            Ny,
            -Ny => [weight],
            -1 => fill(weight, Ny),
            0 => fill(weight, Ny),
            Ny - 1 => [weight],
        )
        weight = T(1)
        repeat_y = spdiagm(Ny + 1, Ny, -Ny => [weight], 0 => fill(weight, Ny))

        diag = 1 ./ gzd
        ∂z = spdiagm(
            Nz + 1,
            Nz,
            -Nz => diag[[1]],
            -1 => -diag[1:(end-1)],
            0 => diag[1:(end-1)],
            Nz - 1 => -diag[[end - 1]],
        )
        weight = T(1 // 2)
        average_z = spdiagm(
            Nz + 1,
            Nz,
            -Nz => [weight],
            -1 => fill(weight, Nz),
            0 => fill(weight, Nz),
            Nz - 1 => [weight],
        )
        weight = T(1)
        repeat_z = spdiagm(Nz + 1, Nz, -Nz => [weight], 0 => fill(weight, Nz))
    else
        diag = 1 ./ gx[2:(end-1)]
        ∂x = spdiagm(Nx - 1, Nx, 0 => -diag, 1 => diag)
        diag = fill(T(1 // 2), Nx - 1)
        average_x = spdiagm(Nx - 1, Nx, 0 => diag, 1 => diag)
        # average_x = sparse(I, Nx - 1, Nx)
        repeat_x = I(Nx - 1)

        diag = 1 ./ gy[2:(end-1)]
        ∂y = spdiagm(Ny - 1, Ny, 0 => -diag, 1 => diag)
        diag = fill(T(1 // 2), Ny - 1)
        average_y = spdiagm(Ny - 1, Ny, 0 => diag, 1 => diag)
        # average_y = sparse(I, Ny - 1, Ny)
        repeat_y = I(Ny - 1)

        diag = 1 ./ gz[2:(end-1)]
        ∂z = spdiagm(Nz - 1, Nz, 0 => -diag, 1 => diag)
        diag = fill(T(1 / 2), Nz - 1)
        average_z = spdiagm(Nz - 1, Nz, 0 => diag, 1 => diag)
        # average_z = sparse(I, Nz - 1, Nz)
        repeat_z = I(Nz - 1)
    end

    # Extend to 3D
    Wu_uy = average_z ⊗ ∂y ⊗ repeat_x
    Wu_uz = ∂z ⊗ average_y ⊗ repeat_x
    Wv_vx = average_z ⊗ repeat_y ⊗ ∂x
    Wv_vz = ∂z ⊗ repeat_y ⊗ average_x
    Ww_wy = repeat_z ⊗ ∂y ⊗ average_x
    Ww_wx = repeat_z ⊗ average_y ⊗ ∂x

    (; Wu_uy, Wu_uz, Wv_vx, Wv_vz, Ww_wy, Ww_wx)
end
