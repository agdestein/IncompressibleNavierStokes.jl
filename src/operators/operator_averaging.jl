"""
    operator_averaging(grid, bc)

Construct averaging operators.
"""
function operator_averaging end

# 2D version
function operator_averaging(grid::Grid{T,2}, bc) where {T}
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t) = grid
    (; Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t) = grid
    (; hx, hy) = grid
    (; order4) = grid

    # Averaging weight:
    weight = 1 / 2

    ## Averaging operators, u-component

    ## Au_ux: evaluate u at ux location
    diag1 = weight * ones(Nux_t - 1)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    # Extend to 2D
    Au_ux = I(Nuy_in) ⊗ (A1D * Au_ux_bc.B1D)
    Au_ux_bc = (; Au_ux_bc..., Bbc = I(Nuy_in) ⊗ (A1D * Au_ux_bc.Btemp))

    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t - 1)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    # Extend to 2D
    Au_uy = (A1D * Au_uy_bc.B1D) ⊗ I(Nux_in)
    Au_uy_bc = (; Au_uy_bc..., Bbc = (A1D * Au_uy_bc.Btemp) ⊗ I(Nux_in))

    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vx_bc = bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Extend to 2D
    Av_vx = I(Nvy_in) ⊗ (A1D * Av_vx_bc.B1D)
    Av_vx_bc = (; Av_vx_bc..., Bbc = I(Nvy_in) ⊗ (A1D * Av_vx_bc.Btemp))

    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t - 1)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    # Extend to 2D
    Av_vy = (A1D * Av_vy_bc.B1D) ⊗ I(Nvx_in)
    Av_vy_bc = (; Av_vy_bc..., Bbc = (A1D * Av_vy_bc.Btemp) ⊗ I(Nvx_in))

    ## Fourth order
    if order4
        ## Au_ux: evaluate u at ux location
        diag1 = weight * ones(Nux_in + 3)
        A1D3 = spdiagm(Nux_in + 3, Nux_t + 4, 0 => diag1, 3 => diag1)

        # Boundary conditions
        Au_ux_bc3 = bc_av3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.x[1],
            bc.u.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Au_ux3 = I(Nuy_in) ⊗ (A1D3 * Au_ux_bc3.B1D)
        Au_ux_bc3 = (; Au_ux_bc3..., Bbc = I(Nuy_in) ⊗ (A1D3 * Au_ux_bc3.Btemp))

        ## Au_uy: evaluate u at uy location
        diag1 = weight * ones(Nuy_in + 3)
        A1D3 = spdiagm(Nuy_in + 3, Nuy_t + 4, 0 => diag1, 3 => diag1)

        # Boundary conditions
        Au_uy_bc3 = bc_av_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.y[1],
            bc.u.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Au_uy3 = (A1D3 * Au_uy_bc3.B1D) ⊗ I(Nux_in)
        Au_uy_bc3 = (; Au_uy_bc3..., Bbc = (A1D3 * Au_uy_bc3.Btemp) ⊗ I(Nux_in))

        ## Av_vx: evaluate v at vx location
        diag1 = weight * ones(Nvx_in + 3)
        A1D3 = spdiagm(Nvx_in + 3, Nvx_t + 4, 0 => diag1, 3 => diag1)

        # Boundary conditions
        Av_vx_bc3 = bc_av_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.x[1],
            bc.v.x[2],
            hx[1],
            hx[end],
        )

        # Extend to 2D
        Av_vx3 = I(Nvy_in) ⊗ (A1D3 * Av_vx_bc3.B1D)
        Av_vx_bc3 = (; Av_vx_bc3..., Bbc = I(Nvy_in) ⊗ (A1D3 * Av_vx_bc3.Btemp))

        ## Av_vy: evaluate v at vy location
        diag1 = weight * ones(Nvy_in + 3)
        A1D3 = spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => diag1, 3 => diag1)

        # Boundary conditions
        Av_vy_bc3 = bc_av3(
            Nvy_t + 4,
            Nvy_in,
            Nvy_t + 4 - Nvy_in,
            bc.v.y[1],
            bc.v.y[2],
            hy[1],
            hy[end],
        )

        # Extend to 2D
        Av_vy3 = (A1D3 * Av_vy_bc3.B1D) ⊗ I(Nvx_in)
        Av_vy_bc3 = (; Av_vy_bc3..., Bbc = (A1D3 * Av_vy_bc3.Btemp) ⊗ I(Nvx_in))
    end

    ## Group operators
    operators = (; Au_ux, Au_uy, Av_vx, Av_vy, Au_ux_bc, Au_uy_bc, Av_vx_bc, Av_vy_bc)

    if order4
        operators = (;
            operators...,
            Au_ux3, Au_uy3, Av_vx3, Av_vy3,
            Au_ux_bc3, Au_uy_bc3, Av_vx_bc3, Av_vy_bc3,
        )
    end

    operators
end

# 3D version
function operator_averaging(grid::Grid{T,3}, bc) where {T}
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t, Nuz_in, Nuz_b, Nuz_t) = grid
    (; Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t, Nvz_in, Nvz_b, Nvz_t) = grid
    (; Nwx_in, Nwx_b, Nwx_t, Nwy_in, Nwy_b, Nwy_t, Nwz_in, Nwz_b, Nwz_t) = grid
    (; hx, hy, hz) = grid

    # Averaging weight:
    weight = 1 / 2


    ## Averaging operators, u-component

    ## Au_ux: evaluate u at ux location
    diag1 = weight * ones(Nux_t - 1)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    # Extend to 3D
    Au_ux = I(Nuz_in) ⊗ I(Nuy_in) ⊗ (A1D * Au_ux_bc.B1D)
    Au_ux_bc = (;
        Au_ux_bc...,
        Bbc = I(Nuz_in) ⊗ I(Nuy_in) ⊗ (A1D * Au_ux_bc.Btemp)
    )


    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t - 1)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    # Extend to 3D
    Au_uy = I(Nuz_in) ⊗ (A1D * Au_uy_bc.B1D) ⊗ I(Nux_in)
    Au_uy_bc = (; Au_uy_bc..., Bbc = I(Nuz_in) ⊗ (A1D * Au_uy_bc.Btemp) ⊗ I(Nux_in))


    ## Au_uz: evaluate u at uz location
    diag1 = weight * ones(Nuz_t - 1)
    A1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uz_bc = bc_general_stag(Nuz_t, Nuz_in, Nuz_b, bc.u.z[1], bc.u.z[2], hz[1], hz[end])

    # Extend to 3D
    Au_uz = (A1D * Au_uz_bc.B1D) ⊗ I(Nuy_in) ⊗ I(Nux_in)
    Au_uz_bc = (; Au_uz_bc..., Bbc = (A1D * Au_uz_bc.Btemp) ⊗ I(Nuy_in) ⊗ I(Nux_in))


    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vx_bc = bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Extend to 3D
    Av_vx = I(Nvz_in) ⊗ I(Nvy_in) ⊗ (A1D * Av_vx_bc.B1D)
    Av_vx_bc = (;
        Av_vx_bc...,
        Bbc = I(Nvz_in) ⊗ I(Nvy_in) ⊗ (A1D * Av_vx_bc.Btemp)
    )

    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t - 1)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    # Extend to 3D
    Av_vy = I(Nvz_in) ⊗ (A1D * Av_vy_bc.B1D) ⊗ I(Nvx_in)
    Av_vy_bc = (; Av_vy_bc..., Bbc = I(Nvz_in) ⊗ (A1D * Av_vy_bc.Btemp) ⊗ I(Nvx_in))

    ## Av_vz: evalvate v at vz location
    diag1 = weight * ones(Nvz_t - 1)
    A1D = spdiagm(Nvz_t - 1, Nvz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vz_bc = bc_general_stag(Nvz_t, Nvz_in, Nvz_b, bc.v.z[1], bc.v.z[2], hz[1], hz[end])

    # Extend to 3D
    Av_vz = (A1D * Av_vz_bc.B1D) ⊗ I(Nvy_in) ⊗ I(Nvx_in)
    Av_vz_bc = (; Av_vz_bc..., Bbc = (A1D * Av_vz_bc.Btemp) ⊗ I(Nvy_in) ⊗ I(Nvx_in))


    ## Averaging operators, w-component

    ## Aw_wx: evaluate w at wx location
    diag1 = weight * ones(Nwx_t - 1)
    A1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wx_bc = bc_general_stag(Nwx_t, Nwx_in, Nwx_b, bc.w.x[1], bc.w.x[2], hx[1], hx[end])

    # Extend to 3D
    Aw_wx = I(Nwz_in) ⊗ I(Nwy_in) ⊗ (A1D * Aw_wx_bc.B1D)
    Aw_wx_bc = (;
        Aw_wx_bc...,
        Bbc = I(Nwz_in) ⊗ I(Nwy_in) ⊗ (A1D * Aw_wx_bc.Btemp)
    )

    ## Aw_wy: evaluate w at wy location
    diag1 = weight * ones(Nwy_t - 1)
    A1D = spdiagm(Nwy_t - 1, Nwy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wy_bc = bc_general_stag(Nwy_t, Nwy_in, Nwy_b, bc.w.y[1], bc.w.y[2], hy[1], hy[end])

    # Extend to 3D
    Aw_wy = I(Nwz_in) ⊗ (A1D * Aw_wy_bc.B1D) ⊗ I(Nwx_in)
    Aw_wy_bc = (; Aw_wy_bc..., Bbc = I(Nwz_in) ⊗ (A1D * Aw_wy_bc.Btemp) ⊗ I(Nwx_in))

    ## Aw_wz: evaluate w at wz location
    diag1 = weight * ones(Nwz_t - 1)
    A1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])

    # Extend to 3D
    Aw_wz = (A1D * Aw_wz_bc.B1D) ⊗ I(Nwy_in) ⊗ I(Nwx_in)
    Aw_wz_bc = (; Aw_wz_bc..., Bbc = kron((A1D * Aw_wz_bc.Btemp), kron(I(Nwy_in), I(Nwx_in))))


    ## Group operators
    (;
        Au_ux, Au_uy, Au_uz,
        Av_vx, Av_vy, Av_vz,
        Aw_wx, Aw_wy, Aw_wz,
        Au_ux_bc, Au_uy_bc, Au_uz_bc,
        Av_vx_bc, Av_vy_bc, Av_vz_bc,
        Aw_wx_bc, Aw_wy_bc, Aw_wz_bc,
    )
end
