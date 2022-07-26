"""
    operator_averaging!(setup)

Construct averaging operators.
"""
function operator_averaging! end

# 2D version
function operator_averaging!(setup::Setup{T,2}) where {T}
    (; grid, operators) = setup
    (; Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t) = grid
    (; Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t) = grid
    (; hx, hy) = grid

    # Averaging weight:
    weight = 1 / 2

    ## Averaging operators, u-component

    ## Au_ux: evaluate u at ux location
    diag1 = weight * ones(Nux_t - 1)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, hx[1], hx[end])

    # Extend to 2D
    Au_ux = I(Nuy_in) ⊗ (A1D * Au_ux_bc.B1D)
    Au_ux_bc = (; Au_ux_bc..., Bbc = I(Nuy_in) ⊗ (A1D * Au_ux_bc.Btemp))

    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t - 1)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, hy[1], hy[end])

    # Extend to 2D
    Au_uy = (A1D * Au_uy_bc.B1D) ⊗ I(Nux_in)
    Au_uy_bc = (; Au_uy_bc..., Bbc = (A1D * Au_uy_bc.Btemp) ⊗ I(Nux_in))

    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vx_bc = bc_general_stag(Nvx_t, Nvx_in, Nvx_b, hx[1], hx[end])

    # Extend to 2D
    Av_vx = I(Nvy_in) ⊗ (A1D * Av_vx_bc.B1D)
    Av_vx_bc = (; Av_vx_bc..., Bbc = I(Nvy_in) ⊗ (A1D * Av_vx_bc.Btemp))

    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t - 1)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, hy[1], hy[end])

    # Extend to 2D
    Av_vy = (A1D * Av_vy_bc.B1D) ⊗ I(Nvx_in)
    Av_vy_bc = (; Av_vy_bc..., Bbc = (A1D * Av_vy_bc.Btemp) ⊗ I(Nvx_in))

    ## Store in setup structure
    @pack! operators = Au_ux, Au_uy, Av_vx, Av_vy
end

# 3D version
function operator_averaging!(setup::Setup{T,3}) where {T}
    (; grid, operators) = setup
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
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, hx[1], hx[end])

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
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, hy[1], hy[end])

    # Extend to 3D
    Au_uy = I(Nuz_in) ⊗ (A1D * Au_uy_bc.B1D) ⊗ I(Nux_in)
    Au_uy_bc = (; Au_uy_bc..., Bbc = I(Nuz_in) ⊗ (A1D * Au_uy_bc.Btemp) ⊗ I(Nux_in))


    ## Au_uz: evaluate u at uz location
    diag1 = weight * ones(Nuz_t - 1)
    A1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uz_bc = bc_general_stag(Nuz_t, Nuz_in, Nuz_b, hz[1], hz[end])

    # Extend to 3D
    Au_uz = (A1D * Au_uz_bc.B1D) ⊗ I(Nuy_in) ⊗ I(Nux_in)
    Au_uz_bc = (; Au_uz_bc..., Bbc = (A1D * Au_uz_bc.Btemp) ⊗ I(Nuy_in) ⊗ I(Nux_in))


    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vx_bc = bc_general_stag(Nvx_t, Nvx_in, Nvx_b, hx[1], hx[end])

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
    Av_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, hy[1], hy[end])

    # Extend to 3D
    Av_vy = I(Nvz_in) ⊗ (A1D * Av_vy_bc.B1D) ⊗ I(Nvx_in)
    Av_vy_bc = (; Av_vy_bc..., Bbc = I(Nvz_in) ⊗ (A1D * Av_vy_bc.Btemp) ⊗ I(Nvx_in))

    ## Av_vz: evalvate v at vz location
    diag1 = weight * ones(Nvz_t - 1)
    A1D = spdiagm(Nvz_t - 1, Nvz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vz_bc = bc_general_stag(Nvz_t, Nvz_in, Nvz_b, hz[1], hz[end])

    # Extend to 3D
    Av_vz = (A1D * Av_vz_bc.B1D) ⊗ I(Nvy_in) ⊗ I(Nvx_in)
    Av_vz_bc = (; Av_vz_bc..., Bbc = (A1D * Av_vz_bc.Btemp) ⊗ I(Nvy_in) ⊗ I(Nvx_in))


    ## Averaging operators, w-component

    ## Aw_wx: evaluate w at wx location
    diag1 = weight * ones(Nwx_t - 1)
    A1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wx_bc = bc_general_stag(Nwx_t, Nwx_in, Nwx_b, hx[1], hx[end])

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
    Aw_wy_bc = bc_general_stag(Nwy_t, Nwy_in, Nwy_b, hy[1], hy[end])

    # Extend to 3D
    Aw_wy = I(Nwz_in) ⊗ (A1D * Aw_wy_bc.B1D) ⊗ I(Nwx_in)
    Aw_wy_bc = (; Aw_wy_bc..., Bbc = I(Nwz_in) ⊗ (A1D * Aw_wy_bc.Btemp) ⊗ I(Nwx_in))

    ## Aw_wz: evaluate w at wz location
    diag1 = weight * ones(Nwz_t - 1)
    A1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wz_bc = bc_general(Nwz_t, Nwz_in, Nwz_b, hz[1], hz[end])

    # Extend to 3D
    Aw_wz = (A1D * Aw_wz_bc.B1D) ⊗ I(Nwy_in) ⊗ I(Nwx_in)
    Aw_wz_bc = (; Aw_wz_bc..., Bbc = kron((A1D * Aw_wz_bc.Btemp), kron(I(Nwy_in), I(Nwx_in))))


    ## Store in setup structure
    @pack! operators = Au_ux, Au_uy, Au_uz
    @pack! operators = Av_vx, Av_vy, Av_vz
    @pack! operators = Aw_wx, Aw_wy, Aw_wz

    setup
end
