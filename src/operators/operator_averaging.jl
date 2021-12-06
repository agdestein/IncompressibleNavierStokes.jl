"""
    operator_averaging!(setup)
Construct averaging operators.
"""
function operator_averaging!(setup)
    @unpack bc = setup
    @unpack Nux_in, Nux_b, Nux_t, Nuy_in, Nuy_b, Nuy_t, Nuz_in, Nuz_b, Nuz_t  = setup.grid
    @unpack Nvx_in, Nvx_b, Nvx_t, Nvy_in, Nvy_b, Nvy_t, Nvz_in, Nvz_b, Nvz_t  = setup.grid
    @unpack Nwx_in, Nwx_b, Nwx_t, Nwy_in, Nwy_b, Nwy_t, Nwz_in, Nwz_b, Nwz_t  = setup.grid
    @unpack hx, hy, hz = setup.grid

    # Averaging weight:
    weight = 1 / 2


    ## Averaging operators, u-component

    ## Au_ux: evaluate u at ux location
    diag1 = weight * ones(Nux_t - 1)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.x[1], bc.u.x[2], hx[1], hx[end])

    # Extend to 3D
    Au_ux = kron(sparse(I, Nuz_in, Nuz_in), kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_bc.B1D))
    Au_ux_bc = (;
        Au_ux_bc...,
        Bbc = kron(sparse(I, Nuz_in, Nuz_in), kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_bc.Btemp))
    )


    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t - 1)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.y[1], bc.u.y[2], hy[1], hy[end])

    # Extend to 3D
    Au_uy = kron(sparse(I, Nuz_in, Nuz_in), kron(A1D * Au_uy_bc.B1D, sparse(I, Nux_in, Nux_in)))
    Au_uy_bc = (; Au_uy_bc..., Bbc = kron(sparse(I, Nuz_in, Nuz_in), kron(A1D * Au_uy_bc.Btemp, sparse(I, Nux_in, Nux_in))))


    ## Au_uz: evaluate u at uz location
    diag1 = weight * ones(Nuz_t - 1)
    A1D = spdiagm(Nuz_t - 1, Nuz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Au_uz_bc = bc_general_stag(Nuz_t, Nuz_in, Nuz_b, bc.u.z[1], bc.u.z[2], hz[1], hz[end])

    # Extend to 3D
    Au_uz = kron(A1D * Au_uz_bc.B1D, kron(sparse(I, Nuy_in, Nuy_in), sparse(I, Nux_in, Nux_in)))
    Au_uz_bc = (; Au_uz_bc..., Bbc = kron(A1D * Au_uz_bc.Btemp, kron(sparse(I, Nuy_in, Nuy_in), sparse(I, Nux_in, Nux_in))))


    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vx_bc = bc_general(Nvx_t, Nvx_in, Nvx_b, bc.v.x[1], bc.v.x[2], hx[1], hx[end])

    # Extend to 3D
    Av_vx = kron(sparse(I, Nvz_in, Nvz_in), kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_bc.B1D))
    Av_vx_bc = (;
        Av_vx_bc...,
        Bbc = kron(sparse(I, Nvz_in, Nvz_in), kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_bc.Btemp))
    )

    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t - 1)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vy_bc = bc_general_stag(Nvy_t, Nvy_in, Nvy_b, bc.v.y[1], bc.v.y[2], hy[1], hy[end])

    # Extend to 3D
    Av_vy = kron(sparse(I, Nvz_in, Nvz_in), kron(A1D * Av_vy_bc.B1D, sparse(I, Nvx_in, Nvx_in)))
    Av_vy_bc = (; Av_vy_bc..., Bbc = kron(sparse(I, Nvz_in, Nvz_in), kron(A1D * Av_vy_bc.Btemp, sparse(I, Nvx_in, Nvx_in))))

    ## Av_vz: evalvate v at vz location
    diag1 = weight * ones(Nvz_t - 1)
    A1D = spdiagm(Nvz_t - 1, Nvz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Av_vz_bc = bc_general_stag(Nvz_t, Nvz_in, Nvz_b, bc.v.z[1], bc.v.z[2], hz[1], hz[end])

    # Extend to 3D
    Av_vz = kron(A1D * Av_vz_bc.B1D, kron(sparse(I, Nvy_in, Nvy_in), sparse(I, Nvx_in, Nvx_in)))
    Av_vz_bc = (; Av_vz_bc..., Bbc = kron(A1D * Av_vz_bc.Btemp, kron(sparse(I, Nvy_in, Nvy_in), sparse(I, Nvx_in, Nvx_in))))


    ## Averaging operators, w-component

    ## Aw_wx: evaluate w at wx location
    diag1 = weight * ones(Nwx_t - 1)
    A1D = spdiagm(Nwx_t - 1, Nwx_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wx_bc = bc_general(Nwx_t, Nwx_in, Nwx_b, bc.w.x[1], bc.w.x[2], hx[1], hx[end])

    # Extend to 3D
    Aw_wx = kron(sparse(I, Nwz_in, Nwz_in), kron(sparse(I, Nwy_in, Nwy_in), A1D * Aw_wx_bc.B1D))
    Aw_wx_bc = (;
        Aw_wx_bc...,
        Bbc = kron(sparse(I, Nwz_in, Nwz_in), kron(sparse(I, Nwy_in, Nwy_in), A1D * Aw_wx_bc.Btemp))
    )

    ## Aw_wy: evaluate w at wy location
    diag1 = weight * ones(Nwy_t - 1)
    A1D = spdiagm(Nwy_t - 1, Nwy_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wy_bc = bc_general_stag(Nwy_t, Nwy_in, Nwy_b, bc.w.y[1], bc.w.y[2], hy[1], hy[end])

    # Extend to 3D
    Aw_wy = kron(sparse(I, Nwz_in, Nwz_in), kron(A1D * Aw_wy_bc.B1D, sparse(I, Nwx_in, Nwx_in)))
    Aw_wy_bc = (; Aw_wy_bc..., Bbc = kron(sparse(I, Nwz_in, Nwz_in), kron(A1D * Aw_wy_bc.Btemp, sparse(I, Nwx_in, Nwx_in))))

    ## Aw_wz: evaluate w at wz location
    diag1 = weight * ones(Nwz_t - 1)
    A1D = spdiagm(Nwz_t - 1, Nwz_t, 0 => diag1, 1 => diag1)

    # Boundary conditions
    Aw_wz_bc = bc_general_stag(Nwz_t, Nwz_in, Nwz_b, bc.w.z[1], bc.w.z[2], hz[1], hz[end])

    # Extend to 3D
    Aw_wz = kron(A1D * Aw_wz_bc.B1D, kron(sparse(I, Nwy_in, Nwy_in), sparse(I, Nwx_in, Nwx_in)))
    Aw_wz_bc = (; Aw_wz_bc..., Bbc = kron(A1D * Aw_wz_bc.Btemp, kron(sparse(I, Nwy_in, Nwy_in), sparse(I, Nwx_in, Nwx_in))))


    ## Store in setup structure
    @pack! setup.discretization = Au_ux, Au_uy, Au_uz
    @pack! setup.discretization = Av_vx, Av_vy, Av_vz
    @pack! setup.discretization = Aw_wx, Aw_wy, Aw_wz
    @pack! setup.discretization = Au_ux_bc, Au_uy_bc, Au_uz_bc
    @pack! setup.discretization = Av_vx_bc, Av_vy_bc, Av_vz_bc
    @pack! setup.discretization = Aw_wx_bc, Aw_wy_bc, Aw_wz_bc

    setup
end
