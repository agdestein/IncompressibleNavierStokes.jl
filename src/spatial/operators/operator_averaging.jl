"""
Construct averaging operators.
"""
function operator_averaging!(setup)
    # boundary conditions
    bc = setup.bc

    # number of interior points and boundary points
    Nux_in = setup.grid.Nux_in
    Nux_b = setup.grid.Nux_b
    Nux_t = setup.grid.Nux_t
    Nuy_in = setup.grid.Nuy_in
    Nuy_b = setup.grid.Nuy_b
    Nuy_t = setup.grid.Nuy_t
    Nvx_in = setup.grid.Nvx_in
    Nvx_b = setup.grid.Nvx_b
    Nvx_t = setup.grid.Nvx_t
    Nvy_in = setup.grid.Nvy_in
    Nvy_b = setup.grid.Nvy_b
    Nvy_t = setup.grid.Nvy_t

    hx = setup.grid.hx
    hy = setup.grid.hy

    order4 = setup.discretization.order4

    # averaging weight:
    weight = 1 / 2

    ## Averaging operators, u-component

    ## Au_ux: evaluate u at ux location
    diag1 = weight * ones(Nux_t - 1)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Au_ux_bc = bc_general(Nux_t, Nux_in, Nux_b, bc.u.left, bc.u.right, hx[1], hx[end])

    # extend to 2D
    Au_ux = kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_bc.B1D)
    Au_ux_bc = (; Au_ux_bc..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_bc.Btemp))


    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t - 1)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Au_uy_bc = bc_general_stag(Nuy_t, Nuy_in, Nuy_b, bc.u.low, bc.u.up, hy[1], hy[end])

    # extend to 2D
    Au_uy = kron(A1D * Au_uy_bc.B1D, sparse(I, Nux_in, Nux_in))
    Au_uy_bc = (; Au_uy_bc..., Bbc = kron(A1D * Au_uy_bc.Btemp, sparse(I, Nux_in, Nux_in)))

    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t - 1)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Av_vx_bc = bc_general_stag(Nvx_t, Nvx_in, Nvx_b, bc.v.left, bc.v.right, hx[1], hx[end])

    # extend to 2D
    Av_vx = kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_bc.B1D)
    Av_vx_bc = (; Av_vx_bc..., Bbc = kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_bc.Btemp))


    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t - 1)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Av_vy_bc = bc_general(Nvy_t, Nvy_in, Nvy_b, bc.v.low, bc.v.up, hy[1], hy[end])

    # extend to 2D
    Av_vy = kron(A1D * Av_vy_bc.B1D, sparse(I, Nvx_in, Nvx_in))
    Av_vy_bc = (; Av_vy_bc..., Bbc = kron(A1D * Av_vy_bc.Btemp, sparse(I, Nvx_in, Nvx_in)))

    ## fourth order
    if order4
        ## Au_ux: evaluate u at ux location
        diag1 = weight * ones(Nux_t + 4)
        A1D3 = spdiagm(Nux_in + 3, Nux_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Au_ux_bc3 = bc_av3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            bc.u.left,
            bc.u.right,
            hx[1],
            hx[end],
        )
        # extend to 2D
        Au_ux3 = kron(sparse(I, Nuy_in, Nuy_in), A1D3 * Au_ux_bc3.B1D)
        Au_ux_bc3 =
            (; Au_ux_bc3..., Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D3 * Au_ux_bc3.Btemp))

        ## Au_uy: evaluate u at uy location
        diag1 = weight * ones(Nuy_t + 4)
        A1D3 = spdiagm(Nuy_in + 3, Nuy_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Au_uy_bc3 = bc_av_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            bc.u.low,
            bc.u.up,
            hy[1],
            hy[end],
        )
        # extend to 2D
        Au_uy3 = kron(A1D3 * Au_uy_bc3.B1D, sparse(I, Nux_in, Nux_in))
        Au_uy_bc3 =
            (; Au_uy_bc3..., Bbc = kron(A1D3 * Au_uy_bc3.Btemp, sparse(I, Nux_in, Nux_in)))

        ## Av_vx: evaluate v at vx location
        diag1 = weight * ones(Nvx_t + 4)
        A1D3 = spdiagm(Nvx_in + 3, Nvx_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Av_vx_bc3 = bc_av_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            bc.v.left,
            bc.v.right,
            hx[1],
            hx[end],
        )
        # extend to 2D
        Av_vx3 = kron(sparse(I, Nvy_in, Nvy_in), A1D3 * Av_vx_bc3.B1D)
        Av_vx_bc3 =
            (; Av_vx_bc3..., Bbc = kron(sparse(I, Nvy_in, Nvy_in), A1D3 * Av_vx_bc3.Btemp))

        ## Av_vy: evaluate v at vy location
        diag1 = weight * ones(Nvy_t + 4)
        A1D3 = spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Av_vy_bc3 =
            bc_av3(Nvy_t + 4, Nvy_in, Nvy_t + 4 - Nvy_in, bc.v.low, bc.v.up, hy[1], hy[end])
        # extend to 2D
        Av_vy3 = kron(A1D3 * Av_vy_bc3.B1D, sparse(I, Nvx_in, Nvx_in))
        Av_vy_bc3 =
            (; Av_vy_bc3..., Bbc = kron(A1D3 * Av_vy_bc3.Btemp, sparse(I, Nvx_in, Nvx_in)))
    end

    ## store in setup structure
    setup.discretization.Au_ux = Au_ux
    setup.discretization.Au_uy = Au_uy
    setup.discretization.Av_vx = Av_vx
    setup.discretization.Av_vy = Av_vy
    setup.discretization.Au_ux_bc = Au_ux_bc
    setup.discretization.Au_uy_bc = Au_uy_bc
    setup.discretization.Av_vx_bc = Av_vx_bc
    setup.discretization.Av_vy_bc = Av_vy_bc

    if order4
        setup.discretization.Au_ux3 = Au_ux3
        setup.discretization.Au_uy3 = Au_uy3
        setup.discretization.Av_vx3 = Av_vx3
        setup.discretization.Av_vy3 = Av_vy3
        setup.discretization.Au_ux_bc3 = Au_ux_bc3
        setup.discretization.Au_uy_bc3 = Au_uy_bc3
        setup.discretization.Av_vx_bc3 = Av_vx_bc3
        setup.discretization.Av_vy_bc3 = Av_vy_bc3
    end
end
