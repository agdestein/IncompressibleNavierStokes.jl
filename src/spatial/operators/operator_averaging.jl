"""
Construct averaging operators.
"""
function operator_averaging!(setup)
    # boundary conditions
    BC = setup.BC

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
    diag1 = weight * ones(Nux_t)
    A1D = spdiagm(Nux_t - 1, Nux_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Au_ux_BC = BC_general(Nux_t, Nux_in, Nux_b, BC.u.left, BC.u.right, hx[1], hx[end])

    # extend to 2D
    Au_ux = kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_BC.B1D)
    Au_ux_BC.Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D * Au_ux_BC.Btemp)


    ## Au_uy: evaluate u at uy location
    diag1 = weight * ones(Nuy_t)
    A1D = spdiagm(Nuy_t - 1, Nuy_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Au_uy_BC = BC_general_stag(Nuy_t, Nuy_in, Nuy_b, BC.u.low, BC.u.up, hy[1], hy[end])

    # extend to 2D
    Au_uy = kron(A1D * Au_uy_BC.B1D, speye(Nux_in))
    Au_uy_BC.Bbc = kron(A1D * Au_uy_BC.Btemp, speye(Nux_in))


    ## Averaging operators, v-component

    ## Av_vx: evaluate v at vx location
    diag1 = weight * ones(Nvx_t)
    A1D = spdiagm(Nvx_t - 1, Nvx_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Av_vx_BC = BC_general_stag(Nvx_t, Nvx_in, Nvx_b, BC.v.left, BC.v.right, hx[1], hx[end])

    # extend to 2D
    Av_vx = kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_BC.B1D)
    Av_vx_BC.Bbc = kron(sparse(I, Nvy_in, Nvy_in), A1D * Av_vx_BC.Btemp)


    ## Av_vy: evaluate v at vy location
    diag1 = weight * ones(Nvy_t)
    A1D = spdiagm(Nvy_t - 1, Nvy_t, 0 => diag1, 1 => diag1)

    # boundary conditions
    Av_vy_BC = BC_general(Nvy_t, Nvy_in, Nvy_b, BC.v.low, BC.v.up, hy[1], hy[end])

    # extend to 2D
    Av_vy = kron(A1D * Av_vy_BC.B1D, speye(Nvx_in))
    Av_vy_BC.Bbc = kron(A1D * Av_vy_BC.Btemp, speye(Nvx_in))


    ## fourth order
    if order4
        ## Au_ux: evaluate u at ux location
        diag1 = weight * ones(Nux_t + 4)
        A1D3 = spdiagm(Nux_in + 3, Nux_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Au_ux_BC3 = BC_av3(
            Nux_t + 4,
            Nux_in,
            Nux_t + 4 - Nux_in,
            BC.u.left,
            BC.u.right,
            hx[1],
            hx[end],
        )
        # extend to 2D
        Au_ux3 = kron(sparse(I, Nuy_in, Nuy_in), A1D3 * Au_ux_BC3.B1D)
        Au_ux_BC3.Bbc = kron(sparse(I, Nuy_in, Nuy_in), A1D3 * Au_ux_BC3.Btemp)

        ## Au_uy: evaluate u at uy location
        diag1 = weight * ones(Nuy_t + 4)
        A1D3 = spdiagm(Nuy_in + 3, Nuy_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Au_uy_BC3 = BC_av_stag3(
            Nuy_t + 4,
            Nuy_in,
            Nuy_t + 4 - Nuy_in,
            BC.u.low,
            BC.u.up,
            hy[1],
            hy[end],
        )
        # extend to 2D
        Au_uy3 = kron(A1D3 * Au_uy_BC3.B1D, sparse(I, Nux_in, Nux_in))
        Au_uy_BC3.Bbc = kron(A1D3 * Au_uy_BC3.Btemp, sparse(I, Nux_in, Nux_in))


        ## Av_vx: evaluate v at vx location
        diag1 = weight * ones(Nvx_t + 4)
        A1D3 = spdiagm(Nvx_in + 3, Nvx_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Av_vx_BC3 = BC_av_stag3(
            Nvx_t + 4,
            Nvx_in,
            Nvx_t + 4 - Nvx_in,
            BC.v.left,
            BC.v.right,
            hx[1],
            hx[end],
        )
        # extend to 2D
        Av_vx3 = kron(sparse(I, Nvy_in, Nvy_in), A1D3 * Av_vx_BC3.B1D)
        Av_vx_BC3.Bbc = kron(sparse(I, Nvy_in, Nvy_in), A1D3 * Av_vx_BC3.Btemp)


        ## Av_vy: evaluate v at vy location
        diag1 = weight * ones(Nvy_t + 4)
        A1D3 = spdiagm(Nvy_in + 3, Nvy_t + 4, 0 => diag1, 3 => diag1)

        # boundary conditions
        Av_vy_BC3 =
            BC_av3(Nvy_t + 4, Nvy_in, Nvy_t + 4 - Nvy_in, BC.v.low, BC.v.up, hy[1], hy[end])
        # extend to 2D
        Av_vy3 = kron(A1D3 * Av_vy_BC3.B1D, sparse(Nvx_in))
        Av_vy_BC3.Bbc = kron(A1D3 * Av_vy_BC3.Btemp, sparse(Nvx_in))


    end

    ## store in setup structure
    setup.discretization.Au_ux = Au_ux
    setup.discretization.Au_uy = Au_uy
    setup.discretization.Av_vx = Av_vx
    setup.discretization.Av_vy = Av_vy
    setup.discretization.Au_ux_BC = Au_ux_BC
    setup.discretization.Au_uy_BC = Au_uy_BC
    setup.discretization.Av_vx_BC = Av_vx_BC
    setup.discretization.Av_vy_BC = Av_vy_BC

    if order4
        setup.discretization.Au_ux3 = Au_ux3
        setup.discretization.Au_uy3 = Au_uy3
        setup.discretization.Av_vx3 = Av_vx3
        setup.discretization.Av_vy3 = Av_vy3
        setup.discretization.Au_ux_BC3 = Au_ux_BC3
        setup.discretization.Au_uy_BC3 = Au_uy_BC3
        setup.discretization.Av_vx_BC3 = Av_vx_BC3
        setup.discretization.Av_vy_BC3 = Av_vy_BC3
    end
end
