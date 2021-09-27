function get_vorticity(V, t, setup)
    # vorticity values at pressure midpoints
    # this should be consistent with operator_postprocessing
    @unpack Nu, Nv, Nux_in, Nvy_in, Nx, Ny = setup.grid
    @unpack Wv_vx, Wu_uy = setup.discretization

    uh = V[1:Nu]
    vh = V[Nu+1:Nu+Nv]

    if setup.bc.u.left == "per" && setup.bc.v.low == "per"
        uh_in = uh
        vh_in = vh
    else
        # velocity at inner points
        diagpos = 0
        if setup.bc.u.left == "pres"
            diagpos = 1
        end
        if setup.bc.u.right == "per" && setup.bc.u.left == "per"
            # like pressure left
            diagpos = 1
        end

        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = kron(sparse(I, Ny, Ny), B1D)

        uh_in = B2D * uh

        diagpos = 0
        if setup.bc.v.low == "pres"
            diagpos = 1
        end
        if setup.bc.v.low == "per" && setup.bc.v.up == "per"
            # like pressure low
            diagpos = 1
        end

        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = kron(B1D, sparse(I, Nx, Nx))

        vh_in = B2D * vh
    end

    Wv_vx * vh_in - Wu_uy * uh_in
end
