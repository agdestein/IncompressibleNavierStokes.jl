"""
    get_vorticity(V, t, setup)

Get vorticity from velocity field.
"""
function get_vorticity(V, t, setup)
    (; bc) = setup
    (; Nx, Ny) = setup.grid
    (; Wv_vx, Wu_uy) = setup.discretization
    Wv_vx, Wu_uy

    Nωx = bc.u.x[1] == :periodic ? Nx + 1 : Nx - 1
    Nωy = bc.v.y[1] == :periodic ? Ny + 1 : Ny - 1
    ω = zeros(Nωx, Nωy)

    vorticity!(ω, V, t, setup)
end

"""
    vorticity!(ω, V, t, setup)

Compute vorticity values at pressure midpoints.
This should be consistent with `operator_postprocessing.jl`.
"""
function vorticity!(ω, V, t, setup)
    (; indu, indv, Nux_in, Nvy_in, Nx, Ny) = setup.grid
    (; Wv_vx, Wu_uy) = setup.discretization

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    ω_flat = reshape(ω, length(ω))

    if setup.bc.u.x[1] == :periodic && setup.bc.v.y[1] == :periodic
        uₕ_in = uₕ
        vₕ_in = vₕ
    else
        # Velocity at inner points
        diagpos = 0
        if setup.bc.u.x[1] == :pressure
            diagpos = 1
        end
        if setup.bc.u.x[2] == :periodic && setup.bc.u.x[1] == :periodic
            # Like pressure left
            diagpos = 1
        end

        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = kron(sparse(I, Ny, Ny), B1D)

        uₕ_in = B2D * uₕ

        diagpos = 0
        if setup.bc.v.y[1] == :pressure
            diagpos = 1
        end
        if setup.bc.v.y[1] == :periodic && setup.bc.v.y[2] == :periodic
            # Like pressure low
            diagpos = 1
        end

        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = kron(B1D, sparse(I, Nx, Nx))

        vₕ_in = B2D * vₕ
    end

    # ω_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in
    mul!(ω_flat, Wv_vx, vₕ_in) # a = b * c
    mul!(ω_flat, Wu_uy, uₕ_in, -1, 1) # a = -b * c + a

    ω
end
