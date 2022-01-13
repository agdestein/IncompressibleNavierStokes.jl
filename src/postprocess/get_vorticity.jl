"""
    get_vorticity(V, t, setup)

Get vorticity from velocity field.
"""
function get_vorticity(V, t, setup::Setup{T,N}) where {T,N}
    (; bc) = setup
    (; Nx, Ny, Nz) = setup.grid

    Nωx = bc.u.x[1] == :periodic ? Nx + 1 : Nx - 1
    Nωy = bc.v.y[1] == :periodic ? Ny + 1 : Ny - 1
    N == 3 && (Nωz = bc.w.z[1] == :periodic ? Nz + 1 : Nz - 1)
    if N == 2
        ω = zeros(Nωx, Nωy)
    else
        ω = zeros(3, Nωx, Nωy, Nωz)
    end

    vorticity!(ω, V, t, setup)
end

"""
    vorticity!(ω, V, t, setup)

Compute vorticity values at pressure midpoints.
This should be consistent with `operator_postprocessing.jl`.
"""
function vorticity! end

function vorticity!(ω, V, t, setup::Setup{T,2}) where {T}
    (; indu, indv, Nux_in, Nvy_in, Nx, Ny) = setup.grid
    (; Wv_vx, Wu_uy) = setup.operators

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

function vorticity!(ω, V, t, setup::Setup{T,3}) where {T}
    (; grid, operators, bc) = setup
    (; indu, indv, indw, Nux_in, Nvy_in, Nwz_in, Nx, Ny, Nz) = grid
    (; Wu_uy, Wu_uz, Wv_vx, Wv_vz, Ww_wx, Ww_wy) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]
    ω_flat = reshape(ω, length(ω))

    if bc.u.x[1] == :periodic && bc.v.y[1] == :periodic && bc.w.z[1] == :periodic
        uₕ_in = uₕ
        vₕ_in = vₕ
        wₕ_in = wₕ
    else
        # Velocity at inner points
        diagpos = 0
        bc.u.x[1] == :pressure && (diagpos = 1)
        bc.u.x == (:periodic, :periodic) && (diagpos = 1)

        B1D = spdiagm(Nx - 1, Nux_in, diagpos => ones(Nx - 1))
        B2D = kron(sparse(I, Ny, Ny), B1D)

        uₕ_in = B2D * uₕ

        diagpos = 0
        bc.v.y[1] == :pressure && (diagpos = 1)
        bc.v.y == (:periodic, :periodic) && (diagpos = 1)

        B1D = spdiagm(Ny - 1, Nvy_in, diagpos => ones(Ny - 1))
        B2D = kron(B1D, I(Nx))

        vₕ_in = B2D * vₕ
    end

    # ωz_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in
    # ωy_flat .= Wu_uz * uₕ_in - Ww_wx * wₕ_in
    # ωx_flat .= Ww_wx * wₕ_in - Wy_yz * vₕ_in
   
    ω_flat .= .√(
        (Wv_vx * vₕ_in - Wu_uy * uₕ_in) .^ 2 +
        (Wu_uz * uₕ_in - Ww_wx * wₕ_in) .^ 2 +
        (Ww_wx * wₕ_in - Wv_vz * vₕ_in) .^ 2
    )

    ω
end
