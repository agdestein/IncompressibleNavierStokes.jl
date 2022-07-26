"""
    get_vorticity(V, t, setup)

Get vorticity from velocity field.
"""
function get_vorticity end

# 2D version
function get_vorticity(V, t, setup::Setup{T,2}) where {T}
    (; Nx, Ny) = setup.grid

    Nωx = Nx + 1
    Nωy = Ny + 1

    ω = zeros(Nωx, Nωy)

    vorticity!(ω, V, t, setup)
end

# 3D version
function get_vorticity(V, t, setup::Setup{T,3}) where {T}
    (; Nx, Ny, Nz) = setup.grid

    Nωx = Nx + 1
    Nωy = Ny + 1
    Nωz = Nz + 1

    ω = zeros(Nωx, Nωy, Nωz)

    vorticity!(ω, V, t, setup)
end


"""
    vorticity!(ω, V, t, setup)

Compute vorticity values at pressure midpoints.
This should be consistent with `operator_postprocessing.jl`.
"""
function vorticity! end

# 2D version
function vorticity!(ω, V, t, setup::Setup{T,2}) where {T}
    (; indu, indv, Nux_in, Nvy_in, Nx, Ny) = setup.grid
    (; Wv_vx, Wu_uy) = setup.operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    ω_flat = reshape(ω, length(ω))

    uₕ_in = uₕ
    vₕ_in = vₕ

    # ω_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in
    mul!(ω_flat, Wv_vx, vₕ_in) # a = b * c
    mul!(ω_flat, Wu_uy, uₕ_in, -1, 1) # a = -b * c + a

    ω
end

# 3D version
function vorticity!(ω, V, t, setup::Setup{T,3}) where {T}
    (; grid, operators) = setup
    (; indu, indv, indw, Nux_in, Nvy_in, Nwz_in, Nx, Ny, Nz) = grid
    (; Wu_uy, Wu_uz, Wv_vx, Wv_vz, Ww_wx, Ww_wy) = operators

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]
    ω_flat = reshape(ω, length(ω))

    uₕ_in = uₕ
    vₕ_in = vₕ
    wₕ_in = wₕ

    # ωx_flat .= Ww_wy * wₕ_in - Wy_yz * vₕ_in
    # ωy_flat .= Wu_uz * uₕ_in - Ww_wx * wₕ_in
    # ωz_flat .= Wv_vx * vₕ_in - Wu_uy * uₕ_in

    ω_flat .=
        sqrt.(
            (Ww_wy * wₕ_in .- Wv_vz * vₕ_in) .^ 2 .+
            (Wu_uz * uₕ_in .- Ww_wx * wₕ_in) .^ 2 .+ (Wv_vx * vₕ_in .- Wu_uy * uₕ_in) .^ 2
        )

    ω
end
