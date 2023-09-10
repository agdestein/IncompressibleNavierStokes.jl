"""
    get_timestep(stepper, cfl; bc_vectors)

Estimate time step based on eigenvalues of operators, using Gershgorin.
"""
function get_timestep end

get_timestep(stepper, cfl) = get_timestep(stepper.setup.grid.dimension, stepper, cfl)

# 2D version
function get_timestep(::Dimension{2}, stepper, cfl)
    (; setup, method, bc_vectors, V) = stepper
    (; grid, operators) = setup
    (; NV, indu, indv, Ω) = grid
    (; Diff) = operators
    (; Cux, Cuy, Cvx, Cvy) = operators
    (; Au_ux, Au_uy, Av_vx, Av_vy) = operators
    (; Iu_ux, Iu_vx, Iv_uy, Iv_vy) = operators
    (; yIu_ux, yIu_vx, yIv_uy, yIv_vy) = bc_vectors

    T = eltype(V)

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    # For explicit methods only
    @assert isexplicit(method) "Adaptive timestep requires explicit method"

    # Convective part
    Cu =
        Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
        Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy
    Cv =
        Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
        Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy
    C = blockdiag(Cu, Cv)
    test = spdiagm(1 ./ Ω) * C
    sum_conv = abs.(test) * ones(T, NV) - diag(abs.(test)) - diag(test)
    λ_conv = maximum(sum_conv)

    # Based on max. value of stability region (not a very good indication
    # For the methods that do not include the imaginary axis)
    Δt_conv = lambda_conv_max(method) / λ_conv

    ## Diffusive part
    test = Diagonal(1 ./ Ω) * Diff
    sum_diff = abs.(test) * ones(T, NV) - diag(abs.(test)) - diag(test)
    λ_diff = maximum(sum_diff)

    # Based on max. value of stability region
    Δt_diff = lambda_diff_max(method) / λ_diff

    Δt = cfl * min(Δt_conv, Δt_diff)

    Δt
end

# 3D version
function get_timestep(::Dimension{3}, stepper, cfl)
    (; setup, method, bc_vectors, V) = stepper
    (; grid, operators) = setup
    (; NV, indu, indv, indw, Ω) = grid
    (; Diff) = operators
    (; Cux, Cuy, Cuz, Cvx, Cvy, Cvz, Cwx, Cwy, Cwz) = operators
    (; Au_ux, Au_uy, Au_uz, Av_vx, Av_vy, Av_vz, Aw_wx, Aw_wy, Aw_wz) = operators
    (; Iu_ux, Iu_vx, Iu_wx, Iv_uy, Iv_vy, Iv_wy, Iw_uz, Iw_vz, Iw_wz) = operators
    (; yIu_ux, yIu_vx, yIu_wx) = bc_vectors
    (; yIv_uy, yIv_vy, yIv_wy) = bc_vectors
    (; yIw_uz, yIw_vz, yIw_wz) = bc_vectors

    T = eltype(V)

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    # For explicit methods only
    @assert isexplicit(method) "Adaptive timestep requires explicit method"

    # Convective part
    Cu =
        Cux * spdiagm(Iu_ux * uₕ + yIu_ux) * Au_ux +
        Cuy * spdiagm(Iv_uy * vₕ + yIv_uy) * Au_uy +
        Cuz * spdiagm(Iw_uz * wₕ + yIw_uz) * Au_uz
    Cv =
        Cvx * spdiagm(Iu_vx * uₕ + yIu_vx) * Av_vx +
        Cvy * spdiagm(Iv_vy * vₕ + yIv_vy) * Av_vy +
        Cvz * spdiagm(Iw_vz * wₕ + yIw_vz) * Av_vz
    Cw =
        Cwx * spdiagm(Iu_wx * uₕ + yIu_wx) * Aw_wx +
        Cwy * spdiagm(Iv_wy * vₕ + yIv_wy) * Aw_wy +
        Cwz * spdiagm(Iw_wz * wₕ + yIw_wz) * Aw_wz
    C = blockdiag(Cu, Cv, Cw)
    test = spdiagm(1 ./ Ω) * C
    sum_conv = abs.(test) * ones(T, NV) - diag(abs.(test)) - diag(test)
    λ_conv = maximum(sum_conv)

    # Based on max. value of stability region (not a very good indication
    # For the methods that do not include the imaginary axis)
    Δt_conv = lambda_conv_max(method) / λ_conv

    ## Diffusive part
    test = Diagonal(1 ./ Ω) * Diff
    sum_diff = abs.(test) * ones(T, NV) - diag(abs.(test)) - diag(test)
    λ_diff = maximum(sum_diff)

    # Based on max. value of stability region
    Δt_diff = lambda_diff_max(method) / λ_diff

    Δt = cfl * min(Δt_conv, Δt_diff)

    Δt
end
