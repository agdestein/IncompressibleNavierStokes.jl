"""
    convection(model, V, ϕ, t, setup; getJacobian = false)

Convenience function for initializing arrays `c` and `∇c` before filling in convection terms.
"""
function convection(model, V, ϕ, t, setup; getJacobian = false)
    (; NV) = setup.grid

    cache = MomentumCache(setup)
    c = zeros(NV)
    ∇c = spzeros(NV, NV)

    convection!(model, c, ∇c, V, ϕ, t, setup, cache; getJacobian)
end

"""
    convection!(model, c, ∇c, V, ϕ, t, setup, cache; getJacobian = false)

Evaluate convective terms `c` and, optionally, Jacobian `∇c = ∂c/∂V`, using the convection
model `model`. The convected quantity is `ϕ` (usually `ϕ = V`).
"""
function convection! end

function convection!(
    ::NoRegConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup,
    cache;
    getJacobian = false,
)
    (; order4, α) = setup.grid
    (; c3, ∇c3) = cache

    # No regularization
    convection_components!(c, ∇c, V, ϕ, setup, cache; getJacobian, order4 = false)

    if order4
        convection_components!(c3, ∇c3, V, ϕ, setup, cache; getJacobian, order4)
        @. c = α * c - c3
        getJacobian && (@. ∇c = α * ∇c - ∇c3)
    end

    c, ∇c
end

# 2D version
function convection!(
    ::C2ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,2},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, M, yM, α) = operators

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    V̄ = [ūₕ; v̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)

    cu .= filter_convection(cu, Diffu_f, yDiffu_f, α)
    cv .= filter_convection(cv, Diffv_f, yDiffv_f, α)

    c, ∇c
end

# 3D version
function convection!(
    ::C2ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,3},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, Diffw_f, yDiffw_f, M, yM, α) = operators

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    V̄ = [ūₕ; v̄ₕ; w̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)

    cu .= filter_convection(cu, Diffu_f, yDiffu_f, α)
    cv .= filter_convection(cv, Diffv_f, yDiffv_f, α)
    cw .= filter_convection(cw, Diffw_f, yDiffw_f, α)

    c, ∇c
end

# 2D version
function convection!(
    ::C4ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,2},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, M, yM, α) = operators
    (; c2, ∇c2, c3, ∇c3) = cache

    cu = @view c[indu]
    cv = @view c[indv]
    cu2 = @view c2[indu]
    cv2 = @view c2[indv]
    cu3 = @view c3[indu]
    cv3 = @view c3[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # C4 consists of 3 terms:
    # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
    #      filter(conv(u', filter(u)))
    # Where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)

    V̄ = [ūₕ; v̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)
    convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache; getJacobian)
    convection_components!(c3, ∇c3, V̄, Δϕ, setup, cache; getJacobian)

    # TODO: consider inner loop parallelization
    # @sync begin
    #     @spawn convection_components!(c, ∇c, V̄, ϕ̄, setup, cache, getJacobian)
    #     @spawn convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache, getJacobian)
    # end

    cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α)
    cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α)

    c, ∇c
end

# 3D version
function convection!(
    ::C4ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,3},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, Diffw_f, yDiffw_f, M, yM, α) = operators
    (; c2, ∇c2, c3, ∇c3) = cache

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    cu2 = @view c2[indu]
    cv2 = @view c2[indv]
    cw2 = @view c2[indw]

    cu3 = @view c3[indu]
    cv3 = @view c3[indv]
    cw3 = @view c3[indw]

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    # C4 consists of 3 terms:
    # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
    #      filter(conv(u', filter(u)))
    # Where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α)

    V̄ = [ūₕ; v̄ₕ; w̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    convection_components!(c, ∇c, V̄, ϕ̄, setup, cache; getJacobian)
    convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache; getJacobian)
    convection_components!(c3, ∇c3, V̄, Δϕ, setup, cache; getJacobian)

    # TODO: consider inner loop parallelization
    # @sync begin
    #     @spawn convection_components!(c, ∇c, V̄, ϕ̄, setup, cache, getJacobian)
    #     @spawn convection_components!(c2, ∇c2, ΔV, ϕ̄, setup, cache, getJacobian)
    #     @spawn convection_components!(c3, ∇c3, V̄, Δϕ, setup, cache, getJacobian)
    # end

    cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α)
    cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α)
    cw .+= filter_convection(cw2 + cw3, Diffw_f, yDiffw_f, α)

    c, ∇c
end

# 2D version
function convection!(
    ::LerayConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,2},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, M, yM, α) = operators

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # TODO: needs finishing
    error("Leray convection not implemented yet")

    # Filter the convecting field
    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(c, ∇c, V, ϕ̄, setup, cache; getJacobian)

    c, ∇c
end

# 3D version
function convection!(
    ::LerayConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    t,
    setup::Setup{T,3},
    cache;
    getJacobian = false,
) where {T}
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, yDiffu_f, Diffv_f, yDiffv_f, Diffw_f, yDiffw_f, M, yM, α) = operators

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    # TODO: needs finishing
    error("Leray convection not implemented yet")

    # Filter the convecting field
    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(c, ∇c, V, ϕ̄, setup, cache; getJacobian)

    c, ∇c
end
