"""
    convection(model, V, ϕ, setup; bc_vectors, get_jacobian = false)

Evaluate convective terms `c` and, optionally, Jacobian `∇c = ∂c/∂V`, using the convection
model `model`. The convected quantity is `ϕ` (usually `ϕ = V`).

Non-mutating/allocating/out-of-place version.

See also [`convection!`](@ref).
"""
function convection end

convection(
    m,
    V,
    ϕ,
    setup;
    kwargs...,
) = convection(
    setup.grid.dimension,
    m,
    V,
    ϕ,
    setup;
    kwargs...,
)

function convection(
    dimension,
    ::NoRegConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; order4, α) = setup.grid

    # No regularization
    c, ∇c = convection_components(
        V,
        ϕ,
        setup;
        bc_vectors,
        get_jacobian,
        newton_factor,
        order4 = false,
    )

    if order4
        c3, ∇c3 = convection_components(
            V,
            ϕ,
            setup;
            bc_vectors,
            get_jacobian,
            newton_factor,
            order4,
        )
        c = @. α * c - c3
        get_jacobian && (∇c = @. α * ∇c - ∇c3)
    end

    c, ∇c
end

# 2D version
function convection(
    ::Dimension{2},
    ::C2ConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    V̄ = [ūₕ; v̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    c, ∇c = convection_components(V̄, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)

    cu = @view c[indu]
    cv = @view c[indv]

    cu = filter_convection(cu, Diffu_f, yDiffu_f, α_reg)
    cv = filter_convection(cv, Diffv_f, yDiffv_f, α_reg)

    c = [cu; cv]

    c, ∇c
end

# 3D version
function convection(
    ::Dimension{3},
    ::C2ConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α_reg)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    V̄ = [ūₕ; v̄ₕ; w̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    c, ∇c =
        convection_components(V̄, ϕ̄, setup, cache; bc_vectors, get_jacobian, newton_factor)

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    cu = filter_convection(cu, Diffu_f, yDiffu_f, α_reg)
    cv = filter_convection(cv, Diffv_f, yDiffv_f, α_reg)
    cw = filter_convection(cw, Diffw_f, yDiffw_f, α_reg)

    c = [cu; cv; cw]

    c, ∇c
end

# 2D version
function convection(
    ::Dimension{2},
    ::C4ConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # C4 consists of 3 terms:
    # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
    #      filter(conv(u', filter(u)))
    # where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)

    V̄ = [ūₕ; v̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    c, ∇c = convection_components(V̄, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
    c2, ∇c2 = convection_components(ΔV, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
    c3, ∇c3 = convection_components(V̄, Δϕ, setup; bc_vectors, get_jacobian, newton_factor)

    cu = @view c[indu]
    cv = @view c[indv]

    cu2 = @view c2[indu]
    cv2 = @view c2[indv]

    cu3 = @view c3[indu]
    cv3 = @view c3[indv]

    cu += filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α_reg)
    cv += filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α_reg)

    c = [cu; cv]

    c, ∇c
end

# 3D version
function convection(
    ::Dimension{3},
    ::C4ConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors
    (; c2, ∇c2, c3, ∇c3) = cache

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    # C4 consists of 3 terms:
    # C4 = conv(filter(u), filter(u)) + filter(conv(filter(u), u') +
    #      filter(conv(u', filter(u)))
    # where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α_reg)

    V̄ = [ūₕ; v̄ₕ; w̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    c, ∇c = convection_components(V̄, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
    c2, ∇c2 = convection_components(ΔV, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
    c3, ∇c3 = convection_components(V̄, Δϕ, setup; bc_vectors, get_jacobian, newton_factor)

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    cu2 = @view c2[indu]
    cv2 = @view c2[indv]
    cw2 = @view c2[indw]

    cu3 = @view c3[indu]
    cv3 = @view c3[indv]
    cw3 = @view c3[indw]

    cu += filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α_reg)
    cv += filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α_reg)
    cw += filter_convection(cw2 + cw3, Diffw_f, yDiffw_f, α_reg)

    c = [cu; cv; cw]

    c, ∇c
end

# 2D version
function convection(
    ::Dimension{2},
    ::LerayConvectionModel,
    V,
    ϕ,
    setup;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # TODO: needs finishing
    error("Leray convection not implemented yet")

    # Filter the convecting field
    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components(V, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
end

# 3D version
function convection(
    ::Dimension{3},
    ::LerayConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    bc_vectors,
    cache;
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    # TODO: needs finishing
    error("Leray convection not implemented yet")

    # Filter the convecting field
    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components(V, ϕ̄, setup; bc_vectors, get_jacobian, newton_factor)
end

"""
    convection!(model, c, ∇c, V, ϕ, setup, cache; bc_vectors, get_jacobian = false)

Evaluate convective terms `c` and, optionally, Jacobian `∇c = ∂c/∂V`, using the convection
model `model`. The convected quantity is `ϕ` (usually `ϕ = V`).

Mutating/non-allocating/in-place version.

See also [`convection`](@ref).
"""
function convection! end

convection!(
    dimension,
    m,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    kwargs...,
) = convection!(
    setup.grid.dimension,
    m,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    kwargs...,
)

function convection!(
    dimension,
    ::NoRegConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; order4, α) = setup.grid
    (; c3, ∇c3) = cache

    # No regularization
    convection_components!(
        c,
        ∇c,
        V,
        ϕ,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
        order4 = false,
    )

    if order4
        convection_components!(
            c3,
            ∇c3,
            V,
            ϕ,
            setup,
            cache;
            bc_vectors,
            get_jacobian,
            newton_factor,
            order4,
        )
        @. c = α * c - c3
        get_jacobian && (@. ∇c = α * ∇c - ∇c3)
    end

    c, ∇c
end

# 2D version
function convection!(
    ::Dimension{2},
    ::C2ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors

    cu = @view c[indu]
    cv = @view c[indv]

    uₕ = @view V[indu]
    vₕ = @view V[indv]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    V̄ = [ūₕ; v̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(
        c,
        ∇c,
        V̄,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    cu .= filter_convection(cu, Diffu_f, yDiffu_f, α_reg)
    cv .= filter_convection(cv, Diffv_f, yDiffv_f, α_reg)

    c, ∇c
end

# 3D version
function convection!(
    ::Dimension{3},
    ::C2ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors

    cu = @view c[indu]
    cv = @view c[indv]
    cw = @view c[indw]

    uₕ = @view V[indu]
    vₕ = @view V[indv]
    wₕ = @view V[indw]

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]
    ϕw = @view ϕ[indw]

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α_reg)

    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    V̄ = [ūₕ; v̄ₕ; w̄ₕ]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(
        c,
        ∇c,
        V̄,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    cu .= filter_convection(cu, Diffu_f, yDiffu_f, α_reg)
    cv .= filter_convection(cv, Diffv_f, yDiffv_f, α_reg)
    cw .= filter_convection(cw, Diffw_f, yDiffw_f, α_reg)

    c, ∇c
end

# 2D version
function convection!(
    ::Dimension{2},
    ::C4ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors
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
    # where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)

    V̄ = [ūₕ; v̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    convection_components!(
        c,
        ∇c,
        V̄,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )
    convection_components!(
        c2,
        ∇c2,
        ΔV,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )
    convection_components!(
        c3,
        ∇c3,
        V̄,
        Δϕ,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α_reg)
    cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α_reg)

    c, ∇c
end

# 3D version
function convection!(
    ::Dimension{3},
    ::C4ConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors
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
    # where u' = u - filter(u)

    # Filter both convecting and convected velocity
    ūₕ = filter_convection(uₕ, Diffu_f, yDiffu_f, α_reg)
    v̄ₕ = filter_convection(vₕ, Diffv_f, yDiffv_f, α_reg)
    w̄ₕ = filter_convection(wₕ, Diffw_f, yDiffw_f, α_reg)

    V̄ = [ūₕ; v̄ₕ; w̄ₕ]
    ΔV = V - V̄

    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)
    ϕ̄w = filter_convection(ϕw, Diffw_f, yDiffw_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v; ϕ̄w]
    Δϕ = ϕ - ϕ̄

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * V̄ + yM))

    convection_components!(
        c,
        ∇c,
        V̄,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )
    convection_components!(
        c2,
        ∇c2,
        ΔV,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )
    convection_components!(
        c3,
        ∇c3,
        V̄,
        Δϕ,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    cu .+= filter_convection(cu2 + cu3, Diffu_f, yDiffu_f, α_reg)
    cv .+= filter_convection(cv2 + cv3, Diffv_f, yDiffv_f, α_reg)
    cw .+= filter_convection(cw2 + cw3, Diffw_f, yDiffw_f, α_reg)

    c, ∇c
end

# 2D version
function convection!(
    ::Dimension{2},
    ::LerayConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv) = grid
    (; Diffu_f, Diffv_f, M, α_reg) = operators
    (; yDiffu_f, yDiffv_f, yM) = bc_vectors

    ϕu = @view ϕ[indu]
    ϕv = @view ϕ[indv]

    # TODO: needs finishing
    error("Leray convection not implemented yet")

    # Filter the convecting field
    ϕ̄u = filter_convection(ϕu, Diffu_f, yDiffu_f, α_reg)
    ϕ̄v = filter_convection(ϕv, Diffv_f, yDiffv_f, α_reg)

    ϕ̄ = [ϕ̄u; ϕ̄v]

    # Divergence of filtered velocity field; should be zero!
    maxdiv_f = maximum(abs.(M * ϕ̄ + yM))

    convection_components!(
        c,
        ∇c,
        V,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    c, ∇c
end

# 3D version
function convection!(
    ::Dimension{3},
    ::LerayConvectionModel,
    c,
    ∇c,
    V,
    ϕ,
    setup,
    cache;
    bc_vectors,
    get_jacobian = false,
    newton_factor = false,
)
    (; grid, operators) = setup
    (; indu, indv, indw) = grid
    (; Diffu_f, Diffv_f, Diffw_f, M, α) = operators
    (; yDiffu_f, yDiffv_f, yDiffw_f, yM) = bc_vectors

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

    convection_components!(
        c,
        ∇c,
        V,
        ϕ̄,
        setup,
        cache;
        bc_vectors,
        get_jacobian,
        newton_factor,
    )

    c, ∇c
end
