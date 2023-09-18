"""
    plot_vorticity(setup, V, t; kwargs...)

Plot vorticity field.
"""
function plot_vorticity end

plot_vorticity(setup, u; kwargs...) =
    plot_vorticity(setup.grid.dimension, setup, u; kwargs...)

# 2D version
function plot_vorticity(::Dimension{2}, setup, u; kwargs...)
    (; grid, boundary_conditions) = setup
    (; xp, xlims) = grid
    T = eltype(xp[1])

    # Get fields
    ω = vorticity(u, setup)
    ωp = interpolate_ω_p(setup, ω)

    # Levels
    μ, σ = mean(ω), std(ω)
    # ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - T(1.5) * σ, μ + T(1.5) * σ, 10)

    # Plot vorticity
    fig = Figure()
    ax = Makie.Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Vorticity",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1]..., xlims[2]...)
    cf = contourf!(ax, xp..., ω; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    # cf = heatmap!(ax, xp..., ωp; kwargs...)
    Colorbar(fig[1, 2], cf)

    # save("output/vorticity.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_vorticity(::Dimension{3}, setup, u; kwargs...)
    (; grid, boundary_conditions) = setup
    (; xp) = grid

    ωp = interpolate_ω_p(setup, vorticity(u, setup))
    qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), ωp...)

    # Levels
    μ, σ = mean(qp), std(qp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)


    xp = Array.(xp)
    qp = Array(qp)
    contour(xp..., qp; levels, kwargs...)
end
