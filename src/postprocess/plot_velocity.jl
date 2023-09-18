"""
    plot_velocity(setup, V, t; kwargs...)

Plot velocity.
"""
function plot_velocity end

plot_velocity(setup, u; kwargs...) =
    plot_velocity(setup.grid.dimension, setup, u; kwargs...)

# 2D version
function plot_velocity(::Dimension{2}, setup, u; kwargs...)
    (; xp, xlims) = setup.grid
    T = eltype(xp[1])

    # Get velocity at pressure points
    up = interpolate_u_p(setup, u)
    # qp = map((u, v) -> √(u^2 + v^2), up, vp)
    qp = sqrt.(up[1] .^ 2 .+ up[2] .^ 2)

    # Levels
    μ, σ = mean(qp), std(qp)
    # ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - T(1.5) * σ, μ + T(1.5) * σ, 10)

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Velocity",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1]..., xlims[2]...)
    cf = contourf!(ax, xp..., qp; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1, 2], cf)
    # Colorbar(fig[2,1], cf; vertical = false)
    fig
end

# 3D version
function plot_velocity(::Dimension{3}, setup, u; kwargs...)
    (; xp) = setup.grid

    # Get velocity at pressure points
    up = interpolate_u_p(setup, u)
    qp = map((u, v, w) -> √sum(u^2 + v^2 + w^2), up...)

    # Levels
    μ, σ = mean(qp), std(qp)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    xp = Array(xp)
    qp = Array(qp)
    contour(xp..., qp; levels, kwargs...)
    # contour(xp..., qp; kwargs...)
end
