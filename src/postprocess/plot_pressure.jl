"""
    plot_pressure(setup, p; kwargs...)

Plot pressure.
"""
function plot_pressure end

plot_pressure(setup, p; kwargs...) =
    plot_pressure(setup.grid.dimension, setup, p; kwargs...)

# 2D version
function plot_pressure(::Dimension{2}, setup, p; kwargs...)
    (; xp, xlims) = setup.grid

    xp = Array.(xp)
    p = Array(p)

    T = eltype(xp[1])

    # Levels
    μ, σ = mean(p), std(p)
    # ≈(μ + σ, μ; rtol = sqrt(eps(T)), atol = sqrt(eps(T))) && (σ = sqrt(sqrt(eps(T))))
    levels = LinRange(μ - T(1.5) * σ, μ + T(1.5) * σ, 10)

    # Plot pressure
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Pressure",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1]..., xlims[2]...)
    cf = contourf!(ax, xp..., p; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1, 2], cf)

    # save("output/pressure.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_pressure(::Dimension{3}, setup, p; kwargs...)
    (; xp) = setup.grid

    # Levels
    μ, σ = mean(p), std(p)
    levels = LinRange(μ - 5σ, μ + 5σ, 10)

    p = Array(p)
    contour(xp..., p; levels, kwargs...)

    # save("output/pressure.png", fig, pt_per_unit = 2)
end
