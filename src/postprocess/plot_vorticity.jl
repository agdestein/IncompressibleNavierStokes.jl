"""
    plot_vorticity(setup, V, t; kwargs...)

Plot vorticity field.
"""
function plot_vorticity end

# 2D version
function plot_vorticity(setup, V, t; kwargs...)
    (; x, y, xlims, ylims) = setup.grid

    xω = x
    yω = y

    # Get fields
    ω = get_vorticity(V, t, setup)

    # Levels
    μ, σ = mean(ω), std(ω)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    # Plot vorticity
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Vorticity",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(ax, xω, yω, ω; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    Colorbar(fig[1, 2], cf)

    # save("output/vorticity.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_vorticity(setup::Setup{T,3}, V, t; kwargs...) where {T}
    (; grid) = setup
    (; x, y, z) = grid

    xω = x
    yω = y
    zω = z

    ω = get_vorticity(V, t, setup)

    # Levels
    μ, σ = mean(ω), std(ω)
    levels = LinRange(μ - 3σ, μ + 3σ, 10)

    contour(xω, yω, zω, ω; levels, kwargs...)
end
