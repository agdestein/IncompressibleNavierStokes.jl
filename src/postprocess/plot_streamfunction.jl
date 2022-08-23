"""
    plot_streamfunction(setup, V, t; kwargs...)

Plot streamfunction.
"""
function plot_streamfunction end

# 2D version
function plot_streamfunction(setup::Setup{T,2}, V, t; kwargs...) where {T}
    (; grid, boundary_conditions) = setup
    (; x, y, xlims, ylims) = grid

    if all(==(:periodic), (boundary_conditions.u.x[1], boundary_conditions.v.y[1]))
        xψ = x
        yψ = y
    else
        xψ = x[2:(end - 1)]
        yψ = y[2:(end - 1)]
    end

    # Get fields
    ψ = get_streamfunction(V, t, setup)

    # Levels
    μ, σ = mean(ψ), std(ψ)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    # Plot stream function
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        aspect = DataAspect(),
        title = "Stream function",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(
        ax,
        xψ,
        yψ,
        ψ;
        extendlow = :auto,
        extendhigh = :auto,
        # levels,
        kwargs...,
    )
    Colorbar(fig[1,2], cf)
    # save("output/streamfunction.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_streamfunction(setup::Setup{T,3}, V, t; kwargs...) where {T}
    error("Not implemented (3D)")
end
