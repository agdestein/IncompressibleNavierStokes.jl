"""
    plot_pressure(setup, p; kwargs...)

Plot pressure.
"""
function plot_pressure end

# 2D version
function plot_pressure(setup::Setup{T,2}, p; kwargs...) where {T}
    (; Nx, Ny, Npx, Npy, xp, yp, xlims, ylims) = setup.grid

    # Reshape
    p = reshape(p, Npx, Npy)

    # Levels
    μ, σ = mean(p), std(p)
    ≈(μ + σ, μ; rtol = 1e-8, atol = 1e-8) && (σ = 1e-4)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    # Plot pressure
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Pressure",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    cf = contourf!(ax, xp, yp, p; extendlow = :auto, extendhigh = :auto, levels, kwargs...)
    # Colorbar(fig[1,2], cf)
    Colorbar(fig[1, 2], cf)

    # save("output/pressure.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_pressure(setup::Setup{T,3}, p; kwargs...) where {T}
    (; Nx, Ny, Nz, Npx, Npy, Npz, xp, yp, zp) = setup.grid

    # Reshape
    p = reshape(p, Npx, Npy, Npz)

    # Levels
    μ, σ = mean(p), std(p)
    levels = LinRange(μ - 5σ, μ + 5σ, 10)

    contour(xp, yp, zp, p; levels, kwargs...)

    # save("output/pressure.png", fig, pt_per_unit = 2)
end
