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

    # Shift pressure to get zero pressure in the centre
    if iseven(Nx) && iseven(Ny)
        Δp = p .- (p[Nx ÷ 2 + 1, Ny ÷ 2 + 1] + p[Nx ÷ 2, Ny ÷ 2]) / 2
    else
        Δp = p .- p[ceil(Int, Nx / 2), ceil(Int, Ny / 2)]
    end

    # Levels
    μ, σ = mean(Δp), std(Δp)
    levels = LinRange(μ - 1.5σ, μ + 1.5σ, 10)

    # Plot pressure
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Pressure deviation Δp",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(ax, xp, yp, Δp; extendlow = :auto, extendhigh = :auto, levels, kwargs...)

    # save("output/pressure.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_pressure(setup::Setup{T,3}, p; kwargs...) where {T}
    (; Nx, Ny, Nz, Npx, Npy, Npz, xp, yp, zp) = setup.grid

    # Reshape
    p = reshape(p, Npx, Npy, Npz)

    # Shift pressure to get zero pressure in the centre
    if iseven(Nx) && iseven(Ny)
        pmid = (p[Npx ÷ 2 + 1, Npy ÷ 2 + 1, Npz ÷ 2 + 1] + p[Npx ÷ 2, Npy ÷ 2, Npz ÷ 2]) / 2
    else
        pmid = p[ceil(Int, Npx / 2), ceil(Int, Ny / 2), ceil(Int, Nz / 2)]
    end
    Δp = p .- pmid

    # Levels
    μ, σ = mean(Δp), std(Δp)
    levels = LinRange(μ - 5σ, μ + 5σ, 10)

    contour(xp, yp, zp, Δp; levels, kwargs...)

    # save("output/pressure.png", fig, pt_per_unit = 2)
end
