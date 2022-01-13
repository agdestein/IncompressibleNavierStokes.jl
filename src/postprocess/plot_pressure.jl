"""
    plot_pressure(setup, p)

Plot pressure.
"""
function plot_pressure end

# 2D version
function plot_pressure(setup::Setup{T,2}, p) where {T}
    (; Nx, Ny, Npx, Npy, xp, yp, xlims, ylims) = setup.grid

    # Reshape
    pres = reshape(p, Npx, Npy)

    # Shift pressure to get zero pressure in the centre
    if iseven(Nx) && iseven(Ny)
        Δpres = pres .- (pres[Nx÷2+1, Ny÷2+1] + pres[Nx÷2, Ny÷2]) / 2
    else
        Δpres = pres .- pres[ceil(Int, Nx / 2), ceil(Int, Ny / 2)]
    end

    # Plot pressure
    levels = [
        # minimum(Δpres) - 0.1
        -0.002
        0.0
        0.02
        0.05
        0.07
        0.09
        0.11
        0.12
        0.17
        0.3
        # maximum(Δpres) + 0.1
    ]
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect = DataAspect(),
        title = "Pressure deviation Δp",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, xlims[1], xlims[2], ylims[1], ylims[2])
    contourf!(
        ax,
        xp,
        yp,
        Δpres;
        levels,
        extendlow = :auto, extendhigh = :auto,
    )

    # save("output/pressure.png", fig, pt_per_unit = 2)

    fig
end

# 3D version
function plot_pressure(setup::Setup{T,3}, p) where {T}
    (; Nx, Ny, Nz, Npx, Npy, Npz, xp, yp, zp) = setup.grid

    # Reshape
    pres = reshape(p, Npx, Npy, Npz)

    # Shift pressure to get zero pressure in the centre
    if iseven(Nx) && iseven(Ny)
        Δpres = pres .- (pres[Npx÷2+1, Npy÷2+1, Npz÷2+1] + pres[Npx÷2, Npy÷2, Npz÷2]) / 2
    else
        Δpres = pres .- pres[ceil(Int, Npx / 2), ceil(Int, Ny / 2), ceil(Int, Nz / 2)]
    end

    contour(
        xp,
        yp,
        zp,
        Δpres;
        extendlow = :auto, extendhigh = :auto,
    )

    # save("output/pressure.png", fig, pt_per_unit = 2)
end
