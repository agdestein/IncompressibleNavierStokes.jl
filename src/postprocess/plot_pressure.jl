"""
Plot pressure.
"""
function plot_pressure(solution, setup)
    @unpack Nx, Ny, Npx, Npy, xp, yp = setup.grid
    @unpack p = solution

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
        minimum(Δpres) - 0.1
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
        maximum(Δpres) + 0.1
    ]
    f = Figure()
    ax = Axis(
        f[1, 1];
        aspect = DataAspect(),
        title = "Pressure deviation Δp",
        xlabel = "x",
        ylabel = "y",
    )
    contourf!(
        ax,
        xp,
        yp,
        Δpres;
        # levels,
    )
    display(f)
end
