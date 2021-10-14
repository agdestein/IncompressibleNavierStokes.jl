"""
Plot streamfunction.
"""
function plot_streamfunction(setup, V, t)
    @unpack bc = setup
    @unpack Nx, Ny, x, y, x1, x2, y1, y2 = setup.grid

    if bc.u.left == :periodic
        xψ = x
    else
        xψ = x[2:end-1]
    end
    if bc.v.low == :periodic
        yψ = y
    else
        yψ = y[2:end-1]
    end

    # Get fields
    ψ = get_streamfunction(V, t, setup)

    # Plot stream function
    levels = [
        minimum(ψ) - 1
        -0.1175
        -0.115
        -0.11
        -0.1
        -0.09
        -0.07
        -0.05
        -0.03
        -0.01
        -0.0001
        -1.0e-5
        -1.0e-10
        0.0
        1.0e-6
        1.0e-5
        5.0e-5
        0.0001
        0.00025
        0.0005
        0.001
        0.0015
        maximum(ψ) + 1
    ]
    f = Figure()
    ax = Axis(
        f[1, 1],
        aspect = DataAspect(),
        title = "Stream function ψ",
        xlabel = "x",
        ylabel = "y",
    )
    limits!(ax, x1, x2, y1, y2)
    contourf!(
        ax,
        xψ,
        yψ,
        ψ;
        # levels,
    )
    display(f)
    save("output/streamfunction.png", f, pt_per_unit = 2)
end
