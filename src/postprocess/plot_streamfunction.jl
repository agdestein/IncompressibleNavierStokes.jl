"""
Plot streamfunction.
"""
function plot_streamfunction(solution, setup)
    @unpack Nx, Ny, x, y =  setup.grid
    @unpack V, p, t = solution

    # Get fields
    ψ_flat = get_streamfunction(V, t, setup)
    ψ = reshape(ψ_flat, Nx - 1, Ny - 1)

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
    contourf!(
        ax,
        x[2:(end-1)],
        y[2:(end-1)],
        ψ;
        # levels,
    )
    display(f)
end
