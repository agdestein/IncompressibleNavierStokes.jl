"""
Postprocess.
"""
function postprocess(solution, setup)
    @unpack Nx, Ny, Nu, Nv, Nux_in, Nuy_in, Nvx_in, Nvy_in, Npx, Npy, x, y, xp, yp =
        setup.grid
    @unpack V, p, t = solution

    # Reshape
    uₕ = @view V[1:Nu]
    vₕ = @view V[Nu+1:Nu+Nv]
    u = reshape(uₕ, Nux_in, Nuy_in)
    v = reshape(vₕ, Nvx_in, Nvy_in)
    pres = reshape(p, Npx, Npy)

    # Shift pressure to get zero pressure in the centre
    if iseven(Nx) && iseven(Ny)
        Δpres = pres .- (pres[Nx÷2+1, Ny÷2+1] + pres[Nx÷2, Ny÷2]) / 2
    else
        Δpres = pres .- pres[ceil(Int, Nx / 2), ceil(Int, Ny / 2)]
    end

    # Get fields
    ω_flat = get_vorticity(V, t, setup)
    ω = reshape(ω_flat, Nx - 1, Ny - 1)
    ψ_flat = get_streamfunction(V, t, setup)
    ψ = reshape(ψ_flat, Nx - 1, Ny - 1)

    # Plot vorticity
    levels = [minimum(ω) - 1, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, maximum(ω) + 1]
    f = Figure()
    ax = Axis(f[1, 1]; aspect = DataAspect(),title = "Vorticity ω", xlabel = "x", ylabel = "y")
    contourf!(
        ax,
        x[2:(end-1)],
        y[2:(end-1)],
        ω;
        #levels,
    )
    display(f)

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
    ax = Axis(f[1, 1]; aspect = DataAspect(),
    title = "Pressure deviation Δp", xlabel = "x", ylabel = "y")
    contourf!(
        ax,
        xp,
        yp,
        Δpres;
        # levels,
    )
    display(f)

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
