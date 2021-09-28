"""
Postprocess.
"""
function postprocess(solution, setup)
    @unpack Nx, Ny, Nu, Nv, Nux_in, Nuy_in, Nvx_in, Nvy_in, Npx, Npy, x, y, xp, yp =
        setup.grid
    @unpack V, p, t = solution

    # Reshape
    uh = @view V[1:Nu]
    vh = @view V[Nu+1:Nu+Nv]
    u = reshape(uh, Nux_in, Nuy_in)
    v = reshape(vh, Nvx_in, Nvy_in)
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
    ψ = get_streamfunction(V, t, setup)

    # # Plot vorticity
    levels = [minimum(ω) - 1, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, maximum(ω) + 1]
    pl = contourf(x[2:(end-1)], y[2:(end-1)], ω'; levels, xlabel = "x", ylabel = "y")
    title!(pl, "Vorticity ω")
    display(pl)
    display(sort(levels))

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
    pl = contourf(xp, yp, Δpres'; levels, xlabel = "x", ylabel = "y")
    title!(pl, "Pressure deviation Δp")
    display(pl)

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
    pl = contourf(
        x[2:(end-1)],
        y[2:(end-1)],
        reshape(ψ, Nx - 1, Ny - 1)';
        # levels,
        xlabel = "x",
        ylabel = "y",
    )
    title!(pl, "Stream function ψ")
    display(pl)
end
