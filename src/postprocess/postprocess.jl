"""
Postprocess.
"""
function postprocess(solution, setup)
    @unpack Nx, Ny, x, y = setup.grid
    @unpack V, p, t = solution

    ω_flat = get_vorticity(V, t, setup)
    ω = reshape(ω_flat, Nx - 1, Ny - 1)
    ψ = get_streamfunction(V, t, setup)

    # Plot vorticity
    p = contourf(x[2:(end - 1)], y[2:(end - 1)], ω)
    display(p)
end

