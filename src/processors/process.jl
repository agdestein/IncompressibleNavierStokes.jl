"""
    process!(processor, stepper)

Process iteration.
"""
function process! end

function process!(logger::Logger, stepper)
    @unpack V, p, t, setup, cache, momentum_cache = stepper
    @unpack model = setup
    @unpack F = cache

    # Calculate mass, momentum and energy
    # maxdiv, umom, vmom, k = compute_conservation(V, t, setup)

    # Residual (in Finite Volume form)
    # For k-ϵ model residual also contains k and ϵ terms
    if !isa(model, KEpsilonModel)
        # Norm of residual
        momentum!(F, nothing, V, V, p, t, setup, momentum_cache)
        maxres = maximum(abs.(F))
    end

    println("n = $(stepper.n), t = $t, maxres = $maxres")
    # println("t = $t")

    logger
end

function process!(plotter::RealTimePlotter, stepper)
    @unpack setup, V, p, t = stepper
    @unpack Nx, Ny, Npx, Npy = setup.grid
    @unpack field, fieldname = plotter
    if fieldname == :velocity
        up, vp, qp = get_velocity(V, t, setup)
        field[] = qp
    elseif fieldname == :vorticity
        field[] = vorticity!(field[], V, t, setup)
    elseif fieldname == :streamfunction
        field[] = get_streamfunction(V, t, setup)
    elseif fieldname == :pressure
        field[] = reshape(p, Npx, Npy)
    end
    # sleep(1 / rtp.fps)

    plotter
end

function process!(writer::VTKWriter, stepper)
    @unpack dir, filename = writer
    @unpack setup, V, p, t = stepper
    @unpack xp, yp = setup.grid;
    vtk_grid("$dir/$(filename)_t=$t", xp, yp) do vtk
        up, vp, = get_velocity(V, t, setup)
        vtk["velocity"] = (up, vp, zero(up))
        vtk["pressure"] = p
        writer.pvd[t] = vtk
    end

    writer
end
