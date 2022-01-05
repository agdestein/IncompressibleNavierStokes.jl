"""
    process!(processor, stepper)

Process iteration.
"""
function process! end

function process!(logger::Logger, stepper)
    (; V, p, t, setup, cache, momentum_cache) = stepper
    (; model) = setup
    (; F) = cache
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
    (; setup, V, p, t) = stepper
    (; Nx, Ny, Npx, Npy) = setup.grid
    (; field, fieldname) = plotter
    if fieldname == :velocity
        up, vp, wp, qp = get_velocity(V, t, setup)
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
    (; setup, V, p, t) = stepper
    (; xp, yp, zp) = setup.grid;
    tformat = replace(string(t), "." => "p")
    vtk_grid("$(writer.dir)/$(writer.filename)_t=$tformat", xp, yp, zp) do vtk
        up, vp, wp, = get_velocity(V, t, setup)
        vtk["velocity"] = (up, vp, wp)
        vtk["pressure"] = p
        writer.pvd[t] = vtk
    end

    writer
end

function process!(tracer::QuantityTracer, stepper)
    (; V, p, t, setup) = stepper
    maxdiv, umom, vmom, wmom, k = compute_conservation(V, t, setup)
    push!(tracer.t, t)
    push!(tracer.maxdiv, maxdiv)
    push!(tracer.umom, umom)
    push!(tracer.vmom, vmom)
    push!(tracer.wmom, wmom)
    push!(tracer.k, k)
    tracer
end
