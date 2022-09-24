"""
    process!(processor, stepper)

Process iteration.
"""
function process! end

function process!(logger::Logger, stepper)
    (; n, t) = stepper
    @printf "Iteration %d\tt = %g\n" n t
    logger
end

function process!(o::StateObserver, stepper) 
    (; V, p, t) = stepper
    o.state[] = (V, p, t)
end

function process!(writer::VTKWriter, stepper)
    (; setup, V, p, t) = stepper
    (; grid) = setup
    (; xp, yp, zp) = grid

    N = get_dimension(grid)
    if N == 2
        coords = (xp, yp)
    elseif N == 3
        coords = (xp, yp, zp)
    end

    tformat = replace(string(t), "." => "p")
    vtk_grid("$(writer.dir)/$(writer.filename)_t=$tformat", coords...) do vtk
        vels = get_velocity(V, t, setup)
        if N == 2
            # ParaView prefers 3D vectors. Add zero z-component.
            wp = zeros(size(vels[1]))
            vels = (vels..., wp)
        end
        vtk["velocity"] = vels
        vtk["pressure"] = p
        writer.pvd[t] = vtk
    end

    writer
end

function process!(tracer::QuantityTracer, stepper)
    (; V, t, setup) = stepper
    N = get_dimension(setup.grid)
    if N == 2
        maxdiv, umom, vmom, k = compute_conservation(V, t, setup)
    elseif N == 3
        maxdiv, umom, vmom, wmom, k = compute_conservation(V, t, setup)
    end
    push!(tracer.t, t)
    push!(tracer.maxdiv, maxdiv)
    push!(tracer.umom, umom)
    push!(tracer.vmom, vmom)
    N == 3 && push!(tracer.wmom, wmom)
    push!(tracer.k, k)
    tracer
end
