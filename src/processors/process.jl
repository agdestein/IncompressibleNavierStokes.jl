"""
    process!(processor, stepper)

Process iteration.
"""
function process! end

function process!(logger::Logger, stepper)
    (; V, p, t, setup, cache, momentum_cache) = stepper
    (; F) = cache

    momentum!(F, nothing, V, V, p, t, setup, momentum_cache)

    @info "Iteration $(stepper.n)" t norm(F) maximum(F)

    logger
end

function process!(plotter::RealTimePlotter, stepper)
    (; setup, V, p, t) = stepper
    (; Npx, Npy, Npz) = setup.grid
    (; field, lims, fieldname, type) = plotter
    N = get_dimension(setup.grid)
    if fieldname == :velocity
        vels = get_velocity(V, t, setup)
        f = map((vels...) -> √sum(vel -> vel^2, vels), vels...)
        n = 3.0
    elseif fieldname == :vorticity
        # Use preallocated field
        f = vorticity!(field[], V, t, setup)
        n = 3.0
    elseif fieldname == :streamfunction
        f = get_streamfunction(V, t, setup)
        n = 3.0
    elseif fieldname == :pressure
        f = copy(p)
        if N == 2
            f = reshape(f, Npx, Npy)
        elseif N == 3
            f = reshape(f, Npx, Npy, Npz)
        end
        n = 5.0
    end

    field[] = f

    N == 2 && (n = 1.5)
    # nlevel = N == 2 ? 10 : N == 3 ? 5 : error()
    nlevel = 10
    if type == heatmap
        lims[] = get_lims(f, n)
    elseif type ∈ (contour, contourf)
        lims[] = LinRange(get_lims(f, n)..., nlevel)
    end

    plotter
end

function process!(writer::VTKWriter, stepper)
    (; setup, V, p, t) = stepper
    (; xp, yp, zp) = setup.grid
    N = get_dimension(setup.grid)
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
