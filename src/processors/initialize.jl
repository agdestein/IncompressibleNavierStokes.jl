"""
    initialize!(processor, stepper)

Initialize processor.
"""
function initialize! end

initialize!(logger::Logger, stepper) = logger

function initialize!(o::StateObserver, stepper) 
    (; V, p, t) = stepper
    o.state[] = (V, p, t)
end

function initialize!(writer::VTKWriter, stepper)
    (; dir, filename) = writer
    ispath(dir) || mkpath(dir)
    pvd = paraview_collection(joinpath(dir, filename))
    @pack! writer = pvd
    writer
end

function initialize!(tracer::QuantityTracer, stepper)
    tracer.t = zeros(0)
    tracer.maxdiv = zeros(0)
    tracer.umom = zeros(0)
    tracer.vmom = zeros(0)
    tracer.wmom = zeros(0)
    tracer.k = zeros(0)
    tracer
end
