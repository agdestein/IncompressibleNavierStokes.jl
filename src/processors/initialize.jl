"""
    initialize!(processor, stepper)

Initialize processor.
"""
function initialize! end

initialize!(logger::Logger, stepper) = logger
initialize!(observer::StateObserver, stepper) = observer

function initialize!(writer::VTKWriter, stepper)
    (; dir, filename) = writer
    ispath(dir) || mkpath(dir)
    pvd = paraview_collection(joinpath(dir, filename))
    writer.pvd = pvd
    writer
end
