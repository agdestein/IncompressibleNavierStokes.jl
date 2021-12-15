"""
    finalize!(processor)

Finalize processing after iterations.
"""
function finalize! end

finalize!(logger::Logger) = nothing
finalize!(plotter::RealTimePlotter) = nothing
finalize!(writer::VTKWriter) = vtk_save(writer.pvd)
finalize!(tracer::QuantityTracer) = nothing
