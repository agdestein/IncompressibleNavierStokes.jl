"""
    finalize!(processor)

Finalize processing after iterations.
"""
function finalize! end

finalize!(::Logger) = nothing
finalize!(::RealTimePlotter) = nothing
finalize!(writer::VTKWriter) = vtk_save(writer.pvd)
finalize!(::QuantityTracer) = nothing
