"""
    finalize!(processor)

Finalize processing after iterations.
"""
function finalize! end

finalize!(::Logger) = nothing
finalize!(::StateObserver) = nothing
finalize!(writer::VTKWriter) = vtk_save(writer.pvd)
