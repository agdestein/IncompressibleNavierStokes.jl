"""
    finalize!(processor)

Finalize processing after iterations.
"""
function finalize! end

finalize!(logger::Logger) = nothing
finalize!(plotter::RealTimePlotter) = nothing
