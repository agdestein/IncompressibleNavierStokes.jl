"""
Postprocess.
"""
function postprocess(solution, setup)
    plot_vorticity(solution, setup)
    plot_pressure(solution, setup)
    plot_streamfunction(solution, setup)
end
