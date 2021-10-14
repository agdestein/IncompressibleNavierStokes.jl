"""
Postprocess.
"""
function postprocess(setup, V, p, t)
    plot_pressure(setup, p)
    plot_vorticity(setup, V, t)
    plot_streamfunction(setup, V, t)
end
