"""
    plot_tracers(tracer)

Plot tracer.
"""
function plot_tracers(tracer)
    f = Figure()
    lines(f[1, 1], tracer.t, tracer.maxdiv, label = "maxdiv")
    lines(f[2, 1], tracer.t, tracer.umom, label = "u momentum")
    lines!(f[2, 1], tracer.t, tracer.vmom, label = "v momentum")
    lines!(f[2, 1], tracer.t, tracer.wmom, label = "w momentum")
    lines(f[3, 1], tracer.t, tracer.k, label = "k")
    f
end
