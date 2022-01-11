"""
    plot_tracers(tracer)

Plot tracer.
"""
function plot_tracers(tracer)
    (; t, maxdiv, umom, vmom, wmom, k) = tracer
    fig = Figure()

    ax = Axis(fig[1, 1], title = "Maxdiv")
    lines!(ax, t, maxdiv)

    ax = Axis(fig[2, 1], title = "Momentum")
    lines!(ax, t, umom, label = "u")
    lines!(ax, t, vmom, label = "v")
    isempty(wmom) || lines!(ax, t, wmom, label = "w")
    axislegend(ax)

    ax = Axis(fig[3, 1], title = "Kinetic energy")
    lines!(ax, t, k)

    fig
end
