"""
    plot_tracers(tracer)

Plot tracer.
"""
function plot_tracers(tracer)
    (; t, maxdiv, umom, vmom, wmom, k) = tracer
    fig = Figure()

    ax = Axis(fig[1, 1]; xlabel = "t", title = "Maxdiv")
    lines!(ax, t, maxdiv)

    ax = Axis(fig[2, 1]; xlabel = "t", title = "Momentum")
    lines!(ax, t, umom, label = "u")
    lines!(ax, t, vmom, label = "v")
    isempty(wmom) || lines!(ax, t, wmom, label = "w")
    axislegend(ax)

    ax = Axis(fig[3, 1]; xlabel = "t", title = "Kinetic energy")
    ylims = extrema(k)
    ≈(ylims...; rtol = 1e-6, atol = 1e-10) && ylims!(ax, (0, max(1, 2ylims[2])))
    lines!(ax, t, k)

    fig
end
