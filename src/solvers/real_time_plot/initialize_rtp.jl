"""
    initialize_rtp(setup, V, p, t)

Initialize real time plot (RTP).
"""
function initialize_rtp(setup, V, p, t)
    @unpack x1, x2, y1, y2, Nx, Ny, x, y, xp, yp = setup.grid
    @unpack rtp_type = setup.visualization

    Lx = x2 - x1
    Ly = y2 - y1

    vel = nothing
    ω = nothing
    ψ = nothing
    pres = nothing

    fig = Figure(resolution = (2000 * Lx / (Lx + Ly), 2000 * Ly / (Lx + Ly)))
    if rtp_type == "velocity"
        up, vp, qp = get_velocity(V, t, setup)
        vel = Node(qp)
        ax, hm = contourf(fig[1, 1], xp, yp, vel)
    elseif rtp_type == "vorticity"
        ω = Node(get_vorticity(V, t, setup))
        ax, hm = contourf(fig[1, 1], x[2:end-1], y[2:end-1], ω; levels = -10:2:10)
    elseif rtp_type == "streamfunction"
        ψ = Node(reshape(get_streamfunction(V, t, setup), Nx - 1, Ny - 1))
        ax, hm = contourf(fig[1, 1], x[2:end-1], y[2:end-1], ψ)
    end
    ax.title = titlecase(rtp_type)
    ax.aspect = DataAspect()
    ax.xlabel = "x"
    ax.ylabel = "y"
    limits!(ax, x1, x2, y1, y2)
    Colorbar(fig[1, 2], hm)
    display(fig)
    fps = 60

    (; fps, vel, ω, ψ, pres)
end
