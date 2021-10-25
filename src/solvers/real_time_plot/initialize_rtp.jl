"""
    initialize_rtp(setup, V, p, t)

Initialize real time plot (RTP).
"""
function initialize_rtp(setup, V, p, t)
    @unpack bc = setup
    @unpack x1, x2, y1, y2, Nx, Ny, x, y, xp, yp = setup.grid
    @unpack rtp_type = setup.visualization

    Lx = x2 - x1
    Ly = y2 - y1

    vel = nothing
    ω = nothing
    ψ = nothing
    pres = nothing

    refsize = 2000

    fig = Figure(resolution = (refsize * Lx / (Lx + Ly), refsize * Ly / (Lx + Ly) + 100))
    if rtp_type == "velocity"
        up, vp, qp = get_velocity(V, t, setup)
        vel = Node(qp)
        ax, hm = contourf(fig[1, 1], xp, yp, vel)
    elseif rtp_type == "vorticity"
        if bc.u.x[1] == :periodic
            xω = x
        else
            xω = x[2:end-1]
        end
        if bc.v.y[1] == :periodic
            yω = y
        else
            yω = y[2:end-1]
        end
        ω = Node(get_vorticity(V, t, setup))
        ax, hm = contour(fig[1, 1], xω, yω, ω; levels = -10:2:10)
        # ax, hm = contourf(fig[1, 1], xω, yω, ω; levels = -10:2:10)
    elseif rtp_type == "streamfunction"
        if bc.u.x[1] == :periodic
            xψ = x
        else
            xψ = x[2:end-1]
        end
        if bc.v.y[1] == :periodic
            yψ = y
        else
            yψ = y[2:end-1]
        end
        ψ = Node(get_streamfunction(V, t, setup))
        ax, hm = contourf(fig[1, 1], xψ, yψ, ψ)
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
