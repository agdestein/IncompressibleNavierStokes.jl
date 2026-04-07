module Bubbles

using Adapt
using CUDA
using CUDSS
using LinearAlgebra
using StaticArrays
using WGLMakie

import AcceleratedKernels as AK
import IncompressibleNavierStokes as NS

getbackend() = CUDA.functional() ? CUDA.CUDABackend() : NS.KernelAbstractions.CPU()
# getbackend() = NS.KernelAbstractions.CPU()

getparams() = (;
    # Domain
    L = 1.0, # Domain side length
    n = 64, # Number of grid points

    # Flow
    viscosity = 5.0e-4,
    lidvelocity = 1.0,
    # dens = (1.0e3, 1.25),
    # grav = (0.0, 0.0, -9.81),
    # visc = (1.0e-3, 1.8e-5),

    # Bubble
    bubble = (;
        σ = 1.0e0, # Surface tension coefficient
        center = (0.6, 0.6), # Initial bubble center
        radius = 0.15, # Initial bubble radius
        npoint = 51, # Initial number of control points for the bubble surface
        markerlims = (1.0e-2, 5.0e-2), # Marker length limits for remeshing
        angle_min = π / 8, # Minimum allowed angle between adjacent segments
    ),

    # Time integration
    dt = 2.0e-3,
    nsubstep = 10, # Steps between plot updates
    nstep = 1000,

    # Plotting
    plotting = (;
        step = 2, # Grid point frequency of arrow plot
        lengthscale = 0.1, # Scaling factor for arrow lengths in the plot
        plotsurfacetension = true, # Show the surface tension vectors for each marker
    ),
)

# Small alias for point, until I decide on what type to use.
# A simple tuple would be the cleanest, but we cannot do `(a, b) + (x, y)` directly,
# whereas
# `Point(a, b) + Point(x, y)` or
# `@SVector [a, b] + @SVector [x, y]`
# works out of the box.
MyPoint(x, y) = @SVector [x, y] # StaticArrays
# MyPoint(x, y) = Point(x, y) # GeometryBasics

"Get lid-driven cavity setup."
function lidsetup()
    params = getparams()
    ax = NS.tanh_grid(0.0, params.L, params.n)
    setup = NS.Setup(;
        x = (ax, ax),
        boundary_conditions = (;
            u = (
                (NS.DirichletBC(), NS.DirichletBC()),
                (NS.DirichletBC(), NS.DirichletBC((params.lidvelocity, 0.0))),
            ),
        ),
        backend = getbackend(),
        # backend = NS.KernelAbstractions.CPU(),
    )
    return setup
end

# "Compute surface tension at markers."
# function surfacetension!(tension, x)
#     (; σ) = getparams().bubble
#     c = curvature(x)
#     n = normals(x)
#     @. tension = -σ * c * n
#     return nothing
# end

"Compute surface tension at markers."
function surfacetension!(tension, x)
    (; σ) = getparams().bubble
    # c = curvature(x)
    n = normals(x)
    npoint = length(x)
    AK.foreachindex(tension) do i
        nleft = n[mod1(i - 1, npoint)]
        nright = n[mod1(i + 1, npoint)]
        fleft = MyPoint(-nleft[2], nleft[1])
        fright = MyPoint(nright[2], -nright[1])
        tension[i] = -σ * (fleft + fright)
    end
    return nothing
end

"Left index `n` times away in direction `i`."
@inline left(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] - n : I[j], D))

"Right index `n` times away in direction `i`."
@inline right(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] + n : I[j], D))

"""
Mask segment p-q by rectangle a-b-c-d-a, where p, q, a, b, c, d are 2D `MyPoint`s.
Return the masked segment and an indicator for whether any part of the original segment is contained in the rectangle.
If `intersect` is false, the original segment is returned but should be ignored.
"""
function masksegment(segment, rect)
    p, q = segment
    a, b, c, d = rect

    # Axis-aligned bounds from the four corners (a=bottom-left, b=bottom-right,
    # c=top-right, d=top-left)
    xmin, xmax = a[1], b[1]
    ymin, ymax = a[2], c[2]

    dx = q[1] - p[1]
    dy = q[2] - p[2]

    # Liang-Barsky: parametrize segment as P(t) = p + t*(q-p), t ∈ [0,1].
    # Each boundary gives a constraint p_k*t <= q_k.
    # pk < 0 → outside→inside, update t0; pk > 0 → inside→outside, update t1.
    t0 = zero(dx)
    t1 = one(dx)

    for (pk, qk) in (
            (-dx, p[1] - xmin),  # left boundary:   x >= xmin
            (dx, xmax - p[1]),  # right boundary:  x <= xmax
            (-dy, p[2] - ymin),  # bottom boundary: y >= ymin
            (dy, ymax - p[2]),  # top boundary:    y <= ymax
        )
        if pk == 0
            qk < 0 && return (p, q), false   # parallel and outside
        elseif pk < 0
            t0 = max(t0, qk / pk)
        else
            t1 = min(t1, qk / pk)
        end
    end

    t0 > t1 && return (p, q), false

    p_clip = MyPoint(p[1] + t0 * dx, p[2] + t0 * dy)
    q_clip = MyPoint(p[1] + t1 * dx, p[2] + t1 * dy)

    return (p_clip, q_clip), true
end

"Interpolate surface tension force at markers to velocity points."
function interpolate_tension!(Fu, bub_F, bub_x, setup)
    (; Δ, Iu) = setup
    xu = setup.x[1][2:end], setup.x[2][2:end]
    npoint = length(bub_x)

    # Loop through all velocity components and let that component receive its due contribution of the integral of the force
    # for dim in 1:2, I in Iu[dim]
    AK.foreachindex(Fu) do index
        II = CartesianIndices(Fu)[index]
        i, j, dim = II.I
        I = CartesianIndex(i, j)
        otherdim = ifelse(dim == 1, 2, 1)
        # i, j = I.I

        I in Iu[dim] || return nothing

        # Accumulator for `dim`-component of marker forces
        FuI = zero(eltype(Fu))

        # Loop through all markers and add their contribution to the current velocity point (if any).
        for ipoint in 1:npoint
            p = bub_x[ipoint]
            q = bub_x[mod1(ipoint + 1, npoint)]

            # "Mask" the line segment (p, q) by the current rectangle defined
            # by the product of the intervals
            # (xu[1][i-1], xu[1][i+1]) and (xu[2][j-1], xu[2][j]) for u1, and
            # (xu[1][i-1], xu[1][i]) and (xu[2][j-1], xu[2][j+1]) for u2.
            a = MyPoint(xu[1][i - 1], xu[2][j - 1])
            b = MyPoint(xu[1][i + (dim == 1)], xu[2][j - 1])
            c = MyPoint(xu[1][i + (dim == 1)], xu[2][j + (dim == 2)])
            d = MyPoint(xu[1][i - 1], xu[2][j + (dim == 2)])

            (p, q), intersect = masksegment((p, q), (a, b, c, d))
            intersect || continue # Go to next marker if no intersection with current velocity control volume

            p, q = ifelse(p[otherdim] < q[otherdim], (p, q), (q, p)) # Ensure p is "left" of q in the integral dimension

            t = bub_F[ipoint][dim] # Surface tension of current marker and current component. This is just a single scalar constant value.

            # Integrate t · w(x_dim(s)) over s = x_otherdim from p to q.
            # x_dim(s) is linear in s, so substitute xd = x_dim(s):
            #   ds = dxd / slope,  integral = t · (W(xd_q) - W(xd_p)) / slope
            # where W is the antiderivative of w (piecewise quadratic, continuous at xM).
            # Special-case slope ≈ 0: w is constant, integral = t · w(xd_p) · s_span.
            xL = xu[dim][I[dim] - 1]
            xM = xu[dim][I[dim]]
            xR = xu[dim][I[dim] + 1]
            hL = xM - xL
            hR = xR - xM

            xd_p = p[dim]
            xd_q = q[dim]
            s_span = q[otherdim] - p[otherdim]   # > 0 after sorting

            # Antiderivative of w, continuous on [xL, xR]
            W(xd) = xd <= xM ? (xd - xL)^2 / (2 * hL) : hL / 2 + hR / 2 - (xR - xd)^2 / (2 * hR)
            w(xd) = xd <= xM ? (xd - xL) / hL : (xR - xd) / hR

            slope = (xd_q - xd_p) / s_span
            contribution = if slope == 0
                t * w(xd_p) * s_span
            else
                t * (W(xd_q) - W(xd_p)) / slope
            end

            # Add contribution from current marker
            FuI += contribution
        end

        # Now write the total contribution from all markers to the current velocity point.
        # Add to existing force (convection-diffusion etc.).
        # Also normalize by the current grid spacing, since all the integrals are weighed by marker lengths
        Fu[I, dim] += FuI / Δ[otherdim][I[otherdim]]

        return nothing
    end
    return nothing
end

"Interpolate velocity field to marker control points."
function interpolate_velocity!(bub_u, bub_x, u, setup)
    (; xp) = setup
    xu = setup.x[1][2:end], setup.x[2][2:end]
    AK.foreachindex(bub_u) do ipoint
        x1, x2 = bub_x[ipoint]

        # Find first velocity point to the RIGHT of marker control point
        # Note: Two types of points: volume centers ("p") and volume edges ("u").
        # The staggered velocity field is indexed as follows:
        # u[i, j, 1] is defined at (RIGHTEDGE, CENTER) of volume (i, j) and
        # u[i, j, 2] is defined at (CENTER, RIGHTEDGE) of volume (i, j).

        # ip = findfirst(>(x1), xp[1])
        # iu = findfirst(>(x1), xu[1])
        # jp = findfirst(>(x2), xp[2])
        # ju = findfirst(>(x2), xu[2])

        ip = 1; while ip < length(xp[1]) && xp[1][ip] < x1; ip += 1; end
        iu = 1; while iu < length(xu[1]) && xu[1][iu] < x1; iu += 1; end
        jp = 1; while jp < length(xp[2]) && xp[2][jp] < x2; jp += 1; end
        ju = 1; while ju < length(xu[2]) && xu[2][ju] < x2; ju += 1; end

        # Linear interpolation weights for each dimension and CENTER/EDGE type
        w1u = (x1 - xu[1][iu - 1]) / (xu[1][iu] - xu[1][iu - 1])
        w1p = (x1 - xp[1][ip - 1]) / (xp[1][ip] - xp[1][ip - 1])
        w2u = (x2 - xu[2][ju - 1]) / (xu[2][ju] - xu[2][ju - 1])
        w2p = (x2 - xp[2][jp - 1]) / (xp[2][jp] - xp[2][jp - 1])

        # Compute velocity at marker control point by bilinear interpolation
        bub_u1 =
            (1 - w1u) * (1 - w2p) * u[iu - 1, jp - 1, 1] +
            w1u * (1 - w2p) * u[iu, jp - 1, 1] +
            (1 - w1u) * w2p * u[iu - 1, jp, 1] +
            w1u * w2p * u[iu, jp, 1]

        bub_u2 =
            (1 - w1p) * (1 - w2u) * u[ip - 1, ju - 1, 2] +
            w1p * (1 - w2u) * u[ip, ju - 1, 2] +
            (1 - w1p) * w2u * u[ip - 1, ju, 2] +
            w1p * w2u * u[ip, ju, 2]

        bub_u[ipoint] = MyPoint(bub_u1, bub_u2)
    end

    return nothing
end

"""
Remesh the bubble surface.

- Splits segments longer than `markerlims[2]` and merges segments shorter than `markerlims[1]`.
- Replaces any node where the interior angle between adjacent segments is smaller than
  `angle_min` with the two midpoints of its adjacent segments, removing the sharp corner.
"""
function remesh(xdevice, markerlims, angle_min)
    lmin, lmax = markerlims

    # Do this on the CPU for now
    # TODO: Do on GPU
    cpu = NS.KernelAbstractions.CPU()
    dev = NS.KernelAbstractions.get_backend(xdevice)
    x = adapt(cpu, xdevice)

    # Pass 1: length-based split/merge
    result = eltype(x)[]
    n = length(x)
    i = 1
    while i <= n
        p = x[i]
        push!(result, p)
        pnext = x[mod1(i + 1, n)]
        dx = pnext[1] - p[1]
        dy = pnext[2] - p[2]
        d = sqrt(dx^2 + dy^2)
        if d > lmax
            # Split: insert evenly-spaced midpoints targeting segment length (lmin+lmax)/2
            nseg = ceil(Int, d / ((lmin + lmax) / 2))
            for k in 1:(nseg - 1)
                t = k / nseg
                push!(result, MyPoint(p[1] + t * dx, p[2] + t * dy))
            end
            i += 1
        elseif d < lmin && i < n
            # Merge: skip the next point (too close)
            i += 2
        else
            i += 1
        end
    end

    # Pass 2: angle-based refinement
    # At each node with a sharp interior angle, insert midpoints on both adjacent
    # segments so the sharp feature is well-resolved.
    result2 = eltype(x)[]
    n2 = length(result)
    for i in 1:n2
        pprev = result[mod1(i - 1, n2)]
        p = result[i]
        pnext = result[mod1(i + 1, n2)]

        # Incoming and outgoing tangent vectors
        t1x = p[1] - pprev[1];  t1y = p[2] - pprev[2]
        t2x = pnext[1] - p[1];  t2y = pnext[2] - p[2]
        l1 = sqrt(t1x^2 + t1y^2)
        l2 = sqrt(t2x^2 + t2y^2)

        sharp = if l1 > 0 && l2 > 0
            # Interior angle: angle between the reversed incoming tangent and outgoing tangent
            cos_θ = (-t1x * t2x - t1y * t2y) / (l1 * l2)
            acos(clamp(cos_θ, -1.0, 1.0)) < angle_min
        else
            false
        end

        if sharp
            # Replace p with midpoints on both adjacent segments, removing the sharp node
            push!(result2, MyPoint((pprev[1] + p[1]) / 2, (pprev[2] + p[2]) / 2))
            push!(result2, MyPoint((p[1] + pnext[1]) / 2, (p[2] + pnext[2]) / 2))
        else
            push!(result2, p)
        end
    end

    result3 = adapt(dev, result2)

    return result3
end

"""
Compute the curvature at the center of each marker (line segment).

Uses the Menger curvature (signed circumradius formula) at each node via its two
neighbours, then averages adjacent nodes to get the value at each segment center.
Returns a vector of length n, where entry i is the curvature at the center of the
segment from points[i] to points[i+1].
"""
function curvature(points)
    n = length(points)
    # Signed Menger curvature at each node i using (i-1, i, i+1)
    κ_nodes = map(1:n) do i
        A = points[mod1(i - 1, n)]
        B = points[i]
        C = points[mod1(i + 1, n)]
        cross_z = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])
        ab = sqrt((B[1] - A[1])^2 + (B[2] - A[2])^2)
        bc = sqrt((C[1] - B[1])^2 + (C[2] - B[2])^2)
        ac = sqrt((C[1] - A[1])^2 + (C[2] - A[2])^2)
        2 * cross_z / (ab * bc * ac)
    end
    # Curvature at center of segment i = average of curvatures at its two endpoints
    return map(i -> (κ_nodes[i] + κ_nodes[mod1(i + 1, n)]) / 2, 1:n)
end

function normals(pointsdevice)
    # For now: Do on CPU.
    # TODO: Do on GPU
    cpu = NS.KernelAbstractions.CPU()
    dev = NS.KernelAbstractions.get_backend(pointsdevice)
    points = adapt(cpu, pointsdevice)

    n = length(points)
    # Shoelace signed area: positive = CCW winding
    area = sum(1:n) do i
        p, q = points[i], points[mod1(i + 1, n)]
        p[1] * q[2] - q[1] * p[2]
    end / 2
    # CCW: outward normal = tangent rotated 90° clockwise = (dy, -dx)
    # CW:  outward normal = tangent rotated 90° counter-clockwise = (-dy, dx)
    s = sign(area)
    nn = map(1:n) do i
        p, q = points[i], points[mod1(i + 1, n)]
        dx, dy = q[1] - p[1], q[2] - p[2]
        nx, ny = s * dy, -s * dx
        len = sqrt(nx^2 + ny^2)
        MyPoint(nx / len, ny / len)
    end
    return adapt(dev, nn)
end

function edgecenters(points)
    n = length(points)
    centers = adapt(NS.KernelAbstractions.get_backend(points), fill(MyPoint(0.0, 0.0), n))
    AK.foreachindex(centers) do i
        x = (points[i][1] + points[mod1(i + 1, n)][1]) / 2
        y = (points[i][2] + points[mod1(i + 1, n)][2]) / 2
        centers[i] = MyPoint(x, y)
    end
    return centers
end

"""
Perform one time step for the total state `U = (; u, x)`, where
`u` is the velocity field and `x` are the control points defining the bubble.
Wray's low-storage RK3 method is used, which only relies on two 
temporary registers `F` and `U0` (same size as `U`).
In addition, we need a pressure register `p` and a surface tension register `tension`.
"""
function rk3step!(F, U0, U, t, dt, tension, p, psolver, viscosity, setup)
    # RK coefficients
    a = 8 / 15, 5 / 12, 3 / 4
    b = 1 / 4, 0.0
    c = 0.0, 8 / 15, 2 / 3
    nstage = length(a)

    # Update current solution
    t0 = t
    foreach(copyto!, U0, U)

    # RK3 substeps
    for i in 1:nstage
        # Apply right-hand side function to current state U, put in F
        fill!(F.u, 0) # Initialize with 0
        NS.convectiondiffusion!(F.u, U.u, setup, viscosity) # This adds to existing force
        surfacetension!(tension, U.x) # This allocates a new array for now
        interpolate_tension!(F.u, tension, U.x, setup) # Add surface tension to existing force
        interpolate_velocity!(F.x, U.x, U.u, setup) # Interpolate velocity to control points

        # Evolve U = U0 + Δt * a[i] * F
        t = t0 + c[i] * dt
        foreach(copyto!, U, U0)
        @. U.u += a[i] * dt * F.u
        @. U.x += a[i] * dt * F.x
        NS.apply_bc_u!(U.u, t, setup)
        NS.project!(U.u, setup; psolver, p)

        # Evolve U0 = U0 + Δt * b[i] * F
        # Skip for last iter
        if i < nstage
            @. U0.u += b[i] * dt * F.u
            @. U0.x += b[i] * dt * F.x
        end

        # Fill boundary values at new time
        NS.apply_bc_u!(U.u, t, setup)
    end

    # Full time step
    t = t0 + dt

    return nothing
end

"""
Plot velocity field and bubble.
The plot is update when the observable Uobs[] = (; u, x) is updated."
"""
function plotstate(Uobs, setup)
    size = 600, 650

    cpu = AK.KernelAbstractions.CPU()

    (; plotting, bubble) = getparams()
    (; σ) = bubble
    (; step, lengthscale, plotsurfacetension) = plotting

    fig = Figure(; size)
    ax = Axis(
        fig[1, 1];
        title = "Velocity field and bubble",
        xlabel = "x",
        ylabel = "y",
        aspect = DataAspect()
    )

    # Plot velocity field as arrows
    # x1 = range(0.0, 1.0, 32)
    # x2 = range(0.0, 1.0, 32)
    x1 = setup.xp[1][2:step:end-1] |> adapt(cpu)
    x2 = setup.xp[2][2:step:end-1] |> adapt(cpu)
    u1 = map(Uobs) do (; u)
        ub = u[2:step:(end - 1), 2:step:(end - 1), 1]
        ua = u[1:step:(end - 2), 2:step:(end - 1), 1]
        avg = @. (ua + ub) / 2
        adapt(cpu, avg)
    end
    u2 = map(Uobs) do (; u)
        ub = u[2:step:(end - 1), 2:step:(end - 1), 2]
        ua = u[2:step:(end - 1), 1:step:(end - 2), 2]
        avg = @. (ua + ub) / 2
        adapt(cpu, avg)
    end
    arrows2d!(ax, x1, x2, u1, u2; lengthscale, label = "Velocity")

    # Plot bubble surface
    pp = map(Uobs) do U
        Ux = adapt(cpu, U.x)
        return map(Point2, [Ux; [Ux[1]]])
    end
    scatterlines!(ax, pp; label = "Bubble surface")

    # Plot surface tension
    if plotsurfacetension
        ppedge = map(Uobs) do U
            c = edgecenters(U.x)
            cc = adapt(cpu, c)
            return map(Point2, cc)
        end
        nn = map(Uobs) do U
            tension = similar(U.x)
            surfacetension!(tension, U.x)
            return adapt(cpu, tension)
        end
        arrows2d!(ax, ppedge, nn; lengthscale = 0.03, color = Makie.wong_colors()[2], label = "Surface tension")
    end

    Legend(
        fig[0, 1], ax;
        tellwidth = false,
        orientation = :horizontal,
        framevisible = false,
    )

    return fig
end

function solveandplot(u, x, setup, psolver)
    params = getparams()
    (; viscosity, dt, nsubstep, nstep) = params
    (; markerlims, angle_min) = params.bubble

    # Allocate registers
    U = (; u, x) # Current state
    U0 = deepcopy(U) # RK3 accumulator for previous stages
    F = deepcopy(U) # RK3 right hand side
    tension = similar(x) # Surface tension at markers
    p = NS.scalarfield(setup) # Pressure

    # Create plot
    Uobs = Observable(U)
    fig = plotstate(Uobs, setup) # This plot "listens" to changes in `Uobs`
    display(fig)

    t = 0.0
    for itime in 1:nstep
        for isub in 1:nsubstep
            # Perform one RK3 step of step size `dt`
            rk3step!(F, U0, U, t, dt, tension, p, psolver, viscosity, setup)
            t += dt

            # Remesh the bubble.
            # Since this changes the size of the point vector, the temporary registers
            # need to be reshaped as well. For now, just allocate new vectors.
            # TODO: Remesh without too much reallocation? (for heavy 3D triangulations)
            # TODO: Maybe only do this every `nremesh` steps? (for heavy 3D triangulations)
            x = remesh(U.x, markerlims, angle_min)
            U = (; U.u, x)
            U0 = (; U0.u, x = zero(x))
            F = (; F.u, x = zero(x))
            tension = similar(x)

            # @info "itime = $itime / $nstep, isub = $isub / $nsubstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)
        end

        @info "itime = $itime / $nstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)

        # Update plot
        Uobs[] = U
        sleep(0.005)
    end

    return U
end

"""
Create a circular bubble centered at `center` with radius `radius`,
discretized by `npoint` control points.
"""
function bubble()
    (; center, radius, npoint) = getparams().bubble
    b = map(1:npoint) do i
        x0, y0 = center
        angle = 2π * i / npoint
        x = x0 + radius * cos(angle)
        y = y0 + radius * sin(angle)
        return MyPoint(x, y)
    end
    return adapt(getbackend(), b)
end

"Make illustration plot of the rectangular segment masking procedure."
function illustrate_masking()
    # Unit rectangle (0,0)–(1,1), corners in CCW order: BL, BR, TR, TL
    rect = [MyPoint(0.0, 0.0), MyPoint(1.0, 0.0), MyPoint(1.0, 1.0), MyPoint(0.0, 1.0)]
    rect_loop = map(Point2, [rect; [rect[1]]])   # closed polygon for plotting

    function draw_case!(ax, segment, title)
        p, q = segment
        masked, hit = masksegment(segment, rect)
        pm, qm = masked

        # Rectangle outline
        lines!(ax, rect_loop; color = :black, linewidth = 2)

        # Original segment (dashed, gray)
        scatterlines!(ax, [Point2(p), Point2(q)]; color = (:gray, 0.6), linewidth = 1.5, markersize = 8)

        # Clipped segment (solid blue) or a cross indicating no intersection
        if hit
            scatterlines!(ax, [Point2(pm), Point2(qm)]; color = Makie.wong_colors()[1], linewidth = 3, markersize = 8)
        else
            text!(ax, 0.5, 0.5; text = "no intersection", align = (:center, :center), color = :red)
        end

        ax.title = title
        ax.aspect = DataAspect()
        # xlims!(ax, -0.65, 1.65)
        # ylims!(ax, -0.65, 1.65)

        return nothing
    end

    fig = Figure(size = (800, 800))

    # Case 1: segment crosses left and right boundaries (typical case)
    draw_case!(
        Axis(fig[1, 1]),
        (MyPoint(-0.4, 0.4), MyPoint(1.4, 0.6)),
        "Crosses left & right boundaries",
    )

    # Case 2: one endpoint inside, one outside (exits through right boundary)
    draw_case!(
        Axis(fig[1, 2]),
        (MyPoint(0.4, 0.5), MyPoint(1.8, 0.2)),
        "One endpoint inside, one outside",
    )

    # Case 3: segment fully outside the rectangle (no intersection)
    draw_case!(
        Axis(fig[2, 1]),
        (MyPoint(1.3, 0.2), MyPoint(1.8, 0.8)),
        "Fully outside — no intersection",
    )

    # Case 4: diagonal segment clipped exactly at two corners (0,0) and (1,1)
    draw_case!(
        Axis(fig[2, 2]),
        (MyPoint(-0.4, -0.4), MyPoint(1.4, 1.4)),
        "Clips exactly at two corners",
    )

    fig
end

end

Bubbles.illustrate_masking()

# Problem definition
setup = Bubbles.lidsetup()

psolver = Bubbles.NS.default_psolver(setup)
u = Bubbles.NS.velocityfield(setup, (dim, x, y) -> zero(x));
x = Bubbles.bubble()

# Solve
(; u, x) = Bubbles.solveandplot(u, x, setup, psolver)

# Compute integral of surface tension (it should be zero)
false && let
    npoint = length(x)
    s = similar(x)
    Bubbles.surfacetension!(s, x)
    sum(1:npoint) do i
        p = x[i]
        q = x[mod1(i + 1, npoint)]
        t = s[i]
        # dx = abs(q[1] - p[1])
        # dy = abs(q[2] - p[2])
        dx = 1
        dy = 1
        return Bubbles.MyPoint(t[1] * dx, t[2] * dy)
    end
end
