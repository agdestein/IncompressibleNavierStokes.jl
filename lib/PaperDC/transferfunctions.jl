using GLMakie
using CairoMakie

palette = (; color = ["#3366cc", "#cc0000", "#669900", "#ffcc00"])

plotdir = joinpath(@__DIR__, "output", "transferfunctions")
ispath(plotdir) || mkpath(plotdir)

G(k, Δ) = sinpi(k * Δ / 2) / (π * k * Δ / 2)

n = 6
# k = logrange(1, 2^n, n + 1)
k = LinRange(0, 2^n, 200)

Δ = 1 / 2^(n + 1)
g = @. G(k, Δ)
g[1] = 1 # Override division by zero

kx = reshape(k, :)
ky = reshape(k, 1, :)
kz = reshape(k, 1, 1, :)
kdiag = @. sqrt(2) * k
kdiagdiag = @. sqrt(3) * k
idiag = findfirst(kdiag .> k[end])
idiagdiag = findfirst(kdiagdiag .> k[end])
kdiag = kdiag[1:idiag]
kdiagdiag = kdiagdiag[1:idiagdiag]
gx = reshape(g, :)
gy = reshape(g, 1, :)
gz = reshape(g, 1, 1, :)

VA2D = @. gx * gy
FAx2D = @. one(gx) * gy
FAy2D = @. gx * one(gy)
VA2D_diag = diag(VA2D)[1:idiag]
FAx2D_diag = map(i -> FAx2D[i, i], 1:idiag)FAy2D_diag = map(i -> FAy2D[i, i], 1:idiag)
VA3D = @. gx * gy * gz
FAx3D = @. one(gx) * gy * gz
FAx3D_diag = map(i -> FAx3D[i, i, 1], 1:idiag)
FAx3D_diagdiag = map(i -> FAx3D[i, i, i], 1:idiagdiag)

with_theme(; palette) do
    fig = Figure(; size = (1300, 700))

    Label(
        fig[0, 1:3],
        "Transfer functions";
        valign = :bottom,
        font = :bold,
    )

    ax = Axis3(
        fig[1, 1];
        xlabel = "k₁",
        ylabel = "k₂",
        zlabel = "Damping",
        azimuth = π / 4,
        title = "G and G¹ (2D)",
    )
    surface!(ax, k, k, VA2D)
    surface!(ax, k, k, FAx2D; alpha = 0.5)

    ax = Axis3(
        fig[1, 2];
        xlabel = "k₁",
        ylabel = "k₂",
        zlabel = "Damping",
        azimuth = π / 4,
        title = "G and G² (2D)",
    )
    surface!(ax, k, k, VA2D)
    surface!(ax, k, k, FAy2D; alpha = 0.5)

    isovalue = 0.95
    kwargs = (; algorithm = :iso, isorange = 0.002, isovalue)
    ax = Axis3(
        fig[1, 3];
        xlabel = "k₁",
        ylabel = "k₂",
        zlabel = "k₃",
        azimuth = π / 4,
        title = "Isocontours at $isovalue for G and G¹ (3D)",
    )
    volume!(ax, k, k, k, VA3D; kwargs...)
    volume!(ax, k, k, k, FAx3D; kwargs..., alpha = 0.5)
    fig

    ax = Axis(
        fig[2, 1];
        # xscale = log2,
        # yscale = log2,
        xlabel = "|| k ||",
        title = "Sections of G and G¹ (2D)",
    )
    lines!(ax, k, VA2D[1, :]; color = Cycled(2), label = "G")
    lines!(ax, k, FAx2D[:, 1]; color = Cycled(1), linestyle = :solid, label = "G¹, k₂ = 0")
    lines!(
        ax,
        kdiag,
        FAx2D_diag;
        color = Cycled(1),
        linestyle = :dash,
        label = "G¹, k₁ = k₂",
    )
    lines!(
        ax,
        k,
        FAx2D[1, :];
        color = Cycled(1),
        linewidth = 3,
        linestyle = :dot,
        label = "G¹, k₁ = 0",
    )
    axislegend(ax; position = :lb)

    ax = Axis(
        fig[2, 2];
        # xscale = log2,
        # yscale = log2,
        xlabel = "|| k ||",
        title = "Sections of G and G² (2D)",
    )
    lines!(ax, k, VA2D[1, :]; color = Cycled(2), label = "G")
    lines!(ax, k, FAy2D[1, :]; color = Cycled(1), linestyle = :solid, label = "G², k₁ = 0")
    lines!(
        ax,
        kdiag,
        FAy2D_diag;
        color = Cycled(1),
        linestyle = :dash,
        label = "G², k₁ = k₂",
    )
    lines!(
        ax,
        k,
        FAy2D[:, 1];
        color = Cycled(1),
        linewidth = 3,
        linestyle = :dot,
        label = "G², k₂ = 0",
    )
    axislegend(ax; position = :lb)

    ax = Axis(
        fig[2, 3];
        # xscale = log2,
        # yscale = log2,
        xlabel = "|| k ||",
        title = "Sections of G and G¹ (3D)",
    )
    lines!(ax, k, VA3D[:, 1, 1]; color = Cycled(2), label = "G")
    lines!(
        ax,
        k,
        FAx3D[:, 1, 1];
        color = Cycled(1),
        linestyle = :solid,
        label = "G¹, k₂ = k₃ = 0",
    )
    lines!(
        ax,
        kdiag,
        FAx3D_diag;
        color = Cycled(1),
        linestyle = :dash,
        label = "G¹, k₁ = k₂, k₃ = 0",
    )
    lines!(
        ax,
        kdiagdiag,
        FAx3D_diagdiag;
        color = Cycled(1),
        linestyle = :dashdotdot,
        label = "G¹, k₁ = k₂ = k₃",
    )
    lines!(
        ax,
        k,
        FAx3D[1, :, 1];
        color = Cycled(1),
        linewidth = 3,
        linestyle = :dot,
        label = "G¹, k₁ = 0",
    )
    axislegend(ax; position = :lb)

    save("$plotdir/transferfunctions.png", fig)

    fig
end

GLMakie.activate!()

let
    isovalue = 0.95
    kwargs = (; algorithm = :iso, isorange = 0.002, isovalue)
    fig = Figure()
    ax = Axis3(
        fig[1, 1];
        xlabel = "kx",
        ylabel = "ky",
        zlabel = "kz",
        azimuth = π / 4,
        title = "Isocontours at $isovalue for VA and FAx (3D)",
    )
    volume!(ax, k, k, k, VA3D; kwargs...)
    volume!(ax, k, k, k, FAx3D; kwargs..., alpha = 0.5)
    fig
end
