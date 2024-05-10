# Show number of threads on GitHub Actions
@info "" Threads.nthreads()

# Get access to example dependencies
push!(LOAD_PATH, joinpath(@__DIR__, "..", "examples"))

using IncompressibleNavierStokes
using NeuralClosure
using Literate
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(
    IncompressibleNavierStokes,
    :DocTestSetup,
    :(using IncompressibleNavierStokes);
    recursive = true,
)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

# Generate examples
examples = [
    "Tutorial: Lid-Driven Cavity (2D)" => "LidDrivenCavity2D",
    "Convergence: Taylor-Green Vortex (2D)" => "TaylorGreenVortex2D",
    "Unsteady inflow: Actuator (2D)" => "Actuator2D",
    # "Actuator (3D)" => "Actuator3D",
    "Walls: Backward Facing Step (2D)" => "BackwardFacingStep2D",
    # "Backward Facing Step (3D)" => "BackwardFacingStep3D",
    "Decaying Turbulunce (2D)" => "DecayingTurbulence2D",
    # "Decaying Turbulunce (3D)" => "DecayingTurbulence3D",
    # "Lid-Driven Cavity (3D)" => "LidDrivenCavity3D",
    # "Planar Mixing (2D)" => "PlanarMixing2D",
    # "Shear Layer (2D)" => "ShearLayer2D",
    # "Taylor-Green Vortex (3D)" => "TaylorGreenVortex3D",
    "Temperature: Rayleigh-Bénard (2D)" => "RayleighBenard2D",
]

output = "generated"
for e ∈ examples
    e = joinpath(@__DIR__, "..", "examples", "$(e.second).jl")
    o = joinpath(@__DIR__, "src", output)
    Literate.markdown(e, o)
    # Literate.notebook(e, o)
    # Literate.script(e, o)
end

makedocs(;
    modules = [IncompressibleNavierStokes, NeuralClosure],
    plugins = [bib],
    authors = "Syver Døving Agdestein, Benjamin Sanderse, and contributors",
    repo = "https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/{commit}{path}#{line}",
    sitename = "IncompressibleNavierStokes.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://agdestein.github.io/IncompressibleNavierStokes.jl",
        assets = String[],
    ),
    pagesonly = true,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [e.first => joinpath(output, e.second * ".md") for e ∈ examples],
        "Equations" => [
            "Incompressible Navier-Stokes equations" => "equations/ns.md",
            "Spatial discretization" => "equations/spatial.md",
            "Time discretization" => "equations/time.md",
        ],
        "Features" => [
            "Boundary conditions" => "features/bc.md",
            "Pressure solvers" => "features/pressure.md",
            "Floating point precision" => "features/precision.md",
            "GPU Support" => "features/gpu.md",
            "Operators" => "features/operators.md",
            "Temperature equation" => "features/temperature.md",
            "Large eddy simulation" => "features/les.md",
            "Neural closure models" => "features/closure.md",
        ],
        "API Reference" =>
            ["API" => "api/api.md", "Runge-Kutta methods" => "api/tableaux.md"],
        "References" => "references.md",
    ],
)

get(ENV, "CI", "false") == "true" && deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    devbranch = "main",
    push_preview = true,
)
