# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
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

# # Generate examples
# examples = [
#     "Tutorial: Lid-Driven Cavity (2D)" => "LidDrivenCavity2D",
#     "Actuator (2D)" => "Actuator2D",
#     # "Actuator (3D)" => "Actuator3D",
#     "Backward Facing Step (2D)" => "BackwardFacingStep2D",
#     # "Backward Facing Step (3D)" => "BackwardFacingStep3D",
#     "Decaying Turbulunce (2D)" => "DecayingTurbulence2D",
#     # "Decaying Turbulunce (3D)" => "DecayingTurbulence3D",
#     # "Lid-Driven Cavity (3D)" => "LidDrivenCavity3D",
#     "Planar Mixing (2D)" => "PlanarMixing2D",
#     "Shear Layer (2D)" => "ShearLayer2D",
#     # "Taylor-Green Vortex (2D)" => "TaylorGreenVortex2D",
#     # "Taylor-Green Vortex (3D)" => "TaylorGreenVortex3D",
# ]
#
# output = "generated"
# for e ∈ examples
#     e = joinpath(@__DIR__, "..", "examples", "$(e.second).jl")
#     o = joinpath(@__DIR__, "src", output)
#     Literate.markdown(e, o)
#     # Literate.notebook(e, o)
#     # Literate.script(e, o)
# end

output = "generated"
examples = ["Tutorial: Lid-Driven Cavity (2D)" => "LidDrivenCavity2D"]

makedocs(;
    modules = [IncompressibleNavierStokes],
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
            "Large eddy simulation" => "features/les.md",
            "Neural closure models" => "features/closure.md",
        ],
        "API Reference" =>
            ["API" => "api/api.md", "Runge-Kutta methods" => "api/tableaux.md"],
        "References" => "references.md",
    ],
)

deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    devbranch = "main",
    push_preview = true,
)
