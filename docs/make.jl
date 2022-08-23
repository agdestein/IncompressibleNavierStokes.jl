# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/IncompressibleNavierStokes.jl")
    using .IncompressibleNavierStokes
end

using IncompressibleNavierStokes
using Literate
using Documenter

DocMeta.setdocmeta!(
    IncompressibleNavierStokes,
    :DocTestSetup,
    :(using IncompressibleNavierStokes);
    recursive = true,
)

# Generate examples
examples = [
    "Actuator (2D)"             => "Actuator2D",
    # "Actuator (3D)"             => "Actuator3D",
    "BackwardFacingStep (2D)"   => "BackwardFacingStep2D",
    # "BackwardFacingStep (3D)"   => "BackwardFacingStep3D",
    "Lid-Driven Cavity (2D)"    => "LidDrivenCavity2D",
    # "Lid-Driven Cavity (3D)"    => "LidDrivenCavity3D",
    # "Planar Mixing (2D)"        => "PlanarMixing2D",
    # "ShearLayer (2D)"           => "ShearLayer2D",
    # "Taylor-Green Vortex (2D)"  => "TaylorGreenVortex2D",
    "Taylor-Green Vortex (3D)"  => "TaylorGreenVortex3D",
]

output = "generated"
for e ∈ examples
    e = joinpath(@__DIR__, "..", "examples", "$(e.second).jl")
    o = joinpath(@__DIR__, "src", output)
    Literate.markdown(e, o)
    Literate.notebook(e, o)
    Literate.script(e, o)
end

makedocs(;
    modules = [IncompressibleNavierStokes],
    authors = "Syver Døving Agdestein, Benjamin Sanderse, and contributors",
    repo = "https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/{commit}{path}#{line}",
    sitename = "IncompressibleNavierStokes.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://agdestein.github.io/IncompressibleNavierStokes.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [e.first => joinpath(output, e.second * ".md") for e ∈ examples],
        "Theory" => [
            "Theory" => "theory/theory.md",
            "Operators" => "theory/operators.md",
            # "Reduced Order Models" => "theory/rom.md",
            # "Immersed Boundary Method" => "theory/ibm.md",
        ],
        "API Reference" => [
            "API" => "api/api.md",
            "Time steppers" => "api/time_steppers.md",
            "Full list" => "api/all_functions.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    devbranch = "main",
    push_preview = true,
)
