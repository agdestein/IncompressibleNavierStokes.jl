# LSP indexing solution
# https://github.com/julia-vscode/julia-vscode/issues/800#issuecomment-650085983
if isdefined(@__MODULE__, :LanguageServer)
    include("../src/DifferentiableNavierStokes.jl")
    using .DifferentiableNavierStokes
end

using DifferentiableNavierStokes
using Literate
using Documenter

DocMeta.setdocmeta!(
    DifferentiableNavierStokes,
    :DocTestSetup,
    :(using DifferentiableNavierStokes);
    recursive = true,
)

# Generate examples
examples = [
    "Lid-Driven Cavity (2D)" => "LidDrivenCavity2D",
    # "Lid-Driven Cavity (3D)" => "LidDrivenCavity3D",
    # "Backward Facing Step (2D)" => "BackwardFacingStep2D",
    # "Backward Facing Step (3D)" => "BackwardFacingStep3D",
    # "Taylor-Green Vortex (2D)" => "TaylorGreenVortex2D",
    # "Taylor-Green Vortex (3D)" => "TaylorGreenVortex3D",
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
    modules = [DifferentiableNavierStokes],
    authors = "Syver Døving Agdestein and contributors",
    repo = "https://github.com/agdestein/DifferentiableNavierStokes.jl/blob/{commit}{path}#{line}",
    sitename = "DifferentiableNavierStokes.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://agdestein.github.io/DifferentiableNavierStokes.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [e.first => joinpath(output, e.second * ".md") for e ∈ examples],
        "Theory" => [
            "Theory" => "theory/theory.md",
            "Operators" => "theory/operators.md",
            "Reduced Order Models" => "theory/rom.md",
            "Immersed Boundary Method" => "theory/ibm.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/agdestein/DifferentiableNavierStokes.jl",
    devbranch = "main",
    pushpreview = true,
)
