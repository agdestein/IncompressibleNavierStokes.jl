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
    # "Lid-Driven Cavity" => "LDC",
    "Lid-Driven Cavity" => "LidDrivenCavity2D",
    # "Backward Facing Step" => "BFS",
    # "Taylor-Green Vortex" => "TGV",
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
    authors = "Syver Døving Agdestein <syverda@icloud.com> and contributors",
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
            "Reduced Order Models" => "theory/rom.md",
            "Immersed Boundary Method" => "theory/ibm.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    devbranch = "main",
)
