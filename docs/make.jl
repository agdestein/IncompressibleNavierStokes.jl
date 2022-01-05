using IncompressibleNavierStokes
using Documenter

DocMeta.setdocmeta!(
    IncompressibleNavierStokes,
    :DocTestSetup,
    :(using IncompressibleNavierStokes);
    recursive = true
)

makedocs(;
    modules = [IncompressibleNavierStokes],
    authors = "Syver Døving Agdestein <syverda@icloud.com> and contributors",
    repo = "https://github.com/agdestein/IncompressibleNavierStokes.jl/blob/{commit}{path}#{line}",
    sitename = "IncompressibleNavierStokes.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://agdestein.github.io/IncompressibleNavierStokes.jl",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => [
            "Lid-Driven Cavity" => "examples/ldc.md",
            "Backwards Facing Step" => "examples/bfs.md",
            "Taylor-Green Vortex" => "examples/tgv.md",
        ],
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
    devbranch = "main"
)
