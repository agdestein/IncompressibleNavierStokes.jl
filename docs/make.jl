# Show number of threads on GitHub Actions
@info "" Threads.nthreads()

# Get access to example dependencies
push!(LOAD_PATH, joinpath(@__DIR__, "..", "examples"))

using IncompressibleNavierStokes
using NeuralClosure
using Literate
using Documenter
using DocumenterCitations
using DocumenterVitepress

DocMeta.setdocmeta!(
    IncompressibleNavierStokes,
    :DocTestSetup,
    :(using IncompressibleNavierStokes);
    recursive = true,
)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

makemarkdown(inputfile, outputdir; run) =
    if run
        # With code execution blocks
        Literate.markdown(inputfile, outputdir)
    else
        # Turn off code execution.
        # Note: Literate has a `documenter = false` option, but this would also remove
        # the "Edit on GitHub" button at the top, therefore we disable the `@example`-blocks
        # manually
        Literate.markdown(
            inputfile,
            outputdir;
            preprocess = content ->
                "# *Note: Output is not generated for this example (to save resources on GitHub).*\n\n" *
                content,
            postprocess = content -> replace(content, r"@example.*" => "julia"),
        )
    end

# Generate examples
examples = [
    "Simple flows" => [
        (; run = true, name = "DecayingTurbulence2D", title = "Decaying Turbulunce (2D)"),
        (; run = false, name = "DecayingTurbulence3D", title = "Decaying Turbulunce (3D)"),
        (; run = true, name = "TaylorGreenVortex2D", title = "Taylor-Green Vortex (2D)"),
        (; run = false, name = "TaylorGreenVortex3D", title = "Taylor-Green Vortex (3D)"),
        (; run = false, name = "ShearLayer2D", title = "Shear Layer (2D)"),
        (; run = false, name = "PlaneJets2D", title = "Plane jets (2D)"),
    ],
    "Mixed boundary conditions" => [
        (; run = true, name = "Actuator2D", title = "Actuator (2D)"),
        (; run = false, name = "Actuator3D", title = "Actuator (3D)"),
        (; run = false, name = "BackwardFacingStep2D", title = "Backward Facing Step (2D)"),
        (; run = false, name = "BackwardFacingStep3D", title = "Backward Facing Step (3D)"),
        (; run = false, name = "LidDrivenCavity2D", title = "Lid-Driven Cavity (2D)"),
        (; run = false, name = "LidDrivenCavity3D", title = "Lid-Driven Cavity (3D)"),
        (; run = false, name = "MultiActuator", title = "Multiple actuators (2D)"),
        (; run = false, name = "PlanarMixing2D", title = "Planar Mixing (2D)"),
    ],
    "With temperature field" => [
        (; run = true, name = "RayleighBenard2D", title = "Rayleigh-Bénard (2D)"),
        (; run = false, name = "RayleighBenard3D", title = "Rayleigh-Bénard (3D)"),
        (; run = true, name = "RayleighTaylor2D", title = "Rayleigh-Taylor (2D)"),
        (; run = false, name = "RayleighTaylor3D", title = "Rayleigh-Taylor (3D)"),
    ],
]

# Convert scripts to executable markdown files
output = "examples/generated"
outputdir = joinpath(@__DIR__, "src", output)
## rm(outputdir; recursive = true)
for e ∈ examples, e ∈ e[2]
    inputfile = joinpath(@__DIR__, "..", "examples", e.name * ".jl")
    makemarkdown(inputfile, outputdir; e.run)
end

example_pages = map(
    topic -> topic[1] => [e.title => joinpath(output, e.name * ".md") for e ∈ topic[2]],
    examples,
)

makedocs(;
    # draft = true,
    # clean = false,
    modules = [IncompressibleNavierStokes, NeuralClosure],
    plugins = [bib],
    authors = "Syver Døving Agdestein, Benjamin Sanderse, and contributors",
    repo = Remotes.GitHub("agdestein", "IncompressibleNavierStokes.jl"),
    sitename = "IncompressibleNavierStokes.jl",
    # format = Documenter.HTML(;
    #     prettyurls = get(ENV, "CI", "false") == "true",
    #     canonical = "https://agdestein.github.io/IncompressibleNavierStokes.jl",
    #     assets = String[],
    # ),
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/agdestein/IncompressibleNavierStokes.jl",
        devurl = "dev",
    ),
    pagesonly = true,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Examples" => vcat("Overview" => "examples/overview.md", example_pages),
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

# Only deploy docs on CI
get(ENV, "CI", "false") == "true" && deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    target = "build",
    devbranch = "main",
    push_preview = true,
)
