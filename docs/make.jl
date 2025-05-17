# Look for environment variable triggering local development modifications
localdev = haskey(ENV, "LOCALDEV")

# Get access to example dependencies
# Those are required for running the examples and generating output
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

# Generate examples
examples = [
    # Simple flows
    (true, "examples/DecayingTurbulence2D"),
    (false, "examples/DecayingTurbulence3D"),
    (true, "examples/TaylorGreenVortex2D"),
    (false, "examples/TaylorGreenVortex3D"),
    (false, "examples/Kolmogorov2D"),
    (false, "examples/ShearLayer2D"),
    (false, "examples/PlaneJets2D"),

    # Mixed boundary conditions
    (true, "examples/Actuator2D"),
    (false, "examples/Actuator3D"),
    (false, "examples/BackwardFacingStep2D"),
    (false, "examples/BackwardFacingStep3D"),
    (false, "examples/LidDrivenCavity2D"),
    (false, "examples/LidDrivenCavity3D"),
    (false, "examples/MultiActuator"),
    (false, "examples/PlanarMixing2D"),

    # With temperature field
    (true, "examples/RayleighBenard2D"),
    (false, "examples/RayleighBenard3D"),
    (true, "examples/RayleighTaylor2D"),
    (false, "examples/RayleighTaylor3D"),

    # # Neural closure models
    # (false, "lib/PaperDC/prioranalysis"),
    # (false, "lib/PaperDC/postanalysis"),
    # (false, "lib/SymmetryClosure/symmetryanalysis"),
]

# Convert scripts to executable markdown files
output = "examples/generated"
outputdir = joinpath(@__DIR__, "src", output)
## rm(outputdir; recursive = true)
scriptdir = mktempdir()
for (run, name) in examples
    inputfile = joinpath(@__DIR__, "..", name * ".jl")
    script = Literate.script(inputfile, scriptdir; config = Dict("credit" => false))
    code = strip(read(script, String))
    codepair = "CODE_CONTENT" => code
    replacefunc(pairs...) = content -> replace(content, pairs...)
    codecoda = """
        # ## Copy-pasteable code
        #
        # Below is the full code for this example stripped of comments and output.
        #
        # ```julia
        # CODE_CONTENT
        # ```
    """
    if run
        # With code execution blocks
        Literate.markdown(
            inputfile,
            outputdir;
            preprocess = content -> content * "\n\n" * codecoda,
            postprocess = replacefunc(codepair),
        )
    else
        # Turn off code execution.
        # Note: Literate has a `documenter = false` option, but this would also remove
        # the "Edit on GitHub" button at the top, therefore we disable the `@example`-blocks
        # manually
        Literate.markdown(
            inputfile,
            outputdir;
            preprocess = content ->
                "# *Note: Output is not generated for this example (to save resources on GitHub).*" *
                "\n\n" *
                content *
                "\n\n" *
                codecoda,
            postprocess = replacefunc(r"@example.*" => "julia", codepair),
        )
    end
end

# There is an issue with linking to other MD pages:
# https://github.com/LuxDL/DocumenterVitepress.jl/issues/172
makedocs(;
    modules = [IncompressibleNavierStokes, NeuralClosure],
    plugins = [bib],
    authors = "Syver Døving Agdestein, Benjamin Sanderse, and contributors",
    repo = Remotes.GitHub("agdestein", "IncompressibleNavierStokes.jl"),
    sitename = "IncompressibleNavierStokes.jl",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
        devurl = "dev",
    ),
    pagesonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/agdestein/IncompressibleNavierStokes.jl",
    target = joinpath(@__DIR__, "build"),
    devbranch = "main",
    push_preview = true,
)
