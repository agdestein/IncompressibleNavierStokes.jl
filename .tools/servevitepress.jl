# Serve the Vitepress site after building documentation

using Pkg

docs = joinpath(@__DIR__, "..", "docs")

Pkg.activate(docs)

using DocumenterVitepress
DocumenterVitepress.dev_docs(
    joinpath(docs, "build"),
    # md_output_path = joinpath(docs, "build/.documenter"),
)
