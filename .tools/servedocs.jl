# Build the documentation and rebuild on changes

root = joinpath(@__DIR__, "..")
docs = joinpath(root, "docs")

using Pkg
Pkg.activate(docs)

# Add live server environment
push!(LOAD_PATH, "@LiveServer")

using LiveServer

# Trigger local development modifications
ENV["LOCALDEV"] = true

# Build documentation and rebuild on changes
servedocs(;
    foldername = docs,
    literate = joinpath(root, "examples"),
    skip_dir = joinpath(docs, "src", "examples", "generated"),
)
