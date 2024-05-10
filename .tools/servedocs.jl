# Build documentation and rebuild on changes

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"))

push!(LOAD_PATH, "@live-server")
using LiveServer

root = joinpath(@__DIR__, "..")

servedocs(;
    foldername = joinpath(root, "docs"),
    literate = joinpath(root, "examples"),
    skip_dir = joinpath(root, "docs", "src", "generated"),
)
