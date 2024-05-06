push!(LOAD_PATH, "@live-server")

using LiveServer

servedocs(;
    foldername = @__DIR__,
    literate = joinpath(@__DIR__, "..", "examples"),
    skip_dir = joinpath(@__DIR__, "src", "generated"),
)
