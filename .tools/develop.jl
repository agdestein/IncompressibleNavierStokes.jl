# Develop all environments in the project

using Pkg

envs = (
    ".",
    "lib/NeuralClosure",
    "lib/NeuralClosure/test",
    "lib/PaperDC",
    "lib/SymmetryClosure",
    "lib/SciMLCompat",
    "test",
    "examples",
    "docs",
)

root = joinpath(@__DIR__, "..")

for e in envs
    e = joinpath(root, e)
    Pkg.activate(e)
    Pkg.instantiate()
end
