# Develop all environments in the project

envs = (
    (".", ()),
    ("lib/NeuralClosure", (".",)),
    ("lib/PaperDC", (".", "lib/NeuralClosure")),
    ("lib/SymmetryClosure", (".", "lib/NeuralClosure")),
    ("test", ()),
    ("examples", (".",)),
    ("docs", (".", "lib/NeuralClosure", "examples")),
)

cd(joinpath(@__DIR__, ".."))

for (e, d) in envs
    cd(e) do
        if !isempty(d)
            d = relpath.(d, e)
            d = join(d, " ")
            run(`julia --project=. -e "using Pkg; pkg\"dev $d\""`)
        end
        run(`julia --project=. -e 'using Pkg; Pkg.update()'`)
    end
end
