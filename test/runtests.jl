using TestItems
using TestItemRunner

@testitem "Grid" begin include("grid.jl") end
@testitem "Pressure" begin include("psolvers.jl") end
@testitem "Operators" begin include("operators.jl") end
@testitem "Chain rules" begin include("chainrules.jl") end
# @testitem "Time steppers" begin include("timesteppers.jl") end
@testitem "Post process" begin include("postprocess.jl") end
@testitem "Aqua" begin include("aqua.jl") end

# Only run tests from this test dir, and not from other packages in monorepo
@run_package_tests filter = t -> occursin(@__DIR__, t.filename)
