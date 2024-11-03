using TestItemRunner

# @testitem "Time steppers" begin include("timesteppers.jl") end

# Only run tests from this test dir, and not from other packages in monorepo
@run_package_tests filter = t -> occursin(@__DIR__, t.filename)
