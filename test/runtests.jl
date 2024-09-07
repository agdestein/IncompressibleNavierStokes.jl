using TestItemRunner
using Logging

# @testitem "Time steppers" begin include("timesteppers.jl") end

# Hide @info output
with_logger(ConsoleLogger(Warn)) do
    # Only run tests from this test dir, and not from other packages in monorepo
    @run_package_tests filter = t -> occursin(@__DIR__, t.filename)
end
