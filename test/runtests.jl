using TestItemRunner

# @testitem "Time steppers" begin include("timesteppers.jl") end

# Only run tests from this test dir, and not from other packages in monorepo
@run_package_tests filter = t -> occursin(@__DIR__, t.filename)

## Or you can run only specific tests using the following
#function myfilter(t)
#    return endswith(t.filename, "enzyme_integration.jl") || endswith(t.filename, "chainrules_enzyme.jl")
#end
#@run_package_tests filter = myfilter