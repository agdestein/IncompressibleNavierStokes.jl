using TestItemRunner

# # Only run tests from this test dir, and not from other packages in monorepo
# @run_package_tests filter = t -> occursin(@__DIR__, t.filename)

# @run_package_tests

# # Or you can run only specific tests using the following
# function myfilter(t)
#     return endswith(t.filename, "enzyme_integration.jl") || endswith(t.filename, "chainrules_enzyme.jl")
# end
# @run_package_tests filter = myfilter
@run_package_tests filter = t -> endswith(t.filename, "operators.jl")
