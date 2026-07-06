using TestItemRunner

# Only run test items from this test dir, and not from other packages/checkouts
# that happen to be discoverable from the environment.
@run_package_tests filter = t -> occursin(@__DIR__, t.filename)
