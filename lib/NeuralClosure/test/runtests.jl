using TestItemRunner
using Logging

# Hide @info output
with_logger(ConsoleLogger(Warn)) do
    @run_package_tests
end
