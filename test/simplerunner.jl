# Including this file makes `@testitem`s executable in the REPL.
# Each testitem is wrapped in a module to avoid name conflicts.

module TestRunner

macro testitem(name, block)
    quote
        @eval module MyTest
        using Main.Revise
        using IncompressibleNavierStokes
        using Test
        @testset $name begin
            $block
        end
        end
        nothing # Suppress output
    end
end

end

using REPL
REPL.activate(TestRunner)
