# Add environment for package being tested via
# load path for live testing, since the test Project.toml
# is not allowed to depend on the tested package for Pkg.test()
# to work correctly
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Test
using Random
Random.seed!(42)

@testset "enzyme" begin
    @test try
        include("test_enzyme.jl")
        true
    catch e
        println("Error: ", e)
        Base.show_backtrace(stderr, catch_backtrace())
        false
    end
end

@testset "Force" begin
    @test try
        include("test_force.jl")
        true
    catch e
        println("Error: ", e)
        Base.show_backtrace(stderr, catch_backtrace())
        false
    end
end

@testset "SciMLCompat" begin
    @test try
        include("test_SciMLCompat.jl")
        true
    catch e
        println("Error: ", e)
        Base.show_backtrace(stderr, catch_backtrace())
        false
    end
end