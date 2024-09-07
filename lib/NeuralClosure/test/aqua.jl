@testitem "Aqua" begin
    using Aqua
    @info "Testing code with Aqua"
    Aqua.test_all(NeuralClosure; ambiguities = false)
end
