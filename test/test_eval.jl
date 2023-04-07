@testset "Eval integration tests" begin
    @test eval_problem(AA.DSLinModel, 5, 0.5) isa Any
    @test eval_problem(AA.DSLinCalModel, 5, 0.5) isa Any
    @test eval_problem(AA.DSConformalizedModel, 5, 0.5) isa Any
end
