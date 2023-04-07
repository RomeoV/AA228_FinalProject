using AA228_FinalProject
AA = AA228_FinalProject;
using Test
using Distributions
using Random
using IntervalSets
import DataStructures: SortedDict
import Integrals: IntegralProblem, QuadGKJL, solve
import Expectations: expectation
import LegibleLambdas: @λ

function lhs_pred_set(D, p)
    -inf..quantile(D, p/2)
end
function rhs_pred_set(D, p)
    quantile(D, 1-p/2)..inf
end
function centered_pred_set(D, p)::ClosedInterval
    quantile(D, 0.5-p/2)..quantile(D, 0.5+p/2)
end

# some distributions to play with
Ds = (
    Normal(10, 5),
    Normal(2*rand()-1, rand()),
    Beta(0.5, 2.5),
    Beta(2.5, 0.5)
)

gs = (
    identity,
    @λ(x->x^2),
    @λ(x->x-x^2),
    # @λ(x->x^3-x^2),
    # @λ(x->log(abs(x))),
)

@testset "Test centered pred set" begin
    Random.seed!(1)
    for _ in 1:3
        μ = 10*rand()
        σ = abs(10*rand())
        D = Normal(μ, σ)
        λs = 0.05:0.05:0.95
        for λ in λs
            @test mean(centered_pred_set(D, λ)) ≈ μ atol=sqrt(eps())
        end
    end
end

@testset "Conformal Expectation Tests" begin
    Random.seed!(1)

    # TODO: distributions where mode and mean/expectation do not coincide.
    for i in 1:10
        μ = 10*rand()
        σ = abs(10*rand())
        D = Normal(μ, σ)
        λs::Vector{Float64} = 0.05:0.05:0.95
        C_T = SortedDict{Float64, ClosedInterval}(
            λ => centered_pred_set(D, λ)
            for λ in λs
        )
        @test AA.conformal_expectation(identity, C_T) ≈ μ atol=sqrt(eps())
    end
end

@testset "Conformal Expectation 2 Tests" begin
    Random.seed!(1)
    # TODO: distributions where mode and mean/expectation do not coincide.
    for D in Ds, g in gs
        @show D, g
        E_D = expectation(D)
        λs = [LinRange(0//1, 1//1, 31)[1:end-1] ; 0.99 ; 0.999]
        C_T = SortedDict{Float64, ClosedInterval}(
            λ => centered_pred_set(D, λ)
            for λ in λs
        )
        @test AA.conformal_expectation_2(g, C_T) ≈ E_D(g) rtol=1e-2
    end
end


@testset "Expectation with Quantile test" begin
    @testset "Normal case" begin
        for D in Ds
            f(p, params) = quantile(D, p)
            @test solve(IntegralProblem(f, 0, 1), QuadGKJL())[1] ≈ mean(D)
        end
    end
    @testset "Upper minus lower case" begin
        for D in Ds
            f_lo(p, params) = quantile(D, p)
            f_hi(p, params) = quantile(D, p)
            val = solve(IntegralProblem(f_lo, 0., 1/2), QuadGKJL())[1] +
                  solve(IntegralProblem(f_hi, 1/2, 1.), QuadGKJL())[1]
            @test val ≈ mean(D)
        end
    end
    @testset "Upper minus lower case" begin
        for D in Ds
            f_lo(p̃, params) = quantile(D, 1/2-p̃/2)
            f_hi(p̃, params) = quantile(D, 1/2+p̃/2)
            val = solve(IntegralProblem(f_lo, 1., 0.), QuadGKJL())[1] * (-1/2) +
                  solve(IntegralProblem(f_hi, 0/2, 2/2), QuadGKJL())[1] * 1/2
            @test val ≈ mean(D)
        end
    end
    @testset "Upper minus lower case" begin
        for D in Ds
            f_lo(p̃) = quantile(D, 1/2-p̃/2)
            f_hi(p̃) = quantile(D, 1/2+p̃/2)
            val = solve(IntegralProblem((p, _)->(f_lo(p) + f_hi(p))/2, 0., 1.), QuadGKJL())[1]
            @test val ≈ mean(D)
        end
    end
    return
end

include("test_eval.jl")
