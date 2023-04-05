import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import AA228_FinalProject: process_row, make_uniform_belief
import DroneSurveillance: predict
import StatsBase: quantile
import Unzip: unzip
import Match: @match
import Integrals: IntegralProblem, QuadGKJL, solve
import IntervalSets: AbstractInterval, Interval, width
import IntervalSets: (..)
import DataStructures: SortedDict
import Base.Order: Ordering

function conformalize_λs(mdp, T_model, history, λs)::Array{<:Real}
    n_calib = length(history)
    dset_s = getfield.(history, :s)
    dset_a = getfield.(history, :a)
    dset_s_ = getfield.(history, :sp)

    pred_Δs = predict.([mdp], [T_model], dset_s, dset_a)

    true_Δxs = [s_.agent.x - s_.quad.x
                for s_ in dset_s_]
    true_Δys = [s_.agent.y - s_.quad.y
                for s_ in dset_s_]

    true_Δs = zip(true_Δxs, true_Δys)

    scores = [begin
                scores = 1 .- pred.probs
                idx = findfirst(==(true_val), pred.vals)  # true value could have been pruned
                (isnothing(idx) ? (1. - 0) : scores[idx])
            end
            for (pred, true_val) in zip(pred_Δs, true_Δs)]

    λs_hat  = [begin
                α = 1 - λ  # error rate
                quantile_val = ceil((n_calib+1)*(1 - α)) / n_calib
                quantile(scores,  quantile_val)
            end
            for λ in λs]
    return λs_hat
end

#### Variant 1
"Currently implements variant 1 from the writeup."
function conformal_expectation(U::AbstractDict{DSState, <:Real}, C_T::SortedDict{<:Real, Set{DSState}})
    # return 0 if prediction set is empty
    mean_(f, set::Set) = (set == Set() ? 0 : mean(f, set))

    λs = keys(C_T)
    w = Dict(λ => 1/length(λs) for λ in λs)
    sum(λ -> w[λ]*mean_(s->U[s], C_T[λ]),
        λs)
end
# set version
function conformal_expectation(g::Function, C_T::AbstractDict{<:Real, <:Interval})
    # return 0 if prediction set is empty
    integrate_(f, interval::Interval) = (interval.left >= interval.right ? 0 :
        solve(IntegralProblem((x, p)->f(x), interval.left, interval.right),
              QuadGKJL()) |> first)

    λs = keys(C_T)
    w = Dict(λ => 1/length(λs) for λ in λs)
    sum(λ -> w[λ]*(integrate_(g, C_T[λ]) / width(C_T[λ])),
        λs)
end

#### Variant 2
# expectation without g
conformal_expectation_2(C_T::Union{SortedDict{<:Real, <:Interval, <:Ordering},
                                   SortedDict{<:Real, <:Set, <:Ordering}}) =
    conformal_expectation_2(identity, C_T)

# set version
function conformal_expectation_2(g::Function, C_T::SortedDict{<:Real, Set{DSState}})
    # return 0 if prediction set is empty
    mean_(g, set::Set) = (set == Set() ? 0 : mean(g, set))

    λs = collect(keys(C_T)); @assert issorted(λs)
    ws = diff(λs)
    λ_pairs = zip(λs[1:end-1], λs[2:end])
    sum(((λ_lo, λ_hi),) -> begin
            X_λ± = setdiff(C_T[λ_hi], C_T[λ_lo])
            (λ_hi-λ_lo) * mean_(g, X_λ±) / 2
        end,
        λ_pairs) * 1/maximum(λs)
end

# 1D interval version
function conformal_expectation_2(g::Function, C_T::SortedDict{<:Real, <:Interval, <:Ordering})
    # return 0 if prediction set is empty
    mean_(f, interval::Interval) = (interval.left >= interval.right ? 0 :
        solve(IntegralProblem((x, p)->f(x), interval.left, interval.right),
              QuadGKJL()) |> first) / width(interval)

    λs = collect(keys(C_T)); @assert issorted(λs)
    ws = diff(λs)
    λ_pairs = zip(λs[1:end-1], λs[2:end])
    sum(((λ_lo, λ_hi),) -> begin
            pred_lo, pred_hi = C_T[λ_lo], C_T[λ_hi];  @assert (pred_hi.left <= pred_lo.left <= pred_lo.right <= pred_hi.right) "$pred_lo ; $pred_hi"
            X_lhs, X_rhs = pred_hi.left..pred_lo.left, pred_lo.right..pred_hi.right
            (λ_hi-λ_lo) * ( mean_(g, X_rhs) + mean_(g, X_lhs) ) / 2
        end,
        λ_pairs) * 1/maximum(λs)
end

#### Variant 3
function conformal_expectation_3(U::AbstractDict{DSState, <:Real}, C_T::SortedDict{<:Real, Set{DSState}, <:Ordering})
    # return 0 if prediction set is empty
    mean_(f, set::Set) = (set == Set() ? 0 : mean(f, set))

    λs = keys(C_T)
    λ_pairs = zip(λs[1:end-1], λs[2:end])
    sum(((λ_lhs, λ_rhs),) -> begin
            w = λ_rhs - λ_lhs
            w * mean(s->U[s], C_T[λ_rhs]) - mean(s->U[s], C_T[λ_lhs])
        end,
        λ_pairs)
end
