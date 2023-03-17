import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import AA228_FinalProject: process_row, make_uniform_belief
import DroneSurveillance: predict
import StatsBase: quantile
import Unzip: unzip

function conformalize_λs(mdp, T_model, n_calib, λs)::Array{<:Real}
    history = vcat([collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(make_uniform_belief(mdp))))
                    for _ in 1:n_calib ]...);
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
