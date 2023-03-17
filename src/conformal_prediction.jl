import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import AA228_FinalProject: process_row
import DroneSurveillance: predict
import StatsBase: quantile
import Unzip: unzip

function make_uniform_belief(mdp)
    nx, ny = mdp.size
    states = DSState[]
    for ax in 1:nx,
        ay in 1:ny,
        dx in 1:nx,
        dy in 1:ny

        if [dx, dy] != [ax ay] && [dx, dy] != mdp.region_B
            push!(states, DSState([dx, dy], [ax, ay]))
        end
    end
    probs = normalize!(ones(length(states)), 1)
    SparseCat(states, probs)
end

function conformalize_λs(mdp, T_model, n_calib, λs)::Tuple{Array{<:Real}, Array{<:Real}}
    history = vcat([collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(make_uniform_belief(mdp))))
                    for _ in 1:n_calib ]...);
    dset_s = getfield.(history, :s)
    dset_a = getfield.(history, :a)
    dset_s_ = getfield.(history, :sp)

    preds = predict.([T_model], dset_s, dset_a)
    preds_Δx, preds_Δy = unzip(preds)

    true_Δxs = [s_.agent.x - s_.quad.x
                for s_ in dset_s_]
    true_Δys = [s_.agent.y - s_.quad.y
                for s_ in dset_s_]
    λs_hat = []
    for (pred_vals, true_vals) in [(preds_Δx, true_Δxs),
                                   (preds_Δy, true_Δys)]
        scores = [begin
                    scores = 1 .- pred.probs
                    idx = findfirst(==(true_val), pred.vals)
                    (isnothing(idx) ? (1. - 0) : scores[idx])
                end
                for (pred, true_val) in zip(pred_vals, true_vals)]

        λs_hat_ = [begin
                    α = 1 - λ  # error rate
                    quantile_val = ceil((n_calib+1)*(1 - α)) / n_calib
                    quantile(scores,  quantile_val)
                end
                for λ in λs]
        push!(λs_hat, λs_hat_)
    end
    return (λs_hat[1], λs_hat[2])
end

function predict_with_conf_model_test()
    # predict(conf_model, DSState(rand(1:nx, 2), rand(1:nx, 2)), DSPos([0, 1]), 0.3)
    s = DSState(rand(1:10, 2), rand(1:10, 2))
    @show s
    a = DSPos([0, 1])
    # [λ=>predict(conf_model, s, a, λ)
    #  for λ in sort(collect(keys(conf_model.conf_map_Δx)))]
    [λ=>predict(conf_model, s, a, λ)
     for λ in [0.1]]
end
