import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import AA228_FinalProject: process_row, process_row2
import DroneSurveillance: predict
import Unzip: unzip
import StatsBase: quantile

mdp = DroneSurveillanceMDP{PerfectCam}();
nx, ny = mdp.size
b0 = begin
    states = []
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

T_model = create_linear_transition_model(mdp)

n_calib = 100
history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(b0)))
                for _ in 1:n_calib ]...);
dset_s = getfield.(history, :s)
dset_a = getfield.(history, :a)
dset_s_ = getfield.(history, :sp)

# dset = f(history)
dset_x = process_row.(history)
dset_y = process_row2.(history)
# λs = [0.5, 0.7, 0.9]
λ = 0.7  # coverage level
α = 1 - λ  # error rate

preds = predict.([T_model], dset_s, dset_a)
preds_Δx, preds_Δy = unzip(preds)


true_Δxs = [s_.agent.x - s_.quad.x
            for s_ in dset_s_]
scores_Δx = [begin
                scores = 1 .- pred.probs
                idx = findfirst(==(true_Δx), pred.vals)
                (isnothing(idx) ? (1. - 0) : scores[idx])
            end
            for (pred, true_Δx) in zip(preds_Δx, true_Δxs)]
# scores_Δy = 1 .- preds_Δx.probs
# scores_Δy = 1 .- preds[dset.y]
quantile_val = ceil((n_calib+1)*(1 - α)) / n_calib
λ_hat = quantile(scores_Δx,  quantile_val)

# later
preds_Δx, preds_Δy = predict(T_model, DSState([1, 1], [5, 5]), DSPos([0, 1]))
final_prediction = Set(preds_Δx.vals[preds_Δx.probs .>= (1 - λ_hat)])
