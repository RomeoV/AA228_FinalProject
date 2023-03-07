using DroneSurveillance
import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
mdp = DroneSurveillanceMDP{PerfectCam}(size=(10, 10), agent_strategy=DSAgentStrat(0.5));
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
    probs = normalize!(ones(1:length(states)), 1)
    SparseCat(states, probs)
end
history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(b0)))
                for _ in 1:1000 ]...);

function process_row((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = s.agent.x - s.quad.x, s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y]'
    return ξ
end
ξs = vcat(process_row.(history)...);

function process_row2((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    return Δx
end
function process_row3((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    return Δy
end
Δxs = vcat(process_row2.(history)...) .+ (nx+1)
Δys = vcat(process_row3.(history)...) .+ (ny+1)

θ_Δx = fit(MultinomialRegression(0.0), ξs, Δxs) |> x->reshape(x, 4+1, :)'
θ_Δy = fit(MultinomialRegression(0.0), ξs, Δys) |> x->reshape(x, 4+1, :)'

model_x, model_y = begin
    mclass = MLJLinearModels.MultinomialClassifier()
    MLJLinearModels.glr(mclass, 2*nx+1), MLJLinearModels.glr(mclass, 2*ny+1)
end

# T_model = DSLinModel(θ_Δx, θ_Δy)