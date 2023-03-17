using DroneSurveillance
import DroneSurveillance: DSLinModel, DSLinCalModel
import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import MLJLinearModels: MultinomialClassifier, glr, fit
import StatsBase: mean

function make_uniform_belief(mdp)
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
    return b0
end

function create_linear_transition_model(mdp::MDP;
                                        dry=false)::DSLinModel
    nx, ny = mdp.size
    b0 = make_uniform_belief(mdp)
    history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(b0)))
                    for _ in 1:(dry ? 10 : 1000) ]...);

    ξs, Δxs, Δys = begin
        ξs, Δxs, Δys = process_row.(history)
        ξs = vcat(ξs...);
        Δxs = vcat(Δxs...) .+ (nx+1)
        Δys = vcat(Δys...) .+ (ny+1)
        ξs, Δxs, Δys
    end


    classifier_x = glr(MultinomialClassifier(), 2*nx+1)
    classifier_y = glr(MultinomialClassifier(), 2*ny+1)

    θ_Δx = fit(classifier_x, ξs, Δxs) |> x->reshape(x, 4+1, :)'
    θ_Δy = fit(classifier_y, ξs, Δys) |> x->reshape(x, 4+1, :)'

    T_model = DSLinModel(θ_Δx, θ_Δy)
    return T_model
end

function create_temp_calibrated_transition_model(mdp::MDP, mdp_calib::MDP;
                                                 dry=false, n_calib=(dry ? 10 : 100))::DSLinCalModel
    nx, ny = mdp.size
    b0 = make_uniform_belief(mdp)
    lin_model = create_linear_transition_model(mdp; dry=dry)
    calib_history = vcat([collect(simulate(HistoryRecorder(), mdp_calib, RandomPolicy(mdp_calib), rand(b0)))
                          for _ in 1:n_calib ]...);

    # Run optimization to find the best parameter T
    Ts = LinRange(0.1, 3.0, 20)
    measure_calibration_error_given_T(T) = begin
        calibrated_model = DSLinCalModel(lin_model, T)
        calibration_results = measure_calibration(calibrated_model, calib_history) |> values
        return mean(calibration_results)
    end
    _, T_idx = findmin(measure_calibration_error_given_T, Ts)
    T_best = Ts[I_idx]

    calibrated_model = DSLinCalModel(lin_model, T_best)
    return calibrated_model
end

"Take one 'row', aka trajectory step, and yield the state and prediction outputs"
function process_row((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = s.agent.x - s.quad.x, s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y]'
    return ξ, Δx, Δy
end
