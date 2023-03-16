using DroneSurveillance
import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import MLJLinearModels: MultinomialClassifier, glr, fit
import StatsBase: mean

function create_linear_transition_model(mdp::MDP; dry=false, calibrate=false, n_calib=100)::DSLinModel
    # mdp = DroneSurveillanceMDP{PerfectCam}(size=(10, 10), agent_strategy=DSAgentStrat(0.5))
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
    history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(b0)))
                    for _ in 1:(dry ? 10 : 1000) ]...);
    ξs = vcat(process_row.(history)...);

    Δxs = vcat(process_row2.(history)...) .+ (nx+1)
    Δys = vcat(process_row3.(history)...) .+ (ny+1)

    classifier_x = glr(MultinomialClassifier(), 2*nx+1)
    classifier_y = glr(MultinomialClassifier(), 2*ny+1)

    θ_Δx = fit(classifier_x, ξs, Δxs) |> x->reshape(x, 4+1, :)'
    θ_Δy = fit(classifier_y, ξs, Δys) |> x->reshape(x, 4+1, :)'

    T_model = DSLinModel(θ_Δx, θ_Δy)
    # Δx_distr, Δy_distr = DroneSurveillance.predict(T_model, DSState([1, 1], [5, 5]), DSPos([1, 0]))
    return T_model
end

function create_temp_calibrated_transition_model(mdp::MDP, mdp_calib::MDP; n_calib=100, dry=false)::DSLinCalModel
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
    lin_model = create_linear_transition_model(mdp; dry=dry)
    calib_history = vcat([ collect(simulate(HistoryRecorder(), mdp_calib, RandomPolicy(mdp_calib), rand(b0)))
                    for _ in 1:(dry ? 10 : n_calib) ]...);
    # Run optimization to find the best parameter T
    Ts = LinRange(1.0, 3.0, 9)
    measure_calibration_error_given_T(T) = begin
        calibrated_model = DSLinCalModel(lin_model, T)
        calibration_results = measure_calibration(calibrated_model, calib_history) |> values
        return mean(calibration_results)
    end
    _, T_idx = findmin(measure_calibration_error_given_T, Ts)
    calibrated_model = DSLinCalModel(lin_model, Ts[T_idx])
    return calibrated_model
end

# extract predictors, which we call \xi
function process_row((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = s.agent.x - s.quad.x, s.agent.y - s.quad.y
    ξ = [Δx, Δy, a.x, a.y]'
    return ξ
end

# extract Δx
function process_row2((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    return Δx
end
# extract Δy
function process_row3((s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    return Δy
end
