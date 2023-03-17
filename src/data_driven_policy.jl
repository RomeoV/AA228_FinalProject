using DroneSurveillance
import DroneSurveillance: DSLinModel, DSLinCalModel
import DroneSurveillance: make_uniform_belief
import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate
import LinearAlgebra: normalize!
import MLJLinearModels: MultinomialClassifier, glr, fit
import StatsBase: mean
import Unzip: unzip

### BASIC LINEAR MODELS ###
function create_linear_transition_model(mdp::MDP;
                                        dry=false)::DSLinModel
    nx, ny = mdp.size
    b0 = make_uniform_belief(mdp)
    history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp), rand(b0)))
                    for _ in 1:(dry ? 10 : 1000) ]...);

    ξs, Δxs, Δys = begin
        ξs, Δxs, Δys = unzip(process_row.([mdp], history))
        ξs = vcat(ξs...);
        Δxs = vcat(Δxs...)
        Δys = vcat(Δys...)
        ξs, Δxs, Δys
    end

    Δ_states = [(Δx, Δy) for Δx in -nx:nx,
                            Δy in -ny:ny][:]
    push!(Δ_states, -2 .* mdp.size)  # this is the "code" for moving to the terminal state
    label(Δx, Δy) = findfirst(==((Δx, Δy)), Δ_states)

    classifier = glr(MultinomialClassifier(), length(Δ_states))
    Δs = label.(Δxs, Δys)
    @assert !any(isnothing, Δs) let idx=findfirst(nothing, Δ_states); "$((Δxs[idx], Δys[idx]))" end

    θ = fit(classifier, ξs, Δs) |> x->reshape(x, 4+1, :)'

    T_model = DSLinModel(θ, mdp.size)
    return T_model
end


### TEMPERATURE SCALED MODELS ###
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
    @debug T_best

    calibrated_model = DSLinCalModel(lin_model, T_best)
    return calibrated_model
end
create_temp_calibrated_transition_model(mdp::MDP; dry=false, n_calib=(dry ? 10 : 100)) =
    create_temp_calibrated_transition_model(mdp, mdp; dry=dry, n_calib=n_calib)


### CONFORMALIZED MODELS ###
function create_conformalized_transition_model(mdp_base, mdp_calib; dry=false, n_calib=(dry ? 10 : 100))
    T_model = create_linear_transition_model(mdp_base; dry=dry)
    λs::Array = 0.1:0.1:0.9; (!dry && append!(λs, [0.99]))
    λs_hat = conformalize_λs(mdp_calib, T_model, n_calib, λs)
    conf_model = DSConformalizedModel(T_model,
                                      Dict(zip(λs, λs_hat_Δx)),  # λ̂ for Δx
                                      Dict(zip(λs, λs_hat_Δy)))  # λ̂ for Δy
    return conf_model
end
create_conformalized_transition_model(mdp; dry=false, n_calib=(dry ? 10 : 100)) =
    create_conformalized_transition_model(mdp, mdp; dry=dry)

### UTIL FUNCTIONS ###

"Take one 'row', aka trajectory step, and yield the state and prediction outputs"
function process_row(mdp, (s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = s.agent.x - s.quad.x, s.agent.y - s.quad.y
    Δx_next, Δy_next = if sp.quad != mdp.terminal_state
        sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    else
        -2 .* mdp.size
    end
    ξ = [Δx, Δy, a.x, a.y]'
    return ξ, Δx_next, Δy_next
end
