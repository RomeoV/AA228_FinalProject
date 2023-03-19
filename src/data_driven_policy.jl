using DroneSurveillance
import DroneSurveillance: DSLinModel, DSLinCalModel
import DroneSurveillance: make_uniform_belief
import POMDPTools: SparseCat, HistoryRecorder, RandomPolicy
import POMDPs: simulate, action, transition, reward
import LinearAlgebra: normalize!
import MLJ
import MLJ: table, coerce, Continuous
import MLJLinearModels: MultinomialClassifier, NewtonCG, LBFGS
import MLJLinearModels.Optim
import StatsBase: mean
import Unzip: unzip

### BASIC LINEAR MODELS ###
function create_linear_transition_model(mdp::MDP;
                                        dry=false)::DSLinModel

    nx, ny = mdp.size
    b0 = make_uniform_belief(mdp)
    sim_one_step(s) = begin
        # Same output as HistoryRecorder would give you.
        s = rand(b0)
        a = action(RandomPolicy(mdp), s)
        s_ = rand(transition(mdp, s, a))
        r = reward(mdp, s, a)
        (; s=s, a=a, sp=s_, reward=r, info=nothing, t=nothing, action_info=nothing)
    end
    history = [sim_one_step(rand(b0))
               for _ in 1:(dry ? 100 : 10000)];
    ξs, Δxs, Δys = begin
        ξs, Δxs, Δys = unzip(process_row.([mdp], history))
        ξs = vcat(ξs...)
        Δxs = vcat(Δxs...)
        Δys = vcat(Δys...)
        (ξs, Δxs, Δys)
    end

    Δ_states = [(Δx, Δy) for Δx in -nx:nx,
                             Δy in -ny:ny][:]
    push!(Δ_states, -2 .* mdp.size)  # this is the "code" for moving to the terminal state
    label(Δx, Δy) = findfirst(==((Δx, Δy)), Δ_states)

    # turn the matrix into a named tuple of (continuous) colums
    ξs_tbl = begin
        col_names = [:Δx, :Δy, :a_x, :a_y]
        tbl = table(ξs; names=col_names)
        coerce(tbl, Dict(n=>Continuous for n in col_names)...)
    end

    # we predict the index of the Δ-state
    Δs_ind = label.(Δxs, Δys)
    @assert !any(isnothing, Δs_ind) let idx=findfirst(nothing, Δ_states); "$((Δxs[idx], Δys[idx]))" end
    Δs_cat = MLJ.categorical(Δs_ind, levels=1:length(Δ_states))

    @debug "Starting to fit linear classifier with $(length(Δ_states)) target classes."
    # solver = LBFGS(; optim_options=Optim.Options(show_trace=true, g_tol=1e-3, ))
    solver = NewtonCG(; optim_options=Optim.Options(show_trace=true, g_tol=1e-3, ))
    # Note: Choice of lambda quite influences the optimization problem.
    # Larger lambda -> faster optimization, but worse results.
    classifier = MultinomialClassifier(lambda=(dry ? 100. : 10.), penalty=:l2;
                                       solver=solver,
                                       scale_penalty_with_samples=false)
    mach = MLJ.machine(classifier, ξs_tbl, Δs_cat)
    MLJ.fit!(mach)
    θ = reshape(mach.fitresult[1], 4+1, :)'
    @debug "Finished fitting linear classifier"

    T_model = DSLinModel(θ)
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
        calibration_results = measure_calibration(mdp, calibrated_model, calib_history) |> values
        return mean(calibration_results)
    end
    _, T_idx = findmin(measure_calibration_error_given_T, Ts)
    T_best = Ts[T_idx]
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
    history = vcat([collect(simulate(HistoryRecorder(), mdp_calib, RandomPolicy(mdp_calib), rand(make_uniform_belief(mdp_calib))))
                    for _ in 1:n_calib ]...);
    λs_hat = conformalize_λs(mdp_calib, T_model, history, λs)
    return DSConformalizedModel(T_model, Dict(zip(λs, λs_hat)))
end
create_conformalized_transition_model(mdp; dry=false, n_calib=(dry ? 10 : 100)) =
    create_conformalized_transition_model(mdp, mdp; dry=dry)

### UTIL FUNCTIONS ###

"Take one 'row', aka trajectory step, and yield the state and prediction outputs"
function process_row(mdp, (s, a, sp, r, info, t, action_info)::NamedTuple)
    Δx, Δy = s.agent.x - s.quad.x, s.agent.y - s.quad.y
    Δx_next, Δy_next = if sp != mdp.terminal_state
        sp.agent.x - sp.quad.x, sp.agent.y - sp.quad.y
    else
        -2 .* mdp.size
    end
    ξ = [Δx, Δy, a.x, a.y]'
    return ξ, Δx_next, Δy_next
end
