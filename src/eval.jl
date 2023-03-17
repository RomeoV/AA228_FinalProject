import FLoops: @floop
import Random: seed!
import DroneSurveillance: ACTION_DIRS, DSAgentStrat, DSPerfectModel
import DroneSurveillance: ACTION_DIRS, DSTransitionModel, DSLinModel, DSLinCalModel, DSConformalizedModel, DSRandomModel
import POMDPTools
import POMDPTools: weighted_iterator, Deterministic
import Base: product
import StatsBase: mean
import Match: @match
# function eval_problem(nx::Int, agent_strategy::DSAgentStrategy, transition_model::DSTransitionModel; seed_val=rand(Int))
function eval_problem(transition_model::Type{<:DSTransitionModel}, nx::Int, agent_strategy_p::Real;
                      seed_val=rand(UInt), verbose=false, dry=false)
    seed!(seed_val)
    P = make_P()
    P.mdp.size = (nx, nx)
    P.mdp.agent_strategy = DSAgentStrat(agent_strategy_p)

    @match transition_model begin
        DSPerfectModel => begin P.mdp.transition_model = DSPerfectModel() end
        DSLinModel  => begin
            T_model::DSLinModel = create_linear_transition_model(P.mdp)
            P.mdp.transition_model = T_model end
        DSLinCalModel => begin
            T_model_cal::DSLinCalModel = create_temp_calibrated_transition_model(P.mdp, P.mdp)
            P.mdp.transition_model = T_model_cal end
        DSConformalizedModel => begin
            T_model_::DSLinModel = create_linear_transition_model(P.mdp)
            λs::Array = 0.1:0.1:0.9; (!dry && append!(λs, [0.99]))
            λs_hat = conformalize_λs(P.mdp, T_model_, 101, λs)
            conf_model = DSConformalizedModel(T_model_, Dict(zip(λs, λs_hat)))
            P.mdp.transition_model = conf_model end
        DSRandomModel => begin P.mdp.transidion_model = DSRandomModel() end
        _ => error("unknown transition model")
    end
    U = value_iteration(P.mdp; dry=dry);
    policy = POMDPTools.FunctionPolicy(s->runtime_policy(P.mdp, P.mdp.transition_model, U, s))

    initial_state = DSState([1, 1], [ceil(Int, (nx + 1) / 2), ceil(Int, (nx + 1) / 2)])
    U_π = policy_evaluation(P.mdp, policy;
                            trace_state=(verbose ? initial_state : nothing),
                            dry=dry)
    policy_value = U_π[initial_state]
    return policy_value
end

function policy_evaluation(mdp::DroneSurveillanceMDP, policy::Policy;
                           trace_state::Union{Nothing, DSState}=nothing, dry=false)
    nx, ny = mdp.size
    γ = mdp.discount_factor
    nonterminal_states = [DSState([qx, qy], [ax, ay])
                          for qx in 1:nx,
                              qy in 1:ny,
                              ax in 1:nx,
                              ay in 1:ny][:]  # <- we flatten here!

    U = Dict(s=>rand() for s in nonterminal_states)
    U[mdp.terminal_state] = reward(mdp, mdp.terminal_state, rand(ACTION_DIRS))
    for i in 1:(dry ? 5 : 50)
        # I benchmarked these (cache misses?) but they're about the same.
        # So we use the Gauss-Seidl version, which should converge faster.
        U_ = U  # Alternative: U_ = copy(U)
        for s in nonterminal_states
            U[s] = let a = action(policy, s),
                       r = reward(mdp, s, a),
                       # note that we use the true transition model here!
                       T_probs = DroneSurveillance.transition(mdp, mdp.agent_strategy, DSPerfectModel(), s, a),
                       T_probs_iter = weighted_iterator(T_probs)
                r + γ * sum(p*U_[s_] for (s_, p) in T_probs_iter)
            end
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end


function value_iteration(mdp::DroneSurveillanceMDP;
                         trace_state::Union{Nothing, DSState}=nothing,
                         dry=false)
    nx, ny = mdp.size
    γ = mdp.discount_factor
    nonterminal_states = [DSState([qx, qy], [ax, ay])
                          for qx in 1:nx,
                              qy in 1:ny,
                              ax in 1:nx,
                              ay in 1:ny][:]  # note that we flatten the array in the end

    U = Dict(s=>rand() for s in nonterminal_states)
    U[mdp.terminal_state] = reward(mdp, mdp.terminal_state, rand(ACTION_DIRS))
    for i in 1:(dry ? 5 : 50)
        # I benchmarked these (cache misses?) but they're about the same.
        # So we use the Gauss-Seidl version, which should converge faster.
        U_ = U  # Alternative: U_ = copy(U)
        @floop for s in nonterminal_states
            U[s] = maximum(
                     a -> let U_wrapped = wrap_U(mdp, U_, s, a),
                              r = reward(mdp, s, a),
                              # note that we use the true transition model here!
                              T_probs = DroneSurveillance.transition(mdp, mdp.agent_strategy, mdp.transition_model, s, a),
                              T_iter = weighted_iterator(T_probs)
                         r + γ * sum(p*U_wrapped(s_)
                                     for (s_, p) in T_iter)
                     end,
                     ACTION_DIRS)
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end

"Currently implements variant 1 from the writeup."
function conformal_expectation(U::Function, C_T::Dict{<:Real, Set{DSState}})
    # return 0 if prediction set is empty
    mean_(f, set::Set) = @match set begin
        Set() => 0
        _     => mean(f, set)
    end

    # TODO: Implement other variants
    λs = keys(C_T)
    w = Dict(λ => 1/length(λs) for λ in λs)
    sum(λ -> w[λ]*mean_(s->U[s], C_T[λ]),
        λs)
end

function bellmann_eq_conformal(mdp, U_fn, s, a)
    γ = mdp.discount_factor
    r = reward(mdp, s, a)
    retval = DroneSurveillance.transition(mdp,
                                          mdp.agent_strategy,
                                          mdp.transition_model,
                                          s,
                                          a)
    @match retval begin
        Deterministic(r)                    => r  # if we reached the terminal state
        λ_to_Δs::Dict{<:Real, Set{DSState}} =>    # map from λ to prediction set
            r + γ*conformal_expectation(U_fn, λ_to_Δs)
        _                                   => error("unknown transition return type")
    end
end

function value_iteration_conformal(mdp::DroneSurveillanceMDP;
                         trace_state::Union{Nothing, DSState}=nothing,
                         dry=false)
    @assert mdp.transition_model isa DSConformalizedModel
    nx, ny = mdp.size
    γ = mdp.discount_factor
    nonterminal_states = [DSState([qx, qy], [ax, ay])
                          for qx in 1:nx,
                              qy in 1:ny,
                              ax in 1:nx,
                              ay in 1:ny][:]  # note that we flatten the array in the end

    U = Dict(s=>rand() for s in nonterminal_states)
    U[mdp.terminal_state] = reward(mdp, mdp.terminal_state, rand(ACTION_DIRS))

    for _ in 1:(dry ? 5 : 50)
        # I benchmarked these (cache misses?) but they're about the same.
        # So we use the Gauss-Seidl version, which should converge faster.
        U_ = U  # Alternative: U_ = copy(U)
        @floop for s in nonterminal_states
            U[s] = maximum(a->let r = reward(mdp, s, a); γ = mdp.discount_factor,
                                  Δs_pred = transition(mdp, mdp.agent_strategy,
                                                       mdp.transition_model, s, a);
                               r + γ * conformal_expectation(U_, Δs_pred)
                           end,
                           ACTION_DIRS)
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end

function runtime_policy(mdp, T_model, U, s)
    a, _ = findmax(a->let r = reward(mdp, s, a); γ = mdp.discount_factor,
                          T_probs = transition(mdp, mdp.agent_strategy,
                                               mdp.transition_model, s, a),
                          T_iter  = weighted_iterator(T_probs);
                       r + γ * sum(p * U[s_]
                                   for (s_, p) in T_iter)
                   end,
                   ACTION_DIRS)
    return a
end
function runtime_policy(mdp, T_model::DSConformalizedModel, U, s)
    a, _ = findmax(a->let U_wrapped = wrap_U(mdp, U_, s, a),
                          r = reward(mdp, s, a); γ = mdp.discount_factor,
                          Δs_pred = transition(mdp, mdp.agent_strategy,
                                               mdp.transition_model, s, a);
                       r + γ * conformal_expectation(U_wrapped, Δs_pred)
                   end,
                   ACTION_DIRS)
    return a
end

function project_inbounds(mdp, s)
    nx, ny = mdp.size
    qx = clamp(s.quad.x, 1:nx)
    qy = clamp(s.quad.y, 1:ny)
    ax = clamp(s.agent.x, 1:nx)
    ay = clamp(s.agent.y, 1:ny)
    return DSState([qx, qy], [ax, ay])
end

