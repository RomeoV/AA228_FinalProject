import FLoops: @floop
import Random: seed!
import DroneSurveillance: ACTION_DIRS, DSAgentStrat, DSPerfectModel, PerfectCam
import DroneSurveillance: ACTION_DIRS, DSTransitionModel, DSLinModel, DSLinCalModel, DSConformalizedModel, DSRandomModel
import POMDPs: reward
import POMDPTools
import POMDPTools: weighted_iterator, Deterministic
import Base: product
import StatsBase: mean
import Match: @match

function eval_problem(transition_model::Type{<:DSTransitionModel}, nx::Int, agent_strategy_p::Real;
                      seed_val=rand(UInt), verbose=false, dry=false)
    seed!(seed_val)
    mdp = DroneSurveillanceMDP{PerfectCam}(size=(nx, nx))
    agent_strategy = DSAgentStrat(agent_strategy_p)

    transition_model = @match transition_model begin
        _::Type{DSPerfectModel} =>  DSPerfectModel(agent_strategy)
        _::Type{DSRandomModel} => DSRandomModel(mdp)
        _::Type{DSLinModel}  => create_linear_transition_model(mdp; dry=dry)
        _::Type{DSLinCalModel} => create_temp_calibrated_transition_model(mdp, mdp; dry=dry)
        _::Type{DSConformalizedModel} => create_conformalized_transition_model(mdp, mdp; dry=dry)
        _ => error("unknown transition model")
    end
    @debug "Finished creating model."
    U = value_iteration(mdp, transition_model; dry=dry);
    policy = POMDPTools.FunctionPolicy(s->runtime_policy(mdp, transition_model, U, s))

    initial_state = DSState([1, 1], [ceil(Int, (nx + 1) / 2), ceil(Int, (nx + 1) / 2)])
    U_π = policy_evaluation(mdp, agent_strategy, policy;
                            trace_state=(verbose ? initial_state : nothing),
                            dry=dry)
    policy_value = U_π[initial_state]
    return policy_value
end

function policy_evaluation(mdp::DroneSurveillanceMDP, agent_strategy::DSAgentStrat, policy::Policy;
                           trace_state::Union{Nothing, DSState}=nothing, dry=false)
    T_model = DSPerfectModel(agent_strategy)
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
        U_ = U  # Alternative: U_ = copy(U)
        for s in nonterminal_states
            U[s] = let a = action(policy, s),
                       r = reward(mdp, s, a),
                       # note that we use the true transition model here!
                       T_probs = DroneSurveillance.transition(mdp, T_model, s, a),
                       T_probs_iter = weighted_iterator(T_probs)
                r + γ * sum(p*U_[s_] for (s_, p) in T_probs_iter)
            end
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end


function value_iteration(mdp::DroneSurveillanceMDP, T_model::DSTransitionModel;
                         trace_state::Union{Nothing, DSState}=nothing,
                         dry=false)
    @debug "Staring value iteration (regular model)"
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
        @debug "Loop: $i"
        U_ = U  # Alternative: U_ = copy(U)
        #@floop
        for s in nonterminal_states
            U[s] = maximum(
                     a -> let r = reward(mdp, s, a),
                              T_probs = DroneSurveillance.transition(mdp, T_model, s, a),
                              T_iter = weighted_iterator(T_probs)
                         r + γ * sum(p*U_[s_]
                                     for (s_, p) in T_iter)
                     end,
                     ACTION_DIRS)
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end

function value_iteration(mdp::DroneSurveillanceMDP, T_model::DSConformalizedModel;
                         trace_state::Union{Nothing, DSState}=nothing,
                         dry=false)
    @debug "Staring value iteration (conformalized model)"
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
        U_ = U  # Alternative: U_ = copy(U)
        @floop for s in nonterminal_states
            U[s] = maximum(a->let r = reward(mdp, s, a),
                                  γ = mdp.discount_factor,
                                  Δs_pred = DroneSurveillance.transition(mdp, T_model, s, a);
                               r + γ * conformal_expectation(U_, Δs_pred)
                           end,
                           ACTION_DIRS)
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end

function runtime_policy(mdp, T_model, U, s)
    _, a_idx = findmax(a->let r = reward(mdp, s, a),
                              γ = mdp.discount_factor,
                              T_probs = DroneSurveillance.transition(mdp, T_model, s, a),
                              T_iter  = weighted_iterator(T_probs);
                           r + γ * sum(p * U[s_]
                                       for (s_, p) in T_iter)
                       end,
                       ACTION_DIRS)
    return ACTION_DIRS[a_idx]
end
function runtime_policy(mdp, T_model::DSConformalizedModel, U, s)
    _, a_idx = findmax(a->let r = reward(mdp, s, a),
                              γ = mdp.discount_factor,
                              Δs_pred = DroneSurveillance.transition(mdp, T_model, s, a);
                           r + γ * conformal_expectation(U, Δs_pred)
                       end,
                       ACTION_DIRS)
    return ACTION_DIRS[a_idx]
end
