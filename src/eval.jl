import FLoops: @floop
import Random: seed!
import DroneSurveillance: ACTION_DIRS, DSAgentStrat, DSPerfectModel
import POMDPTools
import POMDPTools: weighted_iterator
# function eval_problem(nx::Int, agent_strategy::DSAgentStrategy, transition_model::DSTransitionModel; seed_val=rand(Int))
function eval_problem(nx::Int, agent_strategy_p::Real, transition_model::Symbol;
                      seed_val=rand(UInt), verbose=false, dry=false)
    seed!(seed_val)
    P = make_P()
    P.mdp.size = (nx, nx)
    P.mdp.agent_strategy = DSAgentStrat(agent_strategy_p)
    P.mdp.transition_model = DSPerfectModel()  # probably we can remove this from MDP?

    policy = if transition_model == :perfect
        U = value_iteration(P.mdp; dry=dry);
        POMDPTools.FunctionPolicy(s->runtime_policy(P.mdp, U, s));
    elseif transition_model == :linear
        T_model::DSLinModel = create_linear_transition_model(P.mdp)
        P.mdp.transition_model = T_model
        U = value_iteration(P.mdp; dry=dry);
        POMDPTools.FunctionPolicy(s->runtime_policy(P.mdp, U, s));
    elseif transition_model == :random
        RandomPolicy(P.mdp)
    end
    # policy = RolloutLookahead(P, RandomPolicy(P.mdp), 2)
    initial_state = DSState([1, 1], rand(3:nx, 2))
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
                     a -> let r = reward(mdp, s, a),
                              # note that we use the true transition model here!
                              T_probs = DroneSurveillance.transition(mdp, mdp.agent_strategy, DSPerfectModel(), s, a),
                              T_probs_iter = weighted_iterator(T_probs)
                         r + γ * sum(p*U_[s_] for (s_, p) in T_probs_iter)
                     end,
                     ACTION_DIRS)
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end

function runtime_policy(mdp, U, s)
    γ = mdp.discount_factor
    U_a = Dict(
        a => let r = reward(mdp, s, a),
                 # note that we use the true transition model here!
                 T_probs = DroneSurveillance.transition(mdp, mdp.agent_strategy, DSPerfectModel(), s, a),
                 T_probs_iter = weighted_iterator(T_probs)
            r + γ * sum(p*U[s_] for (s_, p) in T_probs_iter)
        end
        for a in ACTION_DIRS
    )
    val, a = findmax(U_a)
    return a
end
