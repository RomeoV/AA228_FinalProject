import FLoops: @floop
import Random: seed!
import DroneSurveillance: ACTION_DIRS
import POMDPTools.POMDPDistributions: weighted_iterator
# function eval_problem(nx::Int, agent_strategy::DSAgentStrategy, transition_model::DSTransitionModel; seed_val=rand(Int))
function eval_problem(nx::Int, agent_strategy_p::Float64, transition_model::Any; seed_val=rand(UInt))
    seed!(seed_val)
    P = make_P()
    P.mdp.size = (nx, nx)
    P.mdp.agent_strategy = DSAgentStrat(agent_strategy_p)
    P.mdp.transition_model = DSPerfectModel()

    # policy = RandomPolicy(mdp)
    policy = RolloutLookahead(P, RandomPolicy(P.mdp), 2)
    initial_state = DSState([1, 1], rand(3:nx, 2))
    policy_value = value_iteration(P.mdp, policy; trace_state=initial_state)[initial_state]
    return policy_value
end

function value_iteration(mdp::DroneSurveillanceMDP, policy::Policy;
                         trace_state::Union{Nothing, DSState}=nothing)
    nx, ny = mdp.size
    γ = mdp.discount_factor
    nonterminal_states = [DSState([qx, qy], [ax, ay])
                          for qx in 1:nx,
                              qy in 1:ny,
                              ax in 1:nx,
                              ay in 1:ny][:]  # note that we flatten the array in the end

    U = Dict(s=>rand() for s in nonterminal_states)
    U[mdp.terminal_state] = reward(mdp, mdp.terminal_state, rand(ACTION_DIRS))
    for i in 1:100
        # I benchmarked these (cache misses?) but they're about the same.
        # So we use the Gauss-Seidl version, which should converge faster.
        U_ = U  # Alternative: U_ = copy(U)
        @floop for s in nonterminal_states
            U[s] = let a = policy(s),
                       r = reward(mdp, s, a),
                       T_probs = transition(mdp, s, a, mdp.agent_strategy, DSPerfectModel()),
                       T_probs_iter = weighted_iterator(T_probs)
                r + γ * sum(p*U_[s_] for (s_, p) in T_probs_iter)
            end
        end
        !isnothing(trace_state) && @info U[trace_state]
    end
    return U
end
