import POMDPs: MDP, Policy, action, transition
using POMDPs
using DroneSurveillance
import DroneSurveillance
import DroneSurveillance: DroneSurveillanceMDP, PerfectCam, DSState

struct MDPProblem
    mdp :: MDP
    γ :: Float64 # discount factor 
    P # state space
    A # action space
    # these functions are defined through multiple dispatch on the type
    # T # transition function
    # R # reward function
    # TR # sample transition and reward
end

make_P() = MDPProblem(
    DroneSurveillanceMDP{PerfectCam}(),
    0.9,
    [],  # <- put an implicit definition of all the states here?
    DroneSurveillance.ACTION_DIRS
)

struct RolloutLookahead <: POMDPs.Policy
    P # problem
    π_inner # rollout policy
    d # depth
end 

action(rollout_obj::RolloutLookahead, s) = rollout_obj(s)

function rollout(P::MDPProblem, s::DSState, π::Policy, d::Int)
    ret = 0.0
    for t in 1:d
        a::DSPos   = action(π, s)
        s::DSState = rand(transition(P.mdp, s, a))
        r::Float64 = reward(P.mdp, s, a)
        ret += P.γ^(t-1) * r
    end
    return ret
end 
    
function (π_rollout::RolloutLookahead)(s) 
    U(s) = rollout(π_rollout.P, s, π_rollout.π_inner, π_rollout.d)
    return greedy(π_rollout.P, U, s).a
end

function greedy(P::MDPProblem, U, s)
    u, a_idx = findmax(a->lookahead(P, U, s, a), P.A)
    return (a=P.A[a_idx], u=u)
end

function lookahead(P::MDPProblem, U::Function, s::DSState, a::DSPos)
    # here we can use the probabilistic future states as predicted by our model
    #                          vvv
    T_probs = weighted_iterator(transition(P.mdp, s, a))
    return reward(P.mdp, s, a) + P.γ*sum(p*U(s) for (s, p) in T_probs)
end

# struct FakeDroneSurveillanceMDP <: DroneSurveillance
#     transition_params
# end

# function transition(mdp::FakeDroneSurveillanceMDP, s, a)
#     if isterminal(mdp, s) || s.quad == s.agent
#         return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
#     end

#     # move quad
#     # if it would move out of bounds, just stay in place
#     actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
#     new_quad = actor_inbounds(s.quad + ACTION_DIRS[a]) ? s.quad + ACTION_DIRS[a] : s.quad

#     # move agent
#     delta_state = s.agent - s.drone
#     relative_new_state_probabilities = pred(model, delta_state, mdp.transition_params)
#     agent_delta = sample(1:5, Weights(relative_new_state_probabilities)) |> class_to_position_delta
#     new_agent_position = agent_inbounds(s.agent + agent_delta) ? s.agent+agent_delta : s.agent

#     return Deterministic(DSState(new_agent_position, new_quad))
# end

# function lookahead(mdp::FakeDroneSurveillanceMDP, U, s, a)
#     # here we can use the probabilistic future states as predicted by our model
#     #                          vvv
#     return R(mdp, s,a) + γ*sum(T(mdp, s,a,s′)*U(s′) for s′ in mdp) 
# end
