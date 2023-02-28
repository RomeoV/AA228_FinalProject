import POMDPs: MDP, Policy, action, transition
using POMDPs
using DroneSurveillance
import DroneSurveillance
import DroneSurveillance: DroneSurveillanceMDP, PerfectCam, DSState

struct MDPProblem
    mdp :: MDP
    γ :: Float64 # discount factor 
    𝒮 # state space
    𝒜 # action space 
    # these functions are defined through multiple dispatch on the type
    # T # transition function
    # R # reward function
    # TR # sample transition and reward
end

make_𝒫() = MDPProblem(
    DroneSurveillanceMDP{PerfectCam}(),
    0.9,
    [],  # <- put an implicit definition of all the states here?
    DroneSurveillance.ACTION_DIRS
)

struct RolloutLookahead
    𝒫 # problem
    π_inner # rollout policy
    d # depth
end 
struct RolloutLookahead_  <: POMDPs.Policy
    𝒫 # problem
    π_inner # rollout policy
    d # depth
end 

action(rollout_obj::RolloutLookahead_, s) = rollout_obj(s)

function rollout(𝒫::MDPProblem, s::DSState, π::Policy, d::Int)
    ret = 0.0 
    for t in 1:d
        a = action(π, s)
        s = rand(transition(𝒫.mdp, s, a))
        r = reward(𝒫.mdp, s, a)
        ret += 𝒫.γ^(t-1) * r
    end
    return ret
end 
    
function (π_rollout::RolloutLookahead)(s) 
    U(s) = rollout(π_rollout.𝒫, s, π_rollout.π_inner, π_rollout.d)
    return greedy(π_rollout.𝒫, U, s).a
end
function (π_rollout::RolloutLookahead_)(s) 
    U(s) = rollout(π_rollout.𝒫, s, π_rollout.π_inner, π_rollout.d) 
    return greedy(π_rollout.𝒫, U, s).a 
end

function greedy(𝒫::MDPProblem, U, s) 
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u) 
end

function lookahead(𝒫::MDPProblem, U::Function, s::DSState, a)
    # here we can use the probabilistic future states as predicted by our model
    #                          vvv
    T_probs, T_vals = let T = transition(𝒫.mdp, s, a)
        T.probs, T.vals
    end
    return reward(𝒫.mdp, s,a) + 𝒫.γ*sum(p*U(s) for (p, s) in zip(T_probs, T_vals))
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
