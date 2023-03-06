import POMDPs: MDP, Policy, action, transition, reward
import POMDPTools: weighted_iterator
import DroneSurveillance
import DroneSurveillance: DroneSurveillanceMDP, PerfectCam, DSState

struct MDPProblem
    mdp :: MDP
    γ :: Float64 # discount factor 
    S # state space
    A # action space
end

make_P() = MDPProblem(
    DroneSurveillanceMDP{PerfectCam}(),
    0.9,
    [],  # <- put an implicit definition of all the states here?
    DroneSurveillance.ACTION_DIRS
)

struct RolloutLookahead <: Policy
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
    T_probs = weighted_iterator(transition(P.mdp, s, a))
    return reward(P.mdp, s, a) + P.γ*sum(p*U(s) for (s, p) in T_probs)
end
