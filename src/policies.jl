struct RolloutLookahead 
    𝒫 # problem 
    π # rollout policy 
    d # depth 
end 
randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)  # <----- here we need the deterministic simulation

function rollout(𝒫, s, π, d)
    ret = 0.0 
    for t in 1:d
        a = π(s)
        s, r = randstep(𝒫, s, a)
        ret += 𝒫.γ^(t-1) * r
    end
    return ret
end 
    
function (π::RolloutLookahead)(s) 
    U(s) = rollout(π.𝒫, s, π.π, π.d) 
    return greedy(π.𝒫, U, s).a 
end

function greedy(𝒫::MDP, U, s) 
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u) 
end

struct FakeDroneSurveillanceMDP <: DroneSurveillance
    transition_params
end

function transition(mdp::FakeDroneSurveillanceMDP, s, a)
    if isterminal(mdp, s) || s.quad == s.agent
        return Deterministic(mdp.terminal_state) # the function is not type stable, returns either Deterministic or SparseCat
    end

    # move quad
    # if it would move out of bounds, just stay in place
    actor_inbounds(actor_state) = (0 < actor_state[1] <= mdp.size[1]) && (0 < actor_state[2] <= mdp.size[2])
    new_quad = actor_inbounds(s.quad + ACTION_DIRS[a]) ? s.quad + ACTION_DIRS[a] : s.quad

    # move agent
    delta_state = s.agent - s.drone
    relative_new_state_probabilities = pred(model, delta_state, mdp.transition_params)
    agent_delta = sample(1:5, Weights(relative_new_state_probabilities)) |> class_to_position_delta
    new_agent_position = agent_inbounds(s.agent + agent_delta) ? s.agent+agent_delta : s.agent

    return Deterministic(DSState(new_agent_position, new_quad))
end

function lookahead(mdp::FakeDroneSurveillanceMDP, U, s, a)
    # here we can use the probabilistic future states as predicted by our model
    #                          vvv
    return R(mdp, s,a) + γ*sum(T(mdp, s,a,s′)*U(s′) for s′ in mdp) 
end