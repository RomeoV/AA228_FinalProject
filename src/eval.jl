import Random: seed!
# function eval_problem(nx::Int, agent_strategy::DSAgentStrategy, transition_model::DSTransitionModel; seed_val=rand(Int))
function eval_problem(nx::Int, agent_strategy::Any, transition_model::Any; seed_val=rand(UInt))
    seed!(seed_val)
    mdp = DroneSurveillanceMDP{PerfectCam}();
    mdp.size = (nx, nx)
    mdp.agent_strategy = DSAgentStrat(0.8)
    mdp.transition_model = DSPerfectModel()

    policy = RandomPolicy(mdp)
    initial_state = DSState([1, 1], rand(2:nx, 2))
    policy_value = evaluate(mdp, policy)(initial_state)
    return policy_value
end
