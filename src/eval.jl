import Random: seed!
import DroneSurveillance: ACTION_DIRS
import POMDPTools.POMDPDistributions: weighted_iterator
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

function value_iteration(mdp::DroneSurveillanceMDP, policy::Policy)
    nx, ny = mdp.size
    γ = mdp.discount_factor
    states = [DSState([qx, qy], [ax, ay]) for qx in 1:nx, qy in 1:ny, ax in 1:nx, ay in 1:ny][:]
    push!(states, mdp.terminal_state)

    U = Dict(s=>rand() for s in states)
    for _ in 1:100
        for s in states
            if isterminal(mdp, s)
                U[s] = reward(mdp, s, rand(ACTION_DIRS))
            else
                U_ = Dict(
                    a => (reward(mdp, s, a) + γ * sum(p*U[s_] for (s_, p) in weighted_iterator(transition(mdp, s, a))))
                    for a in ACTION_DIRS
                )
                U[s] = maximum(values(U_))
            end
        end
    end
    return U
end


"""
function run_experiment()
    # task 1
    mdp = DroneSurveillanceMDP{PerfectCam}();
    history = make_history(mdp)
    df = prep_history(history, mdp)
    CSV.write("/tmp/data.csv", df)

    df = CSV.read("/tmp/data.csv", DataFrame)
    df.delta_state_x = df.ax - df.dx
    df.delta_state_y = df.ay - df.dy
    df.label = position_delta_to_class.(df.ax_ - df.ax, df.ay_ - df.ay)

    model = MLJLinearModels.MultinomialRegression(0.1)
    fitresult = MLJLinearModels.fit(model, Float64.(hcat(df.delta_state_x, df.delta_state_y)), df.label)

    # task 4
    initial_state = DSState([1, 1], [5, 1])
    policy = RandomPolicy(mdp)
    policy_value = evaluate(mdp, policy)(initial_state)

    # task 5
    policy_values = [evaluate(mdp, policy)(initial_state) 
                    for _ in 1:1000]
    UnicodePlots.boxplot(policy_values)
end
"""