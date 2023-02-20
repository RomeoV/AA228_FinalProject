using POMDPs
using Parameters
using StaticArrays
using POMDPModelTools

# task 1
mdp = DroneSurveillanceMDP{PerfectCam}();

# task 2
history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp)))
                 for _ in 1:10 ]...)


# task 4
initial_state = DSState([1, 1], [5, 1])
policy = RandomPolicy(mdp)
policy_value = evaluate(mdp, policy)(initial_state)

# task 5
using UnicodePlots
policy_values = [evaluate(mdp, policy)(initial_state) 
                 for _ in 1:1000]
UnicodePlots.boxplot(policy_values)