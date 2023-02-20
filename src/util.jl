using POMDPs
using Parameters
using StaticArrays
using POMDPModelTools
using POMDPTools
using DroneSurveillance
using DataFrames, CSV
using MLJ
using MLJLinearModels

# task 1
mdp = DroneSurveillanceMDP{PerfectCam}();

# task 2
history = vcat([ collect(simulate(HistoryRecorder(), mdp, RandomPolicy(mdp)))
                 for _ in 1:1000 ]...);

df = begin
    history = filter(row->row.sp != mdp.terminal_state,  history)
    DataFrame(
        Dict(:ax  => Int[h.s.agent[1] for h in history],
            :ay  => Int[h.s.agent[2] for h in history],
            :dx  => Int[h.s.quad[1] for h in history],
            :dy  => Int[h.s.quad[2] for h in history],
            :ax_ => Int[h.sp.agent[1] for h in history],
            :ay_ => Int[h.sp.agent[2] for h in history]));
end
CSV.write("/tmp/data.csv", df)

df = CSV.read("/tmp/data.csv", DataFrame)

function class_to_position_delta(class::Int)
    Dict(
         1 => [-1,  0],
         2 => [ 0, -1],
         3 => [ 0,  0],
         4 => [ 0,  1],
         5 => [ 1,  0]
    )[class]
end

function position_delta_to_class(delta_x::Int, delta_y::Int)
    Dict(
        (-1,  0) => 1,
        ( 0, -1) => 2,
        ( 0,  0) => 3,
        ( 0,  1) => 4,
        ( 1,  0) => 5,
        # (-1, -1) => 1,
        # (-1,  0) => 2,
        # (-1,  1) => 3,
        # ( 0, -1) => 4,
        # ( 0,  0) => 5,
        # ( 0,  1) => 6,
        # ( 1, -1) => 7,
        # ( 1,  0) => 8,
        # ( 1,  1) => 9,
    )[(delta_x, delta_y)]
end

df.delta_state_x = df.ax - df.dx
df.delta_state_y = df.ay - df.dy
df.label = position_delta_to_class.(df.ax_ - df.ax, df.ay_ - df.ay)

model = MultinomialRegression(0.1)
fitresult = MLJLinearModels.fit(model, Float64.(hcat(df.delta_state_x, df.delta_state_y)), df.label)
function pred(model, state, params)
    W = reshape(params, 2+1, 5)
    x = vcat(state, 1)
    return MLJLinearModels.softmax(x'*W)
end
# model = glm(@formula(label ~ 0 + delta_state_x + delta_state_y), df, Poisson(),
#             contrasts = Dict(:label => DummyCoding()))


# task 4
initial_state = DSState([1, 1], [5, 1])
policy = RandomPolicy(mdp)
policy_value = evaluate(mdp, policy)(initial_state)

# task 5
using UnicodePlots
policy_values = [evaluate(mdp, policy)(initial_state) 
                 for _ in 1:1000]
UnicodePlots.boxplot(policy_values)
