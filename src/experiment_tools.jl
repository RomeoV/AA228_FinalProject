using DataFrames
using DroneSurveillance
import DroneSurveillance: DSAgentStrat, DSTransitionModel, DSPerfectModel, DSLinModel, DSLinCalModel, DSConformalizedModel, DSRandomModel
import POMDPTools
import CSV
import Base: product
import StatsBase: mean, std
import FLoops: @floop, @reduce
import LaTeXStrings: @L_str
import Plots: plot, plot!, pgfplotsx; # pgfplotsx()
import StatsPlots: @df
import Random: shuffle

function run_experiments(; dry=false,
                         outpath::Union{String, Nothing}=nothing,
                         verbose=false, n_seeds=(dry ? 1 : 5))
    nx_vals = [10]  # ny is the same
    agent_aggressiveness_vals = LinRange(0//1, 1//1, (dry ? 2 : 7))

    seed_vals = rand(UInt, n_seeds)
    policy_strats = [DSLinModel,
                     DSLinCalModel,
                     DSConformalizedModel,
                     DSPerfectModel,
                     DSRandomModel]

    df = DataFrame([Int[], Float64[], UInt[], String[], Float64[]],
                   [:nx, :agent_aggressiveness_p, :seed_val, :policy_strat, :score],
    )

    lck = ReentrantLock()
    @floop for (nx, p, seed, pol) in product(nx_vals,
                                      agent_aggressiveness_vals,
                                      seed_vals,
                                      policy_strats)
        @show (nx, p, seed, pol)
        val = eval_problem(pol, nx, p; seed_val=seed, dry=dry, parallel=false)
        lock(lck) do
            row = (nx, p, seed, string(pol)[3:end], val)
            @show row
            push!(df, row)
        end
    end
    sort!(df, [:agent_aggressiveness_p, :policy_strat])
    if !isnothing(outpath)
        while isfile(outpath); outpath = string(outpath, "_"); end
        CSV.write(outpath, df)
    end
    return df
end

function eval_domain_shift(transition_model::Type{<:DSTransitionModel},
                           nx::Int,
                           agent_strategy_Δp::Real;
                           p0=0.5,
                           seed_val=rand(UInt), verbose=false, dry=false)
    seed!(seed_val)
    mdp = DroneSurveillanceMDP{PerfectCam}(size=(nx, nx), agent_strategy=DSAgentStrat(p0))
    mdp_shift = let mdp = deepcopy(mdp)
        mdp.agent_strategy = DSAgentStrat(p0 + agent_strategy_Δp)
        mdp
    end

    transition_model = @match transition_model begin
        _::Type{DSPerfectModel} =>  DSPerfectModel(mdp_shift.agent_strategy)
        _::Type{DSRandomModel} => DSRandomModel(mdp_shift)
        _::Type{DSLinModel}  => create_linear_transition_model(mdp; dry=dry, verbose=verbose)
        _::Type{DSLinCalModel} => create_temp_calibrated_transition_model(mdp, mdp_shift; dry=dry, verbose=verbose)
        _::Type{DSConformalizedModel} => create_conformalized_transition_model(mdp, mdp_shift; dry=dry, verbose=verbose)
        _ => error("unknown transition model")
    end
    @debug "Finished creating model."
    U = value_iteration(mdp, transition_model; dry=dry);
    policy = POMDPTools.FunctionPolicy(s->runtime_policy(mdp_shift, transition_model, U, s))

    initial_state = DSState([1, 1], [ceil(Int, (nx + 1) / 2), ceil(Int, (nx + 1) / 2)])
    U_π = policy_evaluation(mdp_shift, mdp_shift.agent_strategy, policy;
                            trace_state=(verbose ? initial_state : nothing),
                            dry=dry)
    policy_value = U_π[initial_state]
    return policy_value
end

function run_domain_shift_experiments(; dry=false, outpath::Union{String, Nothing}=nothing, verbose=false)
    nx_vals = [10]  # ny is the same
    agent_aggressiveness_deltas = LinRange(-1//2, 1//2, (dry ? 2 : 7))

    seed_vals = rand(UInt, (dry ? 2 : 3))
    policy_strats = [DSLinModel,
                     DSLinCalModel,
                     DSConformalizedModel,
                     DSPerfectModel,
                     DSRandomModel]

    df = DataFrame([Int[], Float64[], UInt[], String[], Float64[]],
                   [:nx, :agent_aggressiveness_delta_p, :seed_val, :policy_strat, :score],
    )

    lck = ReentrantLock()
    @floop for (nx, Δp, seed, pol) in product(nx_vals,
                                       agent_aggressiveness_deltas,
                                       seed_vals,
                                       policy_strats)
        @show (nx, Δp, seed, pol)
        val = eval_domain_shift(pol, nx, Δp; seed_val=seed, dry=dry)
        lock(lck) do
            push!(df, (nx, Δp, seed, string(pol)[3:end], val))
        end
    end
    sort!(df, [:agent_aggressiveness_delta_p, :policy_strat])
    if !isnothing(outpath)
        CSV.write(outpath, df)
    end
    return df
end

function plot_results(df::DataFrame)
    n_plots = length(unique(df.policy_strat))
    seriescolors = 1:n_plots |> x->reshape(x, 1, :) |> collect
    plt = @df df plot(:agent_aggressiveness_p, :score_mean;
                      group=:policy_strat, ribbon=2*:score_std, fillalpha=0.25,
                      label=nothing, seriescolor=seriescolors,
                      title="Reward vs agent agressiveness",
                      xlabel="Agent perfect step prob",
                      ylabel="Reward",
                      leg_title=L"$T$ model",
                      leg=:bottomleft)
    @df df plot!(plt, :agent_aggressiveness_p, :score_mean;
                      group=:policy_strat, ribbon=:score_std, fillalpha=0.5,
                      seriescolor=seriescolors)
    return plt
end

function process_domain_shift_data(df::DataFrame; p0=0.5)
    df[!, :agent_aggressiveness_delta] = df.agent_aggressiveness_delta .+ p0
    rename!(df, :agent_aggressiveness_delta => :agent_aggressiveness_p)
    process_data(df)
end

function process_data(df::DataFrame)
    grps = groupby(df, [:policy_strat, :agent_aggressiveness_p])
    df = combine(grps, :score=>mean, :score=>std)
    df = select(df, :policy_strat, :agent_aggressiveness_p, :score_mean, :score_std)
    sort!(df, :policy_strat)
    df
end

function process_data(path::String, outpath::String)
    df = CSV.read(path, DataFrame)
    grps = groupby(df, [:policy_strat, :agent_aggressiveness_p])
    df = combine(grps, :score=>mean, :score=>std)
    df = select(df, :policy_strat, :agent_aggressiveness_p, :score_mean, :score_std)
    # @df df plot(:agent_aggressiveness_p, :score_mean; group=:policy_strat, ribbon=2*:score_std, fillalpha=0.25)
    # @df df plot!(:agent_aggressiveness_p, :score_mean; group=:policy_strat, ribbon=:score_std, fillalpha=0.5)
    sort!(df, :policy_strat)
    rename!(s->replace(s, "_"=>" "), df)
    CSV.write(outpath, df)
    df
end

function run_calibration_experiments(Δp; p0=0.5, dry=false)
    mdp = DroneSurveillanceMDP{PerfectCam}(size=(10, 10), agent_strategy=DSAgentStrat(p0))
    mdp_shift = let mdp = deepcopy(mdp)
        mdp.agent_strategy = DSAgentStrat(p0 + Δp)
        mdp
    end
    @assert mdp.agent_strategy.p == p0  # make sure original mdp is not modified

    lin_model  = create_linear_transition_model(mdp; dry=dry)
    conf_model = create_conformalized_transition_model(mdp, mdp_shift; dry=dry)

    history = make_history(mdp_shift; N=(dry ? 10 : 1000)) |> shuffle
    lin_results  = measure_calibration(lin_model,  history)
    conf_results = measure_calibration(conf_model, history)

    n_lin = length(lin_results); n_conf = length(conf_results)

    DataFrame([vcat(keys(lin_results)...,    keys(conf_results)...),
               vcat(values(lin_results)...,  values(conf_results)...),
               vcat(fill(:linear, n_lin)..., fill(:conformalized, n_conf)...)],
              [:λ, :calib_val, :model])
end

function plot_calibration_experiments(df; kwargs...)
    plt = plot(0:0.01:1, 0:0.01:1;
               linestyle=:dash,
               label="Perfect model",
               xlabel="λ",
               ylabel="coverage",
               kwargs...)
    @df df plot!(plt, :λ, :calib_val; group=:model);
    plt
end
