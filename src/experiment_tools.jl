using DataFrames
using DroneSurveillance
import POMDPTools
import CSV
import Base: product
import StatsBase: mean, std
import FLoops: @floop, @reduce
import LaTeXStrings: @L_str
import Plots: plot, plot!, pgfplotsx; # pgfplotsx()
import StatsPlots: @df
import Random: shuffle

function run_experiments(; dry=false, outpath::Union{String, Nothing}=nothing)
    nx_vals = [10]  # ny is the same
    agent_aggressiveness_vals = LinRange(0//1, 1//1, 3)

    seed_vals = rand(UInt, 3)
    policy_strat_vals = [:perfect, :conformalized, :temp_calibrated, :linear, :random]

    df = DataFrame([Int[], Float64[], UInt[], Symbol[], Float64[]],
                   [:nx, :agent_aggressiveness_p, :seed_val, :policy_strat, :score],
    )

    lck = ReentrantLock()
    @floop for (nx, p, seed, pol) in product(nx_vals, 
                                             agent_aggressiveness_vals,
                                             seed_vals,
                                             policy_strat_vals)
        @show (nx, p, seed, pol)
        val = eval_problem(nx, p, pol; seed_val=seed, dry=dry)
        lock(lck) do
            push!(df, (nx, p, seed, pol, val))
        end
    end
    sort!(df, [:agent_aggressiveness_p, :policy_strat])
    if !isnothing(outpath)
        CSV.write(outpath, df)
    end
    return df
end

function get_conformalized_policy_under_domain_shift(P, Δp; p0=0.5, dry=false)
    P.mdp.transition_model = DSPerfectModel()
    P.mdp.agent_strategy = DSAgentStrat(p0)
    mdp = P.mdp
    mdp_shift = let mdp = deepcopy(mdp)
        mdp.agent_strategy = DSAgentStrat(p0 + Δp)
        mdp
    end
    conf_model = create_conformalized_transition_model(mdp, mdp_shift)

    P.mdp.transition_model = conf_model
    U = value_iteration_conformal(P.mdp; dry=dry);
    POMDPTools.FunctionPolicy(s->runtime_policy_conformal(P.mdp, U, s));
end

function get_linear_policy_under_domain_shift(P, Δp; p0=0.5, dry=false)
    # currently ignores the domain shift
    P.mdp.transition_model = DSPerfectModel()
    P.mdp.agent_strategy = DSAgentStrat(p0)
    T_model::DSLinModel = create_linear_transition_model(P.mdp)
    P.mdp.transition_model = T_model
    U = value_iteration(P.mdp; dry=dry);
    POMDPTools.FunctionPolicy(s->runtime_policy(P.mdp, U, s));
end

function run_domain_shift_experiments(; dry=false, outpath::Union{String, Nothing}=nothing, verbose=false)
    nx_vals = [10]  # ny is the same
    agent_aggressiveness_deltas = [0., 0.3]
    # agent_aggressiveness_vals = [0., 0.5, 1.0]

    seed_vals = rand(UInt, 1)
    policy_strat_vals = [:conformalized, :linear]

    df = DataFrame([Int[], Float64[], UInt[], Symbol[], Float64[]],
                   [:nx, :agent_aggressiveness_delta, :seed_val, :policy_strat, :score],
    )

    lck = ReentrantLock()
    for (nx, Δp, seed, pol) in product(nx_vals, 
                                              agent_aggressiveness_deltas,
                                              seed_vals,
                                              policy_strat_vals)
        # temp commment, remove later
        @show (nx, Δp, seed, pol)
        policy_value = begin
            P = make_P()
            P.mdp.size = (nx, nx)
            P.mdp.transition_model = DSPerfectModel()
            policy = if pol == :linear
                get_linear_policy_under_domain_shift(deepcopy(P), Δp)
            elseif pol == :conformalized
                get_conformalized_policy_under_domain_shift(deepcopy(P), Δp)
            else
                @error "invalid policy type"
            end
            initial_state = DSState([1, 1], rand(3:nx, 2))
            @assert P.mdp.transition_model isa DSPerfectModel
            P.mdp.agent_strategy = DSAgentStrat(0.5+Δp)
            U_π = policy_evaluation(P.mdp, policy;
                                    trace_state=(verbose ? initial_state : nothing),
                                    dry=dry)
            U_π[initial_state]
        end

        lock(lck) do
            push!(df, (nx, Δp, seed, pol, policy_value))
        end
    end

    sort!(df, [:agent_aggressiveness_delta, :policy_strat])
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
                      leg_title=L"$T$ model")
    @df df plot!(plt, :agent_aggressiveness_p, :score_mean;
                      group=:policy_strat, ribbon=:score_std, fillalpha=0.5,
                      seriescolor=seriescolors)
    return plt
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
