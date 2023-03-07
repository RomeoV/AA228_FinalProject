using DataFrames
using DroneSurveillance
# using StatsPlots
import POMDPTools
import CSV
import StatsBase: mean, std
import FLoops: @floop, @reduce
import LaTeXStrings: @L_str
import Plots: plot, plot!, pgfplotsx; # pgfplotsx()
import StatsPlots: @df

function run_experiments(; dry=false, outpath::Union{String, Nothing}=nothing)
    nx_vals = [10]  # ny is the same
    agent_aggressiveness_vals = LinRange(0//1, 1//1, 7)
    # agent_aggressiveness_vals = [0., 0.5, 1.0]

    seed_vals = rand(UInt, 7)
    policy_strat_vals = [:perfect, :random]

    df = DataFrame([Int[], Float64[], UInt[], Symbol[], Float64[]],
                   [:nx, :agent_aggressiveness_p, :seed_val, :policy_strat, :score],
    )

    lck = ReentrantLock()
    @floop for nx ∈ nx_vals,
               p ∈ agent_aggressiveness_vals,
               seed ∈ seed_vals,
               pol ∈ policy_strat_vals
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

function plot_results(df::DataFrame)
    plt = @df df plot(:agent_aggressiveness_p, :score_mean;
                      group=:policy_strat, ribbon=2*:score_std, fillalpha=0.25,
                      label=nothing, seriescolor=[1 ;; 2],
                      title="Reward vs agent agressiveness",
                      xlabel="Agent perfect step prob",
                      ylabel="Reward",
                      leg_title=L"$T$ model")
    @df df plot!(plt, :agent_aggressiveness_p, :score_mean;
                      group=:policy_strat, ribbon=:score_std, fillalpha=0.5,
                      seriescolor=[1 ;; 2])
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
