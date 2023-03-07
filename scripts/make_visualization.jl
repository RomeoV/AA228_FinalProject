using Revise, AA228_FinalProject, DroneSurveillance, POMDPs, POMDPTools,
      Distributions, POMDPGifs, Cairo
AA = AA228_FinalProject;
plan_using_approx_model = true

mdp = DroneSurveillanceMDP{PerfectCam}(size=(10, 10),
                                       agent_strategy=DSAgentStrat(1//4));
T_model = if plan_using_approx_model
    create_linear_transition_model(mdp)
else
    DSPerfectModel()
end

U = let mdp=deepcopy(mdp)
    mdp.transition_model = T_model
    U = AA.value_iteration(mdp);
end
@assert mdp.transition_model isa DSPerfectModel
pol = POMDPTools.FunctionPolicy(s->AA.runtime_policy(mdp, U, s));
makegif(mdp, pol, filename="/tmp/out.gif")
