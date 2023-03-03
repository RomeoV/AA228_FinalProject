using Revise, AA228_FinalProject, DroneSurveillance, POMDPs, POMDPTools,
      Distributions, POMDPGifs, Cairo
mdp = DroneSurveillanceMDP{PerfectCam}(size=(10, 10), agent_strategy=DSAgentStrat(1.0));
AA = AA228_FinalProject;
U = AA.value_iteration(mdp);
pol = POMDPTools.FunctionPolicy(s->AA.runtime_policy(mdp, U, s));
makegif(mdp, pol, filename="/tmp/out.gif")
