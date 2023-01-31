
util.jl  
```julia
function make_DroneSurveillance_MDP()
  # makes an MDP out of the POMDP
end
```

make_data.jl  
```julia
function make_data()
  # make (s, a, s')
  write("data.csv")
end
```

train_model.jl
```julia
open('data.csv')
model = ...
write(model, "model.[x]")
```

```julia
function make_approx_MDP(model)
  # uses the NN as the transition probabilities
  return mdp
end
```

```julia
get_policy(MDP)
eval_policy(Policy) :: Real
```

```julia
function plot_eval(pol)
  res = [eval_policy(pol) for _ in 1:1000]
  boxplot(res)
```
