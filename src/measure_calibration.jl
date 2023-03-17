import StatsBase: mean
import DroneSurveillance: predict
import Unzip: unzip
using OrderedCollections: OrderedDict

# Measures empirical frequency of true value being in prediction set, for
# different converage levels.
# Note, history can e.g. be recorded with a distribution shift.
function measure_calibration(model, history; dry=false) :: OrderedDict{Real, Real}
    dset_s = getfield.(history, :s)
    dset_a = getfield.(history, :a)
    dset_s_ = getfield.(history, :sp)

    true_Δxs = [s_.agent.x - s_.quad.x
                for s_ in dset_s_]
    true_Δys = [s_.agent.y - s_.quad.y
                for s_ in dset_s_]

    results = OrderedDict(
        λ => begin
                preds = predict.([model], dset_s, dset_a, λ)
                preds_Δx, preds_Δy = unzip(preds)
                mean([true_Δxs .∈ preds_Δx ;
                      true_Δys .∈ preds_Δy]) .|> x->round(x, sigdigits=3)
        end
        for λ in vcat(0.1:0.1:0.9, 0.99, 0.995)
    )
    return results
end
