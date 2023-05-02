using Flux
using CUDA
using ProgressMeter
using Plots
using MLUtils
using Statistics

CUDA.allowscalar(false)

include("ScanData.jl");
data = read_scans();
scans = (data[1][:scan], data[2][:scan], data[3][:scan]);
masks = (data[1][:mask], data[2][:mask], data[3][:mask]);

scan_patched = PatchedArray(scans, masks);
mask_patched = PatchedArray(masks, masks);

include("Model.jl");
model = new_model() |> gpu;
optim = Flux.setup(Flux.Adam(1e-8), model);

dataloader = Flux.DataLoader((scan_patched, mask_patched), batchsize = 4, shuffle = true, partial = true, parallel = true);

# let batch = first(dataloader)
#     x, y = batch .|> gpu

#     loss, grads = Flux.withgradient(model) do m
#         ŷ = m(x)
#         Flux.binarycrossentropy(ŷ, y; agg = mean)
#     end

#     Flux.update!(optim, model, grads[1]);

#     batch = nothing
#     x = nothing
#     y = nothing
#     loss = nothing
#     grads = nothing
# end

data = nothing
scans = nothing
masks = nothing
scan_patched = nothing
mask_patched = nothing
model = nothing
optim = nothing
dataloader = nothing

GC.gc();
