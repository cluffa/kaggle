using Flux
using CUDA
using ProgressMeter
using Plots
using MLUtils
using Statistics

CUDA.allowscalar(false)

RES = (256, 256)
OVERLAP = (0, 0)
CPU_BATCH_SIZE = 120
GPU_BATCH_SIZE = 6

include("ScanData.jl");
data = read_scans();
scans = (data[1][:scan], data[2][:scan], data[3][:scan]);
masks = (data[1][:mask], data[2][:mask], data[3][:mask]);

scan_patched = PatchedArray(scans, masks, RES, OVERLAP);
mask_patched = PatchedArray(masks, masks, RES, OVERLAP);

@info "Patched Data" summary(scan_patched) summary(mask_patched)

# cpu_dataloader = Flux.DataLoader((scan_patched, mask_patched), batchsize = CPU_BATCH_SIZE, shuffle = true, partial = true, parallel = true);

include("Model.jl");
model = new_model() |> gpu;
losses = Float32[]
loss_fn = Flux.binarycrossentropy;
optim = Flux.setup(Flux.Adam(1e-8), model);

# for (i, data) in cpu_dataloader |> enumerate
#     @info "Cpu Batch" i
for i in 1:10
    @info "Epoch" i

    gpu_dataloader = Flux.DataLoader((scan_patched, mask_patched), batchsize = GPU_BATCH_SIZE, shuffle = true, partial = true, parallel = true);

    @showprogress for batch in cpu_dataloader
        x, y = batch .|> gpu

        loss, grads = Flux.withgradient(model) do m
            ŷ = m(x)
            loss_fn(ŷ, y; agg = mean)
        end

        Flux.update!(optim, model, grads[1])
        push!(losses, loss)

        scatter(losses, yaxis=:log, legend = false, title = "loss") |> display
    end

    @save "model_$epoch.bson" model
end
