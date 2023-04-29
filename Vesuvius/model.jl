using Flux
using CUDA
using JLD2
using Images
using ProgressMeter
using TiffImages
using MosaicViews
using StackViews
using Statistics
using Plots
using BenchmarkTools
using BSON: @save
using Random: shuffle
using MLUtils
using NNlib
using Metalhead

CUDA.allowscalar(false)

include("jld2_setup.jl")
include("3dconvnext.jl")

BUF = 12
RES = BUF * 2 + 1

scans = Array{N0f16}(undef, scan_size("train.jld2", "3")..., 1);

read_scan!(scans, "train.jld2", "3");

inklabels = read_inklabels("train.jld2", "3");

mask = read_mask("train.jld2", "3");

@info "train data" summary(scans) summary(inklabels) summary(mask)

# model = new_convnext3d() |> gpu;
model = Chain([
    Conv((3, 3), 65 => 65, relu, groups = 65),
    MeanPool((2, 2)),
    Conv((3, 3), 65 => 65, relu, groups = 65),
    MeanPool((2, 2)),
    Conv((3, 3), 65 => 65, relu, groups = 65),
    MeanPool((2, 2)),
    Flux.flatten,
    Dense(65 => 128, relu),
    Dense(128 => 2, relu),
    softmax

    # GlobalMeanPool(),
    # Flux.flatten,
    # Dense(128 => 1),
    # Base.Fix2(reshape, (RES, RES, 3, :)),
    # ViT(:tiny; imsize = (RES, RES), inchannels = 3, patch_size = (5, 5), nclasses = 1)
]) |> gpu;



insize = (RES, RES, 65, 9)
@info "model test" insize outsize=CUDA.rand(Float32, insize) |> model |> size

idxs_true = [(i, j) for i in axes(mask, 1)[BUF + 1:end - BUF] for j in axes(mask, 2)[BUF + 1:end - BUF] if mask[i, j] == 1 && inklabels[i, j] == 1];
idxs_false = [(i, j) for i in axes(mask, 1)[BUF + 1:end - BUF] for j in axes(mask, 2)[BUF + 1:end - BUF] if mask[i, j] == 1 && inklabels[i, j] == 0];

shortest = min(length(idxs_true), length(idxs_false))

idxs_true = shuffle(idxs_true)[1:shortest]
idxs_false = shuffle(idxs_false)[1:shortest]

idxs = vcat(idxs_true, idxs_false) |> shuffle

BS = 1024*16
dataloader = Flux.DataLoader(idxs, batchsize = BS, shuffle = true, partial = false)

@info "DataLoader" batchsize = BS batches = length(dataloader)

x = CUDA.rand(Float32, (RES, RES, 65, BS));
y = CUDA.rand(Float32, (2, BS));

y_true = CUDA.CuArray(Float32[1., 0.])
y_false = CUDA.CuArray(Float32[0., 1.])

@info "batch temp train data" summary(x) summary(y)

loss_fn = Flux.binary_focal_loss

@info "testing grad"
CUDA.@time loss, grads = Flux.withgradient(model) do m
    ŷ = m(x)
    loss_fn(ŷ, y; agg = sum)
end;

losses = Float32[]
optim = Flux.setup(Flux.Adam(1e-3), model);
for epoch in 1:3
    @info "Epoch $epoch"
    @showprogress for (i, idxs_batch) in enumerate(dataloader)
        for (j, (x_idx, y_idx)) in enumerate(idxs_batch)
            @inbounds x[:, :, :, j] = scans[x_idx - BUF:x_idx + BUF, y_idx - BUF:y_idx + BUF, :, :, 1] .|> Float32;
            @inbounds y[:, j] = inklabels[x_idx, y_idx] ? y_true : y_false
        end

        loss, grads = Flux.withgradient(model) do m
            ŷ = m(x)
            loss_fn(ŷ, y; agg = mean)
        end

        Flux.update!(optim, model, grads[1])
        push!(losses, loss)

        
        plot(losses, yaxis=:log, legend = false, title = "loss") |> display

        if i % 100 == 0
            lr = 1e-5
            optim = Flux.setup(Flux.Adam(lr), model);
            @info "lowering learning rate" lr
        end

        if i % 200 == 0
            lr = 1e-8
            optim = Flux.setup(Flux.Adam(lr), model);
            @info "lowering learning rate" lr
        end
    end

    @save "model_$epoch.bson" model
end

idx = rand(idxs)

x = scans[idx[1] - BUF:idx[1] + BUF, idx[2] - BUF:idx[2] + BUF, :, :, 1] .|> Float32 |> gpu;
y = inklabels[idx[1], idx[2]]

x |> model