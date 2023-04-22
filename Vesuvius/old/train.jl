using Colors
using Flux
using CUDA
using Metalhead
using Images
using ProgressMeter
using Plots

include("data_utils.jl")

@assert Threads.nthreads() > 1 "remember to start julia more than 1 thread"
@assert CUDA.functional() "CUDA not available"

scans, inklabels = DataUtils.load_patches();

# backbones
# Metalhead.backbone(Metalhead.ResNet(18; inchannels = 65));
# Metalhead.backbone(Metalhead.ViT(:base; inchannels = 65));

model = Metalhead.backbone(Metalhead.ResNet(18; inchannels = 65)) |>
    backbone -> Metalhead.UNet((512, 512), 65, 1, backbone) |>
    gpu;

scan1 = @view scans[:, :, :, 1:208];
ink1 = @view inklabels[:, :, :, 1:208];

loader = Flux.DataLoader((scan1, ink1), batchsize = 6, shuffle = true, partial = true);

optim = Flux.setup(Flux.Adam(1e-3), model);

losses = Float32[]
for epoch in 1:5
    @info "Epoch $epoch"
    @showprogress for (x, y) in loader
        x = gpu(x)
        y = gpu(y)
        loss, grads = Flux.withgradient(model) do m
            Flux.Losses.binarycrossentropy(m(x), y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end

    GC.gc()
end;

GC.gc()

plot(losses)
