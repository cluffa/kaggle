using Flux
using CUDA
using ProgressMeter
using Plots
using MLUtils
using Random

CUDA.allowscalar(false)

RES = (256, 256)
GPU_BATCH_SIZE = 32

include("ScanData.jl");
data_dict = read_scans();
scans = (data_dict[1][:scan], data_dict[2][:scan], data_dict[3][:scan]);
masks = (data_dict[1][:mask], data_dict[2][:mask], data_dict[3][:mask]);

inklabels = (data_dict[1][:inklabels], data_dict[2][:inklabels], data_dict[3][:inklabels]);

@info "Data" summary(scans) summary(masks) summary(inklabels)

scan_patched = PatchedArray(scans, masks, RES);
ink_patched = PatchedArray(inklabels, masks, RES);

@info "Patched Data" summary(scan_patched) summary(ink_patched)

CPU_BATCH_SIZE = size(scan_patched)[end] ÷ 3

x_buffer = zeros(Float16, (256, 256, 65, CPU_BATCH_SIZE));

@info "Batches" batches

include("Model.jl");
model = new_model() |> gpu;
losses = Float32[]

lrs = Dict(
    1 => 1e-3,
    2 => 1e-4,
    3 => 1e-5,
    4 => 1e-6,
    5 => 1e-7,
    6 => 1e-8,
    7 => 1e-9,
    8 => 1e-10,
    9 => 1e-11,
    10 => 1e-12,
)

for epoch in 1:10
    @info "Epoch $epoch"

    optim = Flux.setup(Flux.Adam(lrs[epoch]) , model)

    r = shuffle(1:size(scan_patched)[end])
    batches = (r[1:CPU_BATCH_SIZE], r[(CPU_BATCH_SIZE + 1):(2 * CPU_BATCH_SIZE)], r[(2 * CPU_BATCH_SIZE + 1):end])

    for (i, batch) in enumerate(batches)
        Threads.@threads for (i, j) in zip(axes(y_buffer, 4), batch) |> collect
            @inbounds x_buffer[:, :, :, i] .= scan_patched[:, :, :, j]
        end

        @info "Cpu Batch $i"

        gpu_dataloader = Flux.DataLoader((x_buffer, @view(inklabels[:, :, :, batch])), batchsize = GPU_BATCH_SIZE, shuffle = false, partial = false);

        @showprogress for batch in gpu_dataloader
            x, y = batch .|> gpu

            loss, grads = Flux.withgradient(model) do m
                Flux.Losses.logitbinarycrossentropy(m(x), y)
            end

            Flux.update!(optim, model, grads[1])
            push!(losses, loss)

            open("losses.txt", "a") do f
                println(f, loss)
            end

            scatter(losses, yaxis=:log, legend = false, title = "loss") |> display
        end
    end

    @save "model_$epoch.bson" model
end

@view(data_dict[2][:scan][:, :, 1, 1]) .|> Gray
data_dict[2][:mask] .|> Gray


















