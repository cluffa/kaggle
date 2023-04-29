using TiffImages
using Images
using JLD2
using ProgressMeter
using CUDA

jldopen("test.jld2", "w") do f
    f["ink"] = joinpath("train", "1", "inklabels.png") |> load .|> x -> x.r > 0.5;
    f["mask"] = joinpath("train", "1", "mask.png") |> load .|> x -> x.r > 0.5;
    scan = Array{N0f16}(undef, size(f["ink"], 1), size(f["ink"], 2), 65)

    files = readdir(joinpath("train", "1", "surface_volume"), join = true)

    for (i, file) in enumerate(files)
        scan[:, :, i] .= file |> load |> channelview
    end

    f["scan"] = scan
end;

jldopen("test.jld2", "r") do f
    @view(f["scan"][:, :, 3])
end .|> Gray

jldopen("test2.jld2", "w") do f
    
end