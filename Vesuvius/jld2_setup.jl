using TiffImages
using Images
using JLD2
using ProgressMeter
using CUDA

function write_artifact(f, n, t, train_dir)
    artifact = JLD2.Group(f, n)
    sv = JLD2.Group(artifact, "surface_volume")
    ink = joinpath(train_dir, n, "inklabels.png") |> load .|> x -> x.r > 0.5;
    artifact["size"] = size(ink);
    artifact["inklabels.png"] = load("train/$n/inklabels.png") .|> x -> x.r > 0.5;
    artifact["mask.png"] = load("train/$n/mask.png") .|> x -> x.r > 0.5;
    @showprogress "Processing artifact $n of 3" for file in readdir("train/$n/surface_volume/")
        sv[file] = load("train/$n/surface_volume/$file") |> channelview .|> t
    end

    GC.gc()
end

function write_all_artifacts(train_dir = "train", out = "train.jld2", t = N0f16)
    jldopen(out, "w") do f
        for n in readdir(train_dir)
            write_artifact(f, n, t, train_dir)
        end
    end
end

function scan_size(jld_file, scan_num)
    jldopen(jld_file, "r") do f
        (f["$scan_num"]["size"]..., length(keys(f["$scan_num"]["surface_volume"])))
    end
end

function read_scan!(A, jld_file, scan_num)
    jldopen(jld_file, "r") do f
        for (i, k) in enumerate(keys(f["$scan_num"]["surface_volume"]))
            A[:, :, i] = f["$scan_num"]["surface_volume"][k]
        end
    end

    return A
end

function read_scan(jld_file, scan_num, gpu = true)
    dims = scan_size(jld_file, scan_num)

    if gpu
        A = CuArray{Float32}(undef, dims...)
    else
        A = Array{Float32}(undef, dims...)
    end
    
    read_scan!(A, jld_file, scan_num)
end

