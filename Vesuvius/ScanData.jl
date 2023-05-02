using Images
using JLD2
using TiffImages
using ProgressMeter
using SharedArrays
using OffsetArrays
using MLUtils

function read_scan(train_set, reload = false)
    @assert isdir("train") "train directory not found"

    type = Float16

    dir = joinpath("train", "$train_set")
    sv_dir = joinpath(dir, "surface_volume")

    if !isdir("data")
        mkpath("data")
    end

    scan_dat = abspath(joinpath("data", "scans_$train_set.dat"))
    data_jld2 = joinpath("data", "data_$train_set.jld2")

    if !reload && isfile(data_jld2)
        file = jldopen(data_jld2, "r")

        mask = file["mask"]
        inklabels = file["inklabels"]
        indices = file["indices"]
        vert, horz = size(mask)

        close(file)

        scan = SharedArray{type}(abspath(scan_dat), (vert, horz, 65));

        return Dict(:scan => scan, :mask => mask, :inklabels => inklabels, :indices => indices)
    else
        mask = load(joinpath(dir, "mask.png")) .|> Gray |> channelview .> 0.5
        inklabels = load(joinpath(dir, "inklabels.png")) .|> Gray |> channelview .> 0.5
        indices = [Int16.((i, j)) for i in axes(mask, 1) for j in axes(mask, 2) if mask[i, j] == 1]

        vert, horz = size(mask)
        
        scan = SharedArray{type}(abspath(scan_dat), (vert, horz, 65));

        jldopen(data_jld2, "w") do file
            file["mask"] = mask
            file["inklabels"] = inklabels
            file["indices"] = indices
        end

        files = readdir(sv_dir, join = true)
        @showprogress "Loading Tiff Files" for (i, file) in enumerate(files)
            scan[:, :, i] = load(file) .|> type
        end

        return Dict(:scan => scan, :mask => mask, :inklabels => inklabels, :indices => indices)
    end
end

function read_scans(reload = false)
    Dict([i => read_scan(i) for i in 1:3])
end

struct PatchedArray{T, N, A} <: AbstractArray{T, N}
    patches::A # vector of patch views

    # make PatchedArray from  patches
    function PatchedArray(;patches)

        T = eltype(first(patches))
        N = ndims(first(patches)) + 1
        A = typeof(patches)

        return new{T, N, A}(patches)
    end

    function PatchedArray(data::Tuple, masks::Tuple, (h, w)::Tuple{Int, Int} = (256, 256), (overlap_h, overlap_w)::Tuple{Int, Int} = (0, 0))
        patched = [PatchedArray(A, M, (h, w), (overlap_h, overlap_w)) for (A, M) in zip(data, masks)]
        patches = vcat([A.patches for A in patched]...)
        return PatchedArray(;patches = patches)
    end

    function PatchedArray(data::Tuple, (h, w)::Tuple{Int, Int} = (256, 256), (overlap_h, overlap_w)::Tuple{Int, Int} = (0, 0))
        patched = [PatchedArray(A, (h, w), (overlap_h, overlap_w)) for A in data]
        patches = vcat([A.patches for A in patched]...)
        return PatchedArray(;patches = patches)
    end

    function PatchedArray(data::AbstractArray, (h, w)::Tuple{Int, Int} = (256, 256), (overlap_h, overlap_w)::Tuple{Int, Int} = (0, 0))
        mask = trues(size(data)[1:2])
        PatchedArray(data, mask, (h, w), (overlap_h, overlap_w))
    end

    function PatchedArray(data::AbstractArray, mask::BitArray, (h, w)::Tuple{Int, Int} = (256, 256), (overlap_h, overlap_w)::Tuple{Int, Int} = (0, 0))
        if ndims(data) == 2
            data = reshape(data, size(data)..., 1)
        end

        @assert ndims(data) == 3 "A must be a 2D or 3D array"
        @assert h > overlap_h && w > overlap_w "patch size must be greater than overlap size"

        effective_patch_size = (h - overlap_h, w - overlap_w)

        res = ceil.(Int, size(data)[1:2] ./ effective_patch_size) .* effective_patch_size

        # TODO possibly pad each incomplete patch instead of padding the whole array
        data_padded = res != size(data)[1:2] ? PaddedView(0, data, (res[1], res[2], size(data, 3))) : data
        mask_padded = res != size(data)[1:2] ? PaddedView(false, mask, (res[1], res[2])) : mask

        rows, cols = size(data_padded)
        patches = AbstractArray{eltype(data_padded), 3}[]
    
        row_step = h - overlap_h
        col_step = w - overlap_w
    
        @inbounds @views for row in 1:row_step:(rows - h + 1)
            for col in 1:col_step:(cols - w + 1)

                patch = data_padded[row:(row + h - 1), col:(col + w - 1), :]
                mask_patch = mask_padded[row:(row + h - 1), col:(col + w - 1)]

                if any(mask_patch)
                    push!(patches, patch)
                end
            end
        end

        T = eltype(data_padded)
        N = ndims(data_padded) + 1
        A = typeof(patches)

        @info "Patching" patch_size=(h, w) overlap_size=(overlap_h, overlap_w) original=size(data) padded=size(data_padded) patches=length(patches) out_size=(size(first(patches))..., length(patches))

        return new{T, N, A}(patches)
    end
end

Base.size(A::PatchedArray) = (size(first(A.patches))..., length(A.patches))
Base.axes(A::PatchedArray) = (axes(first(A.patches))..., Base.OneTo(length(A.patches)))

@inline function Base.getindex(A::PatchedArray, i::Vararg{Int})
    # @boundscheck checkbounds(A, i...)
    idx, idxs = last(i), Base.front(i)
    return @inbounds A.patches[idx][idxs...]
end

@inline function Base.setindex!(A::PatchedArray, x, i::Vararg{Int})
    # @boundscheck checkbounds(A, i...)
    idx, idxs = last(i), Base.front(i)
    @inbounds A.patches[idx][idxs...] = x
end

function concat(arrays)
    @assert length(arrays) > 0 "arrays must be non-empty"
    patches = vcat([array.patches for array in arrays]...)

    return PatchedArray(;patches = patches)
end

# data = read_scans();
# scans = (data[1][:scan], data[2][:scan], data[3][:scan]);
# masks = (data[1][:mask], data[2][:mask], data[3][:mask]);

# scan_patched = PatchedArray(scans, masks, (256, 256), (0, 0)); scan_patched |> size
# mask_patched = PatchedArray(masks, masks, (256, 256), (0, 0)); mask_patched |> size

# using Flux, CUDA

# all = CuArray(all)
# @view(patched[:, :, 1, 3]) .|> Gray

nothing