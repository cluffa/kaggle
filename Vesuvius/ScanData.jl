using Images
using JLD2
using TiffImages
using ProgressMeter
using SharedArrays
using OffsetArrays
using MLUtils

struct ScanDataObs{T, N, A} <: AbstractArray{T, N}
    slices::A
end

function ScanDataObs(slices::Vector{<:AbstractArray})
    T = mapreduce(eltype, promote_type, slices)
    N = ndims(first(slices)) == 3 ? 4 : ndims(first(slices))
    # slices = map(OffsetArrays.no_offset_view, slices)
    return ScanDataObs{T, N, typeof(slices)}(slices)
end

Base.size(A::ScanDataObs) = (size(first(A.slices))[1:end-1]..., length(A.slices))
Base.axes(A::ScanDataObs) = (axes(first(A.slices))[1:end-1]..., Base.OneTo(length(A.slices)))

@inline function Base.getindex(A::ScanDataObs, i::Vararg{Int})
    # @boundscheck checkbounds(A, i...)
    idx, idxs = last(i), Base.front(i)
    return @inbounds A.slices[idx][idxs..., 1]
end

@inline function Base.setindex!(A::ScanDataObs, x, i::Vararg{Int})
    # @boundscheck checkbounds(A, i...)
    idx, idxs = last(i), Base.front(i)
    @inbounds A.slices[idx][idxs..., 1] = x
end

function read_scan(train_set, reload = false)
    @assert isdir("train") "train directory not found"

    type = N0f8

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

struct ScanData
    scan::AbstractArray{N0f8, 3}
    mask::Array{Bool, 2}
    inklabels::Array{Bool, 2}
    indices::Array{Tuple{Int16, Int16}, 1}
    buffer::Int
    function ScanData(i::Int, buffer = 12)
        data = read_scan(i)
        extend_rng = (1 - buffer : size(data[:mask], 1) + buffer, 1 - buffer : size(data[:mask], 2) + buffer, 1:65)
        new(PaddedView(N0f8(0), data[:scan], extend_rng), data[:mask], data[:inklabels], data[:indices], buffer)
    end
end

struct ScanDataGroup
    data::Tuple{ScanData, ScanData, ScanData}
    buffer::Int
    function ScanDataGroup(buffer = 12)
        new((ScanData(1, buffer), ScanData(2, buffer), ScanData(3, buffer)), buffer)
    end
end

MLUtils.numobs(sd::ScanData) = size(sd.indices, 1)
MLUtils.numobs(sdg::ScanDataGroup) = sum([numobs(i) for i in sdg.data])

function MLUtils.getobs(sd::ScanData, i::Integer)
    idx = sd.indices[i]
    @inbounds (ScanDataObs([@view(sd.scan[idx[1] - sd.buffer:idx[1] + sd.buffer, idx[2] - sd.buffer:idx[2] + sd.buffer, :, :])]), sd.inklabels[idx[1], idx[2], :, :])
end

function MLUtils.getobs(sd::ScanData, i::AbstractVector)
    idxs = sd.indices[i]
    slices = [@view(sd.scan[idx[1] - sd.buffer:idx[1] + sd.buffer, idx[2] - sd.buffer:idx[2] + sd.buffer, :, :]) for idx in idxs]
    mask = [sd.inklabels[idx[1], idx[2], :] for idx in idxs]
    
    @inbounds (ScanDataObs(slices), stack(mask; dims = 2))
end

function MLUtils.getobs(sdg::ScanDataGroup, i::Integer)
    if i <= numobs(sdg.data[1])
        getobs(sdg.data[1], i)
    elseif i <= numobs(sdg.data[1]) + numobs(sdg.data[2])
        getobs(sdg.data[2], i - numobs(sdg.data[1]))
    else
        getobs(sdg.data[3], i - numobs(sdg.data[1]) - numobs(sdg.data[2]))
    end
end

function MLUtils.getobs(sdg::ScanDataGroup, i::AbstractVector)
    first_group = i .<= numobs(sdg.data[1])
    second_group = (i .<= (numobs(sdg.data[1]) + numobs(sdg.data[2]))) .& (i .> numobs(sdg.data[1]))
    third_group = i .> (numobs(sdg.data[1]) + numobs(sdg.data[2]))

    slices = AbstractArray[]
    mask = AbstractArray[]

    if any(first_group)
        x, y = getobs(sdg.data[1], i[first_group])

        append!(slices, x.slices)
        push!(mask, y)
    end

    if any(second_group)
        x, y = getobs(sdg.data[2], i[second_group] .- numobs(sdg.data[1]))

        append!(slices, x.slices)
        push!(mask, y)
    end

    if any(third_group)
        x, y = getobs(sdg.data[3], i[third_group] .- numobs(sdg.data[1]) .- numobs(sdg.data[2]))

        append!(slices, x.slices)
        push!(mask, y)
    end

    @inbounds (ScanDataObs(slices), cat(mask..., dims = 2))
end

### example usage

# data = ScanData(1); # 1, 2, or 3
data = ScanDataGroup();

numobs(data) # 152_201_833 for grouped data

getobs(data, 1) .|> size # ((25, 25, 65, 1), (1, 1))
getobs(data, 1:10) .|> size # ((25, 25, 65, 10), (1, 10))

N = 1024*10

dataloader = DataLoader(data, batchsize = N, shuffle = true, partial = false)

@time x, y = first(dataloader);

@showprogress for (x, y) in dataloader
    sum(x) + sum(y)
end

#TODO get rid of PaddedView? and find redundant views