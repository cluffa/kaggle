module DataUtils

using TiffImages
using ProgressMeter: @showprogress
using Colors: Float32, Gray
using Flux: pad_constant
using SharedArrays
using Images

"""
    pad_to_multiple(array, multiple::Tuple)


Pads the given array to be a multiple of `multiple` in each dimension.
"""
function pad_to_multiple(array, multiple::Tuple)
    pad = Int[]
    for (s, m) in zip(size(array), multiple)
        r = s % m
        p = r == 0 ? 0 : m - r
        push!(pad, p ÷ 2)
        push!(pad, p - p ÷ 2)
    end
    
    return pad_constant(array, tuple(pad...), 0)
end


"""
    pad_to_multiple(array, multiple::Int)

    
Pads the given array to be a multiple of `multiple` in the first two dimensions.
"""
function pad_to_multiple(array, multiple::Int)
    ndims = length(size(array))
    pad = (multiple, multiple, fill(1, ndims - 2)...)

    return pad_to_multiple(array, pad)
end


"""
    pad_to_512(array)


Pads the given array to be a multiple of 512 in the first two dimensions.
"""
function pad_to_512(array)
    ndims = length(size(array))
    multiple = (512, 512, fill(1, ndims - 2)...)

    pad = Int[]
    for (s, m) in zip(size(array), multiple)
        r = s % m
        p = r == 0 ? 0 : m - r
        push!(pad, p ÷ 2)
        push!(pad, p - p ÷ 2)
    end
    
    return pad_constant(array, tuple(pad...), 0)
end


"""
    load_patches()

Loads SharedArray from .dat file if it exists, otherwise loads from .tif files,  
splits them into 512x512 patches, loads into a SharedArray, and returns the SharedArray.
"""
function load_patches(reload = false)
    scans_data_path = abspath("scans.dat")
    inklabels_data_path = abspath("inklabels.dat")

    
    num_patches = 924 # should never change
    # for i in 1:3
    #     s = load(joinpath("train", "$i", "inklabels.png")) |> size
    #     num_patches += s .|> (x -> x / 512) .|> ceil .|> Int |> prod
    # end


    # run if files don't exist or reload is true
    if !isfile(scans_data_path) || !isfile(inklabels_data_path) || reload
        scans = SharedArray{Float32, 4}(scans_data_path, (512, 512, 65, num_patches))
        inklabels = SharedArray{Float32, 4}(inklabels_data_path, (512, 512, 1, num_patches))

        @info "Loading from tif files..."

        scans_index = 1
        inklabels_index = 1

        for i in 1:3
            GC.gc()
            # inklabels
            ink = load(joinpath("train", "$i", "inklabels.png")) .|> Gray .|> Float32 |> pad_to_512
            @showprogress "loading inklabels for scan $i" for j in 1:512:size(ink, 1)
                for k in 1:512:size(ink, 2)
                    inklabels[:, :, :, inklabels_index] = ink[j:j + 512 - 1, k:k + 512 - 1]
                    inklabels_index += 1
                end
            end

            # scans
            files = readdir(joinpath("train", "$i", "surface_volume"), join=true)
            layer_scans_index = nothing
            @showprogress "loading scans for scan $i" for (layer_index, file) in enumerate(files)
                scan = TiffImages.load(file; lazyio=false) .|> Float32 |> pad_to_512
                layer_scans_index = scans_index
                for j in 1:512:size(scan, 1)
                    for k in 1:512:size(scan, 2)
                        scans[:, :, layer_index, layer_scans_index] = scan[j:j + 512 - 1, k:k + 512 - 1]
                        layer_scans_index += 1
                    end
                end
            end
            @info "from $scans_index to $layer_scans_index"
            scans_index = layer_scans_index
        end

        GC.gc()

        return scans, inklabels
    else
        scans = SharedArray{Float32, 4}(scans_data_path, (512, 512, 65, num_patches))
        inklabels = SharedArray{Float32, 4}(inklabels_data_path, (512, 512, 1, num_patches))
        
        return scans, inklabels
    end
end

end # module DataUtils
