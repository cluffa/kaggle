module DataUtils

using TiffImages
using ProgressMeter: @showprogress
using Colors: Float32, Gray
using Flux: pad_constant
using SharedArrays
using Images

"""
    pad_to(array)


Pads the given array to be a multiple of n in the first two dimensions.
"""
function pad_to(array, n)
    ndims = length(size(array))
    multiple = (n, n, fill(1, ndims - 2)...)

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
splits them into patch_sizexpatch_size patches, loads into a SharedArray, and returns the SharedArray.
"""
function load_patches(;reload = false)
    scans_data_path = abspath("scans.dat")
    inklabels_data_path = abspath("inklabels.dat")
    patch_size = 128
    
    num_patches = 0
    for i in 1:3
        s = load(joinpath("train", "$i", "inklabels.png")) |> size
        num_patches += s .|> (x -> x / patch_size) .|> ceil .|> Int |> prod
    end


    # run if files don't exist or reload is true
    if !isfile(scans_data_path) || !isfile(inklabels_data_path) || reload
        scans = SharedArray{Float32, 4}(scans_data_path, (patch_size, patch_size, 65, num_patches))
        inklabels = SharedArray{Float32, 4}(inklabels_data_path, (patch_size, patch_size, 1, num_patches))

        @info "Loading from tif files..."

        scans_index = 1
        inklabels_index = 1

        for i in 1:3
            GC.gc()
            # inklabels
            ink = load(joinpath("train", "$i", "inklabels.png")) .|> Gray .|> Float32 |> (x -> pad_to(x, patch_size))
            @showprogress "loading inklabels for scan $i" for j in 1:patch_size:size(ink, 1)
                for k in 1:patch_size:size(ink, 2)
                    inklabels[:, :, :, inklabels_index] = ink[j:j + patch_size - 1, k:k + patch_size - 1]
                    inklabels_index += 1
                end
            end

            # scans
            files = readdir(joinpath("train", "$i", "surface_volume"), join=true)
            layer_scans_index = nothing
            @showprogress "loading scans for scan $i" for (layer_index, file) in enumerate(files)
                scan = TiffImages.load(file; lazyio=false) .|> Float32 |>  (x -> pad_to(x, patch_size))
                layer_scans_index = scans_index
                for j in 1:patch_size:size(scan, 1)
                    for k in 1:patch_size:size(scan, 2)
                        scans[:, :, layer_index, layer_scans_index] = scan[j:j + patch_size - 1, k:k + patch_size - 1]
                        layer_scans_index += 1
                    end
                end

                scan = nothing
                GC.gc()
            end
            @info "from $scans_index to $layer_scans_index"
            scans_index = layer_scans_index
        end

        GC.gc()

        return scans, inklabels
    else
        scans = SharedArray{Float32, 4}(scans_data_path, (patch_size, patch_size, 65, num_patches))
        inklabels = SharedArray{Float32, 4}(inklabels_data_path, (patch_size, patch_size, 1, num_patches))
        
        return scans, inklabels
    end
end

end # module DataUtils
