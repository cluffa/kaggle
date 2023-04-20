using Flux
using CUDA
using TiffImages
using Images
using SharedArrays

# Load the Images
files = readdir("train/3/surface_volume/", join=true)

mask = load("train/3/inklabels.png") .|> x -> x.r > 0.5;
scans = SharedArray{N0f16}(abspath("scans_N0f16.dat"), (size(mask)..., length(files)));

# for (i, file) in enumerate(files)
#     @info "loading" file
#     img = load(file) |> channelview
#     scans[:,:,i] .= img[:,:,1]
# end

function increment_fn()
    i = 0
    return () -> begin
        i += 1
        if i > length(files)
            i = 1
        end
        return i
    end
end

i = increment_fn()


@view(scans[:, :, i()]) .|> Gray |> display