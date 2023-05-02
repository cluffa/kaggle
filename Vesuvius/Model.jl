using Metalhead
using CUDA
using Flux

function new_model(; in_channels = 65, res = (256, 256))
    ch = [16, 16, 16, 16, 16]
    middle_channels = 16

    return Chain([
        # convert to 3D, add channel dimension, treat inchannels as depth
        Base.Fix2(reshape, (res..., in_channels, 1, :)),
        Conv((3, 3, 3), 1 => ch[1], pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        Conv((3, 3, 3), ch[1] => ch[2], pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        Conv((3, 3, 3), ch[2] => ch[3], pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        Conv((3, 3, 3), ch[3] => ch[4], pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        Conv((3, 3, 3), ch[4] => ch[5], pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        Conv((3, 3, 3), ch[5] => middle_channels, pad=(1, 1, 1), relu),
        MaxPool((1, 1, 2)),
        # convert back to 2D
        Base.Fix2(reshape, (res..., middle_channels, :)),
        Metalhead.unet(
            Metalhead.backbone(DenseNet(121; inchannels = middle_channels)),
            (res..., middle_channels, 1),
            1
        ).layers...
    ])
end

# model = new_model() |> gpu;
# x = CUDA.rand(Float32, (256, 256, 65, 9));
# CUDA.@time y = x |> model |> size
