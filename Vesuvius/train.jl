using Colors
using Flux
using CUDA
using Metalhead
using MosaicViews
using Images

include("data_utils.jl")

@assert Threads.nthreads() > 1 "remember to start julia more than 1 thread"
@assert CUDA.functional() "CUDA not available"

scans, inklabels = DataUtils.load_patches();

# backbones
# Metalhead.backbone(Metalhead.ResNet(18; inchannels = 65));
# Metalhead.backbone(Metalhead.ViT(:base; inchannels = 65));

model = Metalhead.UNet((512, 512), 65, 1, Metalhead.backbone(Metalhead.ResNet(18; inchannels = 65))) |> gpu;