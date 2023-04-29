using Flux
using Metalhead

new_convnext3d() = Chain(
    Chain(
        # 3 layers
        Conv((4, 4, 4), 1 => 96, stride=4, bias=false),
        Metalhead.Layers.ChannelLayerNorm(Flux.Scale(1, 1, 1, 96, relu), 1.0f-6),
        # 3 layer conv 7, 96 => 96, dense 96 => 384, 384 => 96
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 96 => 96, pad=3, groups=96),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(96),
                Chain(
                    Dense(96 => 384, gelu),
                    Dropout(0.0),
                    Dense(384 => 96),
                    Dropout(0.0),
                ),
                Flux.Scale(96; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 96 => 96, pad=3, groups=96),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(96),
                Chain(
                    Dense(96 => 384, gelu),
                    Dropout(0.0),
                    Dense(384 => 96),
                    Dropout(0.0),
                ),
                Flux.Scale(96; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 96 => 96, pad=3, groups=96),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(96),
                Chain(
                    Dense(96 => 384, gelu),
                    Dropout(0.0),
                    Dense(384 => 96),
                    Dropout(0.0),
                ),
                Flux.Scale(96; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        Metalhead.Layers.ChannelLayerNorm(Flux.Scale(1, 1, 1, 96), 1.0f-6),
        Conv((2, 2, 2), 96 => 192, relu, stride=2, bias=false),
        # 3 layer conv 7, 192 => 192, dense 192 => 768, 768 => 192
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 192 => 192, pad=3, groups=192),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(192),
                Chain(
                    Dense(192 => 768, gelu),
                    Dropout(0.0),
                    Dense(768 => 192),
                    Dropout(0.0),
                ),
                Flux.Scale(192; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 192 => 192, pad=3, groups=192),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(192),
                Chain(
                    Dense(192 => 768, gelu),
                    Dropout(0.0),
                    Dense(768 => 192),
                    Dropout(0.0),
                ),
                Flux.Scale(192; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 192 => 192, pad=3, groups=192),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(192),
                Chain(
                    Dense(192 => 768, gelu),
                    Dropout(0.0),
                    Dense(768 => 192),
                    Dropout(0.0),
                ),
                Flux.Scale(192; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        Metalhead.Layers.ChannelLayerNorm(Flux.Scale(1, 1, 1, 192), 1.0f-6),
        Conv((2, 2, 2), 192 => 384, relu, stride=2, bias=false),
        # 9 layer conv 7, 384 => 384, dense 384 => 1536, 1536 => 384
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection( 
            Chain(
                Conv((7, 7, 7), 384 => 384, pad=3, groups=384),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(384),
                Chain(
                    Dense(384 => 1536, gelu),
                    Dropout(0.0),
                    Dense(1536 => 384),
                    Dropout(0.0),
                ),
                Flux.Scale(384; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        Metalhead.Layers.ChannelLayerNorm(Flux.Scale(1, 1, 1, 384), 1.0f-6),
        Conv((2, 2, 2), 384 => 768, relu, stride=2, bias=false),
        # 3 layer conv 3, 768 => 768, dense 768 => 3072, 3072 => 768
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 768 => 768, pad=3, groups=768),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(768),
                Chain(
                    Dense(768 => 3072, gelu),
                    Dropout(0.0),
                    Dense(3072 => 768),
                    Dropout(0.0),
                ),
                Flux.Scale(768; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 768 => 768, pad=3, groups=768),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(768),
                Chain(
                    Dense(768 => 3072, gelu),
                    Dropout(0.0),
                    Dense(3072 => 768),
                    Dropout(0.0),
                ),
                Flux.Scale(768; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
        SkipConnection(
            Chain(
                Conv((7, 7, 7), 768 => 768, pad=3, groups=768),
                Base.Fix2(permutedims, (4, 1, 2, 3, 5)), # to channel first
                LayerNorm(768),
                Chain(
                    Dense(768 => 3072, gelu),
                    Dropout(0.0),
                    Dense(3072 => 768),
                    Dropout(0.0),
                ),
                Flux.Scale(768; bias=false),
                Base.Fix2(permutedims, (2, 3, 4, 1, 5)), # return to channel second to last
                Dropout(0.0, dims=4),
            ),
            +,
        ),
    ),
    Chain(
        GlobalMeanPool(),
        Flux.flatten,
        LayerNorm(768),
        Dense(768 => 1),
        sigmoid_fast
    ),
)