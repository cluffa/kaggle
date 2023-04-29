# function patch_5x5()
#     rows = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 1],
#          [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 1],
#          [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1],
#          [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 1],
#          [1, 1, 0, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 1],
#          [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]

#     mats = [BitMatrix([i j k l m]') for i in rows for j in rows for k in rows for l in rows for m in rows]

#     patch_label = Dict(mat => i - 1 for (i, mat) in enumerate(mats))
#     label_patch = Dict(i - 1 => mat for (i, mat) in enumerate(mats))

#     return patch_label, label_patch
# end

# function patch_4x4()
#     rows = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0],
#          [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1],
#          [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0],
#          [1, 1, 1, 1]]

#     mats = [BitMatrix([i j k l]') for i in rows for j in rows for k in rows for l in rows]

#     patch_label = Dict(mat => i - 1 for (i, mat) in enumerate(mats))
#     label_patch = Dict(i - 1 => mat for (i, mat) in enumerate(mats))

#     return patch_label, label_patch
# end

function patch_3x3()
    rows = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
         [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    mats = [BitMatrix([i j k]') for i in rows for j in rows for k in rows]

    patch_label = Dict(mat => UInt16(i - 1) for (i, mat) in enumerate(mats))
    label_patch = Dict(UInt16(i - 1) => mat for (i, mat) in enumerate(mats))

    return patch_label, label_patch
end

patch_3x3()

patch_label, label_patch = patch_3x3()

# returns a 2d array of patches, where each patch represents a 3x3 matrix
function patchify(mat, patch_label, T = UInt16)
    # get size of patches
    ps = size(first(keys(patch_label)))[1]

    h = Int(ceil(size(mat, 1) / ps))
    w = Int(ceil(size(mat, 2) / ps))

    patches = Array{T}(undef, h, w)
    # run 3x3 window over mat
    for i in 1:h
        for j in 1:w
            # get 3x3 window
            i_range = ps * (i - 1) + 1:ps * i
            j_range = ps * (j - 1) + 1:ps * j

            mat_pad = PaddedView(0, mat, (1:h*ps, 1:w*ps))

            @info "ranges" i_range j_range
            window = mat_pad[i_range, j_range]
            # get patch label
            patches[i, j] = patch_label[window]
        end
    end
    
    return patches
end

function unpatchify(patches, label_patch, orig_size)
    ps = size(first(values(label_patch)))[1]

    h = size(patches, 1)
    w = size(patches, 2)

    mat = Array{Bool}(undef, h * ps, w * ps)

    @inbounds for i in 1:h
        for j in 1:w
            i_range = ps * (i - 1) + 1:ps * i
            j_range = ps * (j - 1) + 1:ps * j

            mat[i_range, j_range] = label_patch[patches[i, j]]
        end
    end

    return mat[1:orig_size[1], 1:orig_size[2]]
end


# test

# rand_size = rand(1:100, 2) |> Tuple
# mask = BitMatrix(rand(0:1, rand_size))
# patches = patchify(mask, patch_label)
# unpatched = unpatchify(patches, label_patch, rand_size)
# @assert mask == unpatched
