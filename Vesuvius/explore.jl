using Colors
using MosaicViews
using Images

include("data_utils.jl")

scans, inklabels = DataUtils.load_patches();

scan1_pres = (16, 13)
scan2_pres = (29, 19)
scan3_pres = (15, 11)

scan1 = @view scans[:, :, :, 1:prod(scan1_pres)];
scan2 = @view scans[:, :, :, prod(scan1_pres) + 1:prod(scan1_pres) + prod(scan2_pres)];
scan3 = @view scans[:, :, :, prod(scan1_pres) + prod(scan2_pres) + 1:prod(scan1_pres) + prod(scan2_pres) + prod(scan3_pres)];

ink1 = @view inklabels[:, :, :, 1:prod(scan1_pres)];
ink2 = @view inklabels[:, :, :, prod(scan1_pres) + 1:prod(scan1_pres) + prod(scan2_pres)];
ink3 = @view inklabels[:, :, :, prod(scan1_pres) + prod(scan2_pres) + 1:prod(scan1_pres) + prod(scan2_pres) + prod(scan3_pres)];

i = 1
begin
    @show i = i + 1
    hcat(
        mosaicview(scan2[:, :, i, :] .|> Gray, nrow = scan2_pres[1], ncol = scan2_pres[2], rowmajor = true),
        mosaicview(ink2[:, :, 1, :] .|> Gray, nrow = scan2_pres[1], ncol = scan2_pres[2], rowmajor = true)
    ) 
end

GC.gc()
