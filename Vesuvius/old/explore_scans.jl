### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2b5c216e-ded2-11ed-2ec2-a9bb6be43216
begin
	using Pkg
	Pkg.activate(".")
	using Images
	using SharedArrays
	using PlutoUI
	using MosaicViews

	files = readdir("train/3/surface_volume/", join=true);
	mask = load("train/3/inklabels.png") .|> x -> x.r > 0.5;
	scans = SharedArray{N0f16}(abspath("scans_N0f16.dat"), (size(mask)..., length(files)));
end;

# ╔═╡ 159cfa5b-cafe-48cc-a553-59eafadf3a66
md"""
### Front Layer: $(@bind i Slider(eachindex(axes(scans, 3)), default = rand(eachindex(axes(scans, 3))), show_value = true))
### Top Layer: $(@bind k Slider(eachindex(axes(scans, 1)), default = rand(eachindex(axes(scans, 1))), show_value = true))
### Side Layer: $(@bind j Slider(eachindex(axes(scans, 2)), default = rand(eachindex(axes(scans, 2))), show_value = true))
"""

# ╔═╡ eb85b79e-cf60-4419-a728-78169378471b
md"""
### Alt view thickness ratio:
$(@bind thick_ratio Slider(1:20, default = 10, show_value = true))
"""

# ╔═╡ c19e6f94-2644-4546-8a2b-e73c3c102ea4
buffer = fill(Gray(N0f16(0)), 65 * thick_ratio, 65 * thick_ratio);

# ╔═╡ 2286cbde-9863-474f-9ed4-5c930b493ec5
md"""
### Downscale Ratio:
$(@bind downscale_ratio Slider(0.0:0.05:1.0, default = 0.5, show_value = true))
"""

# ╔═╡ 6c80bf60-c508-4632-909a-4b0e79355cdf
frontview = @view(scans[:, :, i]) |>
	img -> imresize(img, ratio = (downscale_ratio, downscale_ratio)) .|>
	Gray;

# ╔═╡ 65e4f6f7-7bfe-4534-9ba1-b86554c360b9
topview = @view(scans[k, :, :]) |>
	transpose |>
	img -> imresize(img, ratio = (thick_ratio, downscale_ratio)) .|>
	Gray;

# ╔═╡ ed5963fa-a741-45cd-90fb-9b343eaecab4
sideview = @view(scans[:, j, :]) |>
	img -> imresize(img, ratio = (downscale_ratio, thick_ratio)) .|>
	Gray;

# ╔═╡ f5dbbb8e-ea0d-4e44-ae51-7d3dea6c7f4e
vcat(
	hcat(frontview, sideview),
	hcat(topview, buffer)
)

# ╔═╡ Cell order:
# ╟─159cfa5b-cafe-48cc-a553-59eafadf3a66
# ╠═f5dbbb8e-ea0d-4e44-ae51-7d3dea6c7f4e
# ╠═c19e6f94-2644-4546-8a2b-e73c3c102ea4
# ╠═eb85b79e-cf60-4419-a728-78169378471b
# ╠═2286cbde-9863-474f-9ed4-5c930b493ec5
# ╠═6c80bf60-c508-4632-909a-4b0e79355cdf
# ╠═65e4f6f7-7bfe-4534-9ba1-b86554c360b9
# ╠═ed5963fa-a741-45cd-90fb-9b343eaecab4
# ╠═2b5c216e-ded2-11ed-2ec2-a9bb6be43216
